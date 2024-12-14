"""
Python version: 3.10
Description: Trains an authorship identification or verification with Contrastive loss for text-based Authorship tasks on Backpage advertisements.
"""
# %% Importing Libraries
import os
import sys
import pickle
import argparse
import time
import datetime
import random
from pathlib import Path

import pandas as pd
import numpy as np

import torch

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

import lightning as L
import lightning.pytorch as pl
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
# from lightning.pytorch.strategies import DeepSpeedStrategy
# from lightning.pytorch.plugins.precision import DeepSpeedPrecisionPlugin

# Custom library
sys.path.append('../../process/')
from loadData import HTContraDataModule

import warnings
warnings.filterwarnings('ignore')

# %% Setting up the Argparser
parser = argparse.ArgumentParser(description="Trains a transformers based classifier or a Semi-Supervised model with Contrastive loss for text-based Authorship tasks on Backpage advertisements.")
parser.add_argument('--model_name_or_path', type=str, default="johngiorgi/declutr-small", help="Name of the model to be trained (can only be between distilbert-base-cased)")
parser.add_argument('--tokenizer_name_or_path', type=str, default="johngiorgi/declutr-small", help="Name of the tokenizer to be used (can only be between distilbert-base-cased)")
parser.add_argument('--logged_entry_name', type=str, default="declutr-small-contra-loss:CE+SupCon-seed:1111", help="Logged entry name visible on weights and biases")
parser.add_argument('--data_dir', type=str, default='/workspace/persistent/HTClipper/data/processed/', help="""Data directory""")
parser.add_argument('--geography', type=str, default='south', help="""geography of data, can be only between midwest, northeast, south, west""")
parser.add_argument('--save_dir', type=str, default="/workspace/persistent/HTClipper/models/grouped-and-masked/text-baselines/contra-learn/semi-supervised", help="""Directory for models to be saved""")
parser.add_argument('--loss1_type', type=str, default="CE", help="Can be None or CE, only used for classification task")
parser.add_argument('--loss2_type', type=str, default="SupCon-negatives", help="Can be KL, infoNCE, infoNCE-negatives, SupCon, triplet, or SupCon-negatives, only used for classification task")
parser.add_argument('--loss_type', type=str, default="SupCon", help="Can be SupCon or triplet, only used for the semi-supervised task")
parser.add_argument('--coefficient', type=float, default=1.0, help="Lambda coefficient to balance loss1 and loss2")
parser.add_argument('--task', type=str, default="classification", help="can be classification of semi-supervised")
parser.add_argument('--batch_size', type=int, default=32, help="Batch Size")
parser.add_argument('--nb_epochs', type=int, default=40, help="Number of Epochs")
parser.add_argument('--max_seq_length', type=int, default=512, help="Maximum sequence length")
parser.add_argument('--sample_unit_size', type=int, default=2, help="Maximum sample size")
parser.add_argument('--emb_len', type=int, default=768, help="Embedding size")
parser.add_argument('--hidden_dim', type=int, default=512, help="Hidden dimensions")
parser.add_argument('--num_hard_negatives', type=int, default=5, help=" Number of in-batch hard negatives")
parser.add_argument('--patience', type=int, default=5, help="Patience for Early Stopping")
parser.add_argument('--seed', type=int, default=1111, help='Random seed value')
parser.add_argument('--warmup_steps', type=int, default=0, help="Warmup proportion")
parser.add_argument('--grad_steps', type=int, default=4, help="Gradient accumulating step")
parser.add_argument('--learning_rate', type=float, default=0.0001, help="learning rate")
parser.add_argument('--dropout', type=float, default=0.3, help="Dropout")
parser.add_argument('--train_data_percentage', type=float, default=1.0, help="Percentage of training data to be used")
parser.add_argument('--adam_epsilon', type=float, default=1e-6, help="Epsilon value for adam optimizer")
parser.add_argument('--min_delta_change', type=float, default=0.5, help="Minimum change in delta in validation loss for Early Stopping")
parser.add_argument('--temp', type=float, default=0.1, help="Tempertaure variable for the loss function")
parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay")
parser.add_argument('--nb_triplets', type=int, default=5, help="number of in-batch triplets")
parser.add_argument('--pooling', type=bool, default=True, help="Performs mean pooling on the CLS token, if turned off, the representations in the last layer of the model will be flattened and used.")
args = parser.parse_args()

# Setting seed value for reproducibility    
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
seed_everything(args.seed)

assert args.geography in ["midwest", "northeast", "south", "west"]
assert args.loss_type in ["SupCon", "triplet"]
assert args.task in ["classification", "semi-supervised"]
assert args.loss1_type in ["None", "CE"]
assert args.loss2_type in ["KL", "infoNCE", "infoNCE-negatives", "SupCon", "SupCon-negatives", "triplet"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Creating directories
directory = os.path.join(args.save_dir, args.model_name_or_path.split("/")[-1], args.geography, "pooled", "seed:" + str(args.seed), "lr-" + str(args.learning_rate), "coeff-" + str(args.coefficient), "temp:" + str(args.temp), args.loss_type)
Path(directory).mkdir(parents=True, exist_ok=True)

# Loading data
dm = HTContraDataModule(file_dir=os.path.join(args.data_dir, args.geography + '.csv'), tokenizer_name_or_path=args.tokenizer_name_or_path, seed=args.seed, train_batch_size=args.batch_size, 
                        eval_batch_size=args.batch_size)
dm.setup(stage="fit")

args.num_classes = pd.read_csv(os.path.join(args.data_dir, args.geography + '.csv')).VENDOR.nunique()
args.num_training_steps = len(dm.train_dataloader()) * args.nb_epochs
# Setting the warmup steps to 1/10th the size of training data
args.warmup_steps = int(len(dm.train_dataloader()) * 10/100)

# %% Loading the model
if args.task == "classification":
    sys.path.append('../../architectures/')
    from ContraLayer import HTContraClassifierModel
    model = HTContraClassifierModel(args)
else:
    sys.path.append('../../architectures/')
    from ContraLayer import SemiConstrativeTextModel
    model = SemiConstrativeTextModel(args)
model.to(device)

early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=args.min_delta_change, patience=args.patience, verbose=False, mode="min")
model_checkpoint = ModelCheckpoint(dirpath=directory, filename="{epoch}-{step}-{val_loss:2f}", save_last=True, save_top_k=3, monitor="val_loss", mode="min", verbose=True)
wandb_logger = WandbLogger(save_dir=os.path.join(directory, "logs"), name=args.logged_entry_name, project="Multimodal-textBaselines")

# %% Setting up the trainer
trainer = L.Trainer(max_epochs=args.nb_epochs, accelerator="gpu", fast_dev_run=False, 
                    accumulate_grad_batches = args.grad_steps, # To run the backward step after n batches, helps to increase the batch size
                    benchmark = True, # Fastens the training process
                    deterministic=True, # Ensures reproducibility 
                    limit_train_batches=args.train_data_percentage, # trains on 10% of the data,
                    check_val_every_n_epoch = 1, # run val loop every 1 training epochs
                    # callbacks=[model_checkpoint, early_stop_callback], # Enables model checkpoint and early stopping
                    callbacks=[early_stop_callback],
                    logger = wandb_logger,
                    # strategy=DeepSpeedStrategy(stage=3, offload_optimizer=True, offload_parameters=True, offload_params_device='cpu'), # Enable CPU Offloading, and offload parameters to CPU
                    # plugins=DeepSpeedPrecisionPlugin(precision='16-mixed') # Mixed Precision system
                    precision='16-mixed' # Mixed Precision system
                    )

# %% Training model
start_time = time.time()
trainer.fit(model, dm)
print("Total training:", str(datetime.timedelta(seconds=time.time()-start_time)))
trainer.save_checkpoint(os.path.join(directory, "final_model.ckpt"))
torch.save(model.state_dict(), os.path.join(directory, "final_model.model"))

# %% Testing model performance
# print("Train data performance:")
# trainer.test(model=model, dataloaders=dm.train_dataloader())
print("Test data performance:")
trainer.test(model=model, dataloaders=dm.test_dataloader())
# print("Validation data performance:")
# trainer.test(model=model, dataloaders=dm.val_dataloader())
