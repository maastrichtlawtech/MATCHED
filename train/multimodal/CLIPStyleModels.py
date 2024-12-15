"""
Python version: 3.10
Description: Takes the DeCLUTR-ViT backbone and performs CLIP, CLIP-ITM, or BLIP2 pre-training for the text-image alignment tasks
"""

# %% Importing Libraries
import os
import re
import sys
import argparse
import time
import datetime
import random
from pathlib import Path
from PIL import Image

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report

import torch
from torch.utils.data import Dataset, DataLoader

from pytorch_lightning.loggers import WandbLogger

import lightning as L
import lightning.pytorch as pl
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from transformers import AutoTokenizer, ViTImageProcessor

# Custom library
sys.path.append('../../process/')
from utilities import map_images_with_text_for_clip_model

import warnings
warnings.filterwarnings('ignore')

# Suppress TorchDynamo errors and fall back to eager execution
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# %% Setting up the Argparser
parser = argparse.ArgumentParser(description="Finetunes a Declutr-small and ViT-patch16 based classifier to establish CLIP-like baselines.")
parser.add_argument('--logged_entry_name', type=str, default="multimodal-latent-fusion-seed:1111", help="Logged entry name visible on weights and biases")
parser.add_argument('--data_dir', type=str, default='/workspace/persistent/HTClipper/data/processed', help="""Data directory""")
parser.add_argument('--image_dir', type=str, default="/workspace/persistent/HTClipper/data/IMAGES", help="""Image directory""")
parser.add_argument('--save_dir', type=str, default=os.path.join(os.getcwd(), "/workspace/persistent/HTClipper/models/grouped-and-masked/multimodal-baselines/pre-training/"), help="""Directory for models to be saved""")
parser.add_argument('--model_dir_name', type=str, default=None, help="Save the model with the folder name as mentioned.")
parser.add_argument('--pairing_mode', type=str, default="non-associated", help="assoicated: The negatives are text-image pairs that are associated but come from a different region. Or non-associated: The negatives are text-image pairs that are not associated with each other, and they also come from a different region.")
parser.add_argument('--model_type', type=str, default="CLIP", help="Can be between CLIP, BLIP2, and CLIPITM")
parser.add_argument('--batch_size', type=int, default=32, help="Batch Size")
parser.add_argument('--nb_epochs', type=int, default=40, help="Number of Epochs")
parser.add_argument('--patience', type=int, default=3, help="Patience for Early Stopping")
parser.add_argument('--nb_negatives', type=int, default=1, help="Number of in-batch negatives")
parser.add_argument('--seed', type=int, default=1111, help='Random seed value')
parser.add_argument('--warmup_steps', type=int, default=0, help="Warmup proportion")
parser.add_argument('--grad_steps', type=int, default=1, help="Gradient accumulating step")
parser.add_argument('--learning_rate', type=float, default=6e-4, help="learning rate")
parser.add_argument('--train_data_percentage', type=float, default=1.0, help="Percentage of training data to be used")
parser.add_argument('--adam_epsilon', type=float, default=1e-6, help="Epsilon value for adam optimizer")
parser.add_argument('--min_delta_change', type=float, default=0.01, help="Minimum change in delta in validation loss for Early Stopping")
parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay")
parser.add_argument('--loss', type=str, default='NTXENT', help='Loss function to use. Can be NTXENT')
parser.add_argument('--temp', type=float, default=0.5, help="Tempertaure variable for the Constrastive loss function")
args = parser.parse_args()

# Setting seed value for reproducibility    
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
# Set TOKENIZERS_PARALLELISM to false to disable parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
seed_everything(args.seed)

# Set matrix multiplication precision
# This setting offers a balance between precision and performance. It’s typically a good starting point for mixed precision training
#  with FP16.
torch.set_float32_matmul_precision("high")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

assert args.loss in ["NTXENT"]
assert args.pairing_mode in ["associated", "non-associated"]
assert args.model_type in ["CLIP", "CLIPITM", "BLIP2"]

# Creating directories
if args.model_dir_name == None:
    directory = os.path.join(args.save_dir, args.model_type, args.pairing_mode, "seed:" + str(args.seed), "lr-" + str(args.learning_rate), args.loss, str(args.temp), "negatives-" + str(args.nb_negatives))
else:
    directory = os.path.join(args.save_dir, args.model_type, args.model_dir_name, args.pairing_mode, "seed:" + str(args.seed), "lr-" + str(args.learning_rate), args.loss, str(args.temp), "negatives-" + str(args.nb_negatives))

Path(directory).mkdir(parents=True, exist_ok=True)
Path(args.save_dir).mkdir(parents=True, exist_ok=True)

# %% Load your DataFrame
south_df = pd.read_csv(os.path.join(args.data_dir, "south.csv"))
midwest_df = pd.read_csv(os.path.join(args.data_dir, "midwest.csv"))
northeast_df = pd.read_csv(os.path.join(args.data_dir, "northeast.csv"))
west_df = pd.read_csv(os.path.join(args.data_dir, "west.csv"))

# Label each dataset
south_df['region'] = 'south'
midwest_df['region'] = 'midwest'
northeast_df['region'] = 'northeast'
west_df['region'] = 'west'

# mapping every image to it's corresponding text
south_df = map_images_with_text_for_clip_model(south_df, img_dir=args.image_dir).drop_duplicates()
midwest_df = map_images_with_text_for_clip_model(midwest_df, img_dir=args.image_dir).drop_duplicates()
northeast_df = map_images_with_text_for_clip_model(northeast_df, img_dir=args.image_dir).drop_duplicates()
west_df = map_images_with_text_for_clip_model(west_df, img_dir=args.image_dir).drop_duplicates()

# Combine all datasets and split into train, validation, and test sets
df = pd.concat([south_df, midwest_df, northeast_df, west_df])

# Identify and remove classes with fewer than 2 instances
# Since we use stratify during splitting, we should atleast have one training example in training and one in test dataset
class_counts = df['VENDOR'].value_counts()
valid_classes = class_counts[class_counts >= 3].index
df_filtered = df[df['VENDOR'].isin(valid_classes)]

# Split the data into train, validation, and test sets
train_df, test_df = train_test_split(df_filtered, test_size=0.2, random_state=args.seed, stratify=df_filtered['VENDOR'])
train_df, val_df = train_test_split(train_df, test_size=0.05, random_state=args.seed, stratify=train_df['VENDOR'])

# Replacing all the numbers in the training dataset with the letter "N"
train_df['TEXT'] = train_df['TEXT'].apply(lambda x: re.sub(r'\d', 'N', str(x)))

# Augment the training data by adding multiple entries for each image
# train_df = augment_image_training_data(train_df)

# %% Intializing the tokenizers and models
# Since these are the two models that performed individually on the text and image modalities, we establish them as benchmarks and
# only run use them in our further experiments.
text_tokenizer = AutoTokenizer.from_pretrained('johngiorgi/declutr-small')
image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

# Create the datasets
if args.model_type in ["CLIP", "CLIPITM"]:
    sys.path.append('../../process/')
    from loadData import CLIPDataset    

    train_dataset = CLIPDataset(
        df = train_df,
        text_tokenizer=text_tokenizer, 
        image_processor=image_processor, 
        num_negatives=args.nb_negatives, 
        pairing_mode=args.pairing_mode
    )

    val_dataset = CLIPDataset(
        df = val_df,
        text_tokenizer=text_tokenizer, 
        image_processor=image_processor, 
        num_negatives=args.nb_negatives, 
        pairing_mode=args.pairing_mode
    )

    test_dataset = CLIPDataset(
        df = test_df,
        text_tokenizer=text_tokenizer, 
        image_processor=image_processor, 
        num_negatives=args.nb_negatives, 
        pairing_mode=args.pairing_mode
    )

else:
    sys.path.append('../../process/')
    from loadData import BLIP2Dataset   

    t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

    train_dataset = BLIP2Dataset(
        df = train_df,
        text_tokenizer=text_tokenizer,
        t5_tokenizer=t5_tokenizer, 
        image_processor=image_processor, 
        num_negatives=args.nb_negatives, 
        pairing_mode=args.pairing_mode
    )

    val_dataset = BLIP2Dataset(
        df = val_df,
        text_tokenizer=text_tokenizer, 
        t5_tokenizer=t5_tokenizer, 
        image_processor=image_processor, 
        num_negatives=args.nb_negatives, 
        pairing_mode=args.pairing_mode
    )

    test_dataset = BLIP2Dataset(
        df = test_df,
        text_tokenizer=text_tokenizer, 
        t5_tokenizer=t5_tokenizer, 
        image_processor=image_processor, 
        num_negatives=args.nb_negatives, 
        pairing_mode=args.pairing_mode
    )

# Create the DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

print(f"Number of samples in the train dataset: {len(train_dataloader.dataset)}")
print(f"Number of samples in the val dataset: {len(val_dataloader.dataset)}")
print(f"Number of samples in the test dataset: {len(test_dataloader.dataset)}")

num_training_steps = args.nb_epochs * (len(train_dataloader.dataset)/args.batch_size)
# Setting the warmup steps to 1/10th the size of training data
warmup_steps = int(0.1 * num_training_steps)

# Initialize the model
if args.model_type == "CLIP":

    sys.path.append('../../architectures/')
    from CLIPLayer import CLIPModel
    
    model = CLIPModel(
        text_model_name='johngiorgi/declutr-small', 
        image_model_name='google/vit-base-patch16-224', 
        learning_rate=args.learning_rate, 
        num_negatives=args.nb_negatives,
        weight_decay=args.weight_decay,
        eps=args.adam_epsilon,
        warmup_steps=warmup_steps,
        num_training_steps=args.nb_epochs * len(train_dataloader),
        temperature=args.temp
    )

elif args.model_type == "CLIPITM":

    sys.path.append('../../architectures/')
    from CLIPITMLayer import CLIPITMModel

    model = CLIPITMModel(
    text_model_name='johngiorgi/declutr-small', 
    image_model_name='google/vit-base-patch16-224', 
    learning_rate=args.learning_rate, 
    num_negatives=args.nb_negatives,
    weight_decay=args.weight_decay,
    eps=args.adam_epsilon,
    warmup_steps=warmup_steps,
    num_training_steps=args.nb_epochs * len(train_dataloader),
    temperature=args.temp
    )

else:
    sys.path.append('../../architectures/')
    from BLIP2Layer import BLIP2Model

    model = BLIP2Model(
    text_model_name='johngiorgi/declutr-small', 
    image_model_name='google/vit-base-patch16-224', 
    t5_model_name='google/flan-t5-small',
    learning_rate=args.learning_rate, 
    num_negatives=args.nb_negatives,
    weight_decay=args.weight_decay,
    eps=args.adam_epsilon,
    warmup_steps=warmup_steps,
    num_training_steps=args.nb_epochs * len(train_dataloader),
    temperature=args.temp
    )

# %% Loading the trainer
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=args.min_delta_change, patience=args.patience, verbose=False, mode="min")
# model_checkpoint = ModelCheckpoint(dirpath=directory, filename="{epoch}-{step}-{val_loss:2f}", save_last=True, save_top_k=3, monitor="val_loss", mode="min", verbose=True)
wandb_logger = WandbLogger(save_dir=os.path.join(directory, "logs"), name=args.logged_entry_name, project="Multimodal-Baselines")

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
                    precision='16-mixed', # Mixed Precision system
                    # limit_val_batches=0.1,   # Use only 10% of the validation data
                    # limit_test_batches=2,    # Use only 2 batches from the test data
                    # detect_anomaly=True  # This will help identify where NaNs are introduced
                    )

# %% Training model
start_time = time.time()
trainer.fit(model, train_dataloader, val_dataloader)
print("Total training:", str(datetime.timedelta(seconds=time.time()-start_time)))
trainer.save_checkpoint(os.path.join(directory, "final_model.ckpt"))
torch.save(model.state_dict(), os.path.join(directory, "final_model.model"))

print("Test data performance:")
trainer.test(model=model, dataloaders=test_dataloader)
