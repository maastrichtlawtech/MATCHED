"""
Python version: 3.10
Description: Fine-tunes pre-trained CLIP, CLIPITM, or BLIP2 models for Multimodal authorship identification task using VENDOR labels.
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import lightning as L
import lightning.pytorch as pl
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from transformers import AutoTokenizer, ViTImageProcessor

# Custom library
sys.path.append('../../process/')
from utilities import map_images_with_text_for_clip_model
from loadData import FineTuneCLIPstyleModelDataset

import warnings
warnings.filterwarnings('ignore')

# Suppress TorchDynamo errors and fall back to eager execution
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# %% Setting up the Argparser
parser = argparse.ArgumentParser(description="Fine-tunes pre-trained models for authorship attribution using VENDOR labels.")
parser.add_argument('--logged_entry_name', type=str, default="finetune-authorship-attribution", help="Logged entry name visible on Weights & Biases")
parser.add_argument('--data_dir', type=str, default='/workspace/persistent/HTClipper/data/processed', help="Data directory")
parser.add_argument('--image_dir', type=str, default="/workspace/persistent/HTClipper/data/IMAGES", help="Image directory")
parser.add_argument('--save_dir', type=str, default="/workspace/persistent/HTClipper/models/grouped-and-masked/multimodal-baselines/classification/finetuned", help="Directory for models to be saved")
parser.add_argument('--model_dir_name', type=str, default=None, help="Save the model with the folder name as mentioned.")
parser.add_argument('--pretrained_model_dir', type=str, default="/workspace/persistent/HTClipper/models/grouped-and-masked/multimodal-baselines/pre-training/BLIP2/non-associated/seed:1111/lr-0.0001/NTXENT/0.1/negatives-5", help="Directory of pre-trained text-image alignment model")
parser.add_argument('--geography', type=str, default='south', help="""geography of data, can be only between chicago, atlanta, houston, dallas, detroit, ny, sf or all""")
parser.add_argument('--model_type', type=str, default="BLIP2", help="Can be between CLIP, BLIP2, and CLIPITM")
parser.add_argument('--batch_size', type=int, default=32, help="Batch Size")
parser.add_argument('--nb_epochs', type=int, default=20, help="Number of Epochs")
parser.add_argument('--patience', type=int, default=3, help="Patience for Early Stopping")
parser.add_argument('--seed', type=int, default=1111, help='Random seed value')
parser.add_argument('--warmup_steps', type=int, default=0, help="Warmup steps for learning rate scheduler")
parser.add_argument('--grad_steps', type=int, default=1, help="Gradient accumulating step")
parser.add_argument('--learning_rate', type=float, default=1e-5, help="Learning rate")
parser.add_argument('--adam_epsilon', type=float, default=1e-6, help="Epsilon value for Adam optimizer")
parser.add_argument('--min_delta_change', type=float, default=0.01, help="Minimum change in delta in validation loss for Early Stopping")
parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay")
parser.add_argument('--train_data_percentage', type=float, default=1.0, help="Percentage of training data to be used")
parser.add_argument('--loss', type=str, default='CE', help='Loss function to use. Can be CE or CE+SupCon')
parser.add_argument('--finetune_mode', type=str, default='all', help='Finetuning mode: "all" or "finetune_layers"')
parser.add_argument('--layers_to_finetune', nargs='+', type=int, default=[], help='List of layer indices to fine-tune if finetune_mode is "finetune_layers"')
parser.add_argument("--extract_representation_from", type=str, default="CLS", help='Token to extract representations from: "CLS" or "EOS"')
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
torch.set_float32_matmul_precision("high")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

assert args.loss in ["CE", "CE+SupCon"]
assert args.extract_representation_from in ["CLS", "EOS"]
assert args.model_type in ["CLIP", "CLIPITM", "BLIP2"]
assert args.finetune_mode in ["all", "finetune_layers"]
assert args.geography in ["chicago", "atlanta", "dallas", "detroit", "houston", "sf", "ny", "all", "midwest", "northeast", "south", "west"]

# Creating directories
if args.model_dir_name is None:
    directory = os.path.join(args.save_dir, args.model_type, f"finetune-{args.finetune_mode}", f"representations_{args.extract_representation_from}", "seed:" + str(args.seed), "lr-" + str(args.learning_rate), f"temp-{args.temp}", args.loss)
else:
    directory = os.path.join(args.save_dir, args.model_type, args.model_dir_name, f"finetune-{args.finetune_mode}", f"representations_{args.extract_representation_from}", "seed:" + str(args.seed), "lr-" + str(args.learning_rate), f"temp-{args.temp}", args.loss)

Path(directory).mkdir(parents=True, exist_ok=True)
Path(args.save_dir).mkdir(parents=True, exist_ok=True)

# %% Load your DataFrame
df = pd.read_csv(os.path.join(args.data_dir, f"{args.geography}.csv"))
df['region'] = 'south'

# mapping every image to it's corresponding text
df = map_images_with_text_for_clip_model(df, img_dir=args.image_dir).drop_duplicates()

# Identify and keep vendors with at least 2 instances
class_counts = df['VENDOR'].value_counts()
valid_classes = class_counts[class_counts >= 2].index
df_filtered = df[df['VENDOR'].isin(valid_classes)]

# Re-encode labels after filtering
label_encoder = LabelEncoder()
df_filtered = df_filtered[["TEXT", "IMAGES", "region", "VENDOR"]].drop_duplicates()
df_filtered['VENDOR'] = label_encoder.fit_transform(df_filtered['VENDOR'])

# Split the data into train, validation, and test sets without mapping images to text yet
train_df, test_df = train_test_split(
    df_filtered, test_size=0.2, random_state=args.seed, stratify=df_filtered['VENDOR'], shuffle=True
)

# Adjust the validation split size based on the number of unique vendors
min_val_size = len(df_filtered['VENDOR'].unique()) / len(train_df)
val_size = max(0.05, min_val_size)  # Choose a larger value if needed, e.g., 0.05 or 5%

train_df, val_df = train_test_split(
    train_df, test_size=val_size, random_state=args.seed, stratify=train_df['VENDOR']
)

# Replacing all the numbers in the training dataset with the letter "N"
train_df['TEXT'] = train_df['TEXT'].apply(lambda x: re.sub(r'\d', 'N', str(x)))

# %% Initialize the tokenizers and models
text_tokenizer = AutoTokenizer.from_pretrained('johngiorgi/declutr-small')
image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

# % Loading the Dataset
train_dataset = FineTuneCLIPstyleModelDataset(train_df, text_tokenizer, image_processor)
val_dataset = FineTuneCLIPstyleModelDataset(val_df, text_tokenizer, image_processor)
test_dataset = FineTuneCLIPstyleModelDataset(test_df, text_tokenizer, image_processor)

# Create the DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

print(f"Number of samples in the train dataset: {len(train_dataloader.dataset)}")
print(f"Number of samples in the val dataset: {len(val_dataloader.dataset)}")
print(f"Number of samples in the test dataset: {len(test_dataloader.dataset)}")

num_training_steps = args.nb_epochs * (len(train_dataloader.dataset) // args.batch_size)
# Setting the warmup steps to 1/10th the size of training data
warmup_steps = int(0.1 * num_training_steps)
num_classes = len(label_encoder.classes_)

# %% Loading Model
if args.model_type == "CLIP":
    sys.path.append("../../architectures/")
    from CLIPLayer import CLIPModel, FineTuneCLIPClassifier

    # Initialize the pre-trained model
    model = CLIPModel(weight_decay=args.weight_decay, eps=args.adam_epsilon, warmup_steps=warmup_steps, num_training_steps=num_training_steps)
    checkpoint = torch.load(os.path.join(args.pretrained_model_dir, "final_model.ckpt"), map_location=device)
    # Load the state dictionary into the model
    model.load_state_dict(checkpoint['state_dict'])

    # Loading the classifier
    model = FineTuneCLIPClassifier(pretrained_model=model, finetune_mode=args.finetune_mode, num_classes=num_classes, weight_decay=args.weight_decay, eps=args.adam_epsilon, 
                                    warmup_steps=warmup_steps, num_training_steps=num_training_steps, learning_rate=args.learning_rate, loss_fn=args.loss,
                                    temperature=args.temp, extract_representation_from=args.extract_representation_from)

elif args.model_type == "CLIPITM":
    sys.path.append("../../architectures/")
    from CLIPITMLayer import CLIPITMModel, FineTuneCLIPITMClassifier
    # Initialize the pre-trained model
    model = CLIPITMModel(weight_decay=args.weight_decay, eps=args.adam_epsilon, warmup_steps=warmup_steps, num_training_steps=num_training_steps)
    checkpoint = torch.load(os.path.join(args.pretrained_model_dir, "final_model.ckpt"), map_location=device)
    # Load the state dictionary into the model
    model.load_state_dict(checkpoint['state_dict'])

    # Loading the classifier
    model = FineTuneCLIPITMClassifier(pretrained_model=model, finetune_mode=args.finetune_mode, num_classes=num_classes, weight_decay=args.weight_decay, eps=args.adam_epsilon, 
                                    warmup_steps=warmup_steps, num_training_steps=num_training_steps, learning_rate=args.learning_rate, loss_fn=args.loss,
                                    temperature=args.temp)

else:
    sys.path.append("../../architectures/")
    from BLIP2Layer import BLIP2Model, FineTuneBLIP2Classifier
    # Initialize the pre-trained model
    model = BLIP2Model(weight_decay=args.weight_decay, eps=args.adam_epsilon, warmup_steps=warmup_steps, num_training_steps=num_training_steps)
    checkpoint = torch.load(os.path.join(args.pretrained_model_dir, "final_model.ckpt"), map_location=device)
    # Load the state dictionary into the model
    model.load_state_dict(checkpoint['state_dict'])

    # Loading the classifier
    model = FineTuneBLIP2Classifier(pretrained_model=model, finetune_mode=args.finetune_mode, num_classes=num_classes, weight_decay=args.weight_decay, eps=args.adam_epsilon, 
                                    warmup_steps=warmup_steps, num_training_steps=num_training_steps, learning_rate=args.learning_rate, loss_fn=args.loss,
                                    temperature=args.temp)

# %% Loading the trainer
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=args.min_delta_change, patience=args.patience, verbose=False, mode="min")
wandb_logger = WandbLogger(save_dir=os.path.join(directory, "logs"), name=args.logged_entry_name, project="Multimodal-Baselines")

trainer = L.Trainer(
    max_epochs=args.nb_epochs,
    accelerator="gpu",
    fast_dev_run=False,
    accumulate_grad_batches=args.grad_steps,
    benchmark=True,
    deterministic=True,
    limit_train_batches=args.train_data_percentage,
    check_val_every_n_epoch=1,
    callbacks=[early_stop_callback],
    logger=wandb_logger,
    precision='16-mixed', # Mixed Precision system
    # limit_val_batches=0.25,   # Use only 10% of the validation data
    # limit_test_batches=10,    # Use only 5 batches from the test data
)

# %% Training model
start_time = time.time()
trainer.fit(model, train_dataloader, val_dataloader)
print("Total training:", str(datetime.timedelta(seconds=time.time() - start_time)))
trainer.save_checkpoint(os.path.join(directory, "final_model.ckpt"))
torch.save(model.state_dict(), os.path.join(directory, "final_model.model"))

print("Test data performance:")
trainer.test(model=model, dataloaders=test_dataloader)