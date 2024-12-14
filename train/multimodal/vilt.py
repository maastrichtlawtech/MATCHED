"""
Python version: 3.10
Description: Trains a ViLT based classifier to establish baselines for Multimodal authorship identification task on Backpage advertisements.
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
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from transformers import ViltProcessor, ViltModel

# Custom library
sys.path.append('../../process/')
from utilities import map_images_with_text, augment_image_training_data
from loadData import ViLTMultimodalDataset

sys.path.append('../../architectures/')
from viltLayer import ViLTClassifier

import warnings
warnings.filterwarnings('ignore')

# Suppress TorchDynamo errors and fall back to eager execution
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# %% Setting up the Argparser
parser = argparse.ArgumentParser(description="Trains a ViLT based classifier to establish baselines for Multimodal Authorship tasks on Backpage advertisements.")
parser.add_argument('--logged_entry_name', type=str, default="multimodal-latent-fusion-seed:1111", help="Logged entry name visible on weights and biases")
parser.add_argument('--data_dir', type=str, default='/workspace/persistent/HTClipper/data/processed', help="""Data directory""")
parser.add_argument('--city', type=str, default='chicago', help="""Demography of data, can be only between chicago, atlanta, houston, dallas, detroit, ny, sf or all""")
parser.add_argument('--save_dir', type=str, default=os.path.join(os.getcwd(), "/workspace/persistent/HTClipper/models/grouped-and-masked/multimodal-baselines/classification/vilt/"), help="""Directory for models to be saved""")
parser.add_argument('--batch_size', type=int, default=32, help="Batch Size")
parser.add_argument('--nb_epochs', type=int, default=40, help="Number of Epochs")
parser.add_argument('--patience', type=int, default=3, help="Patience for Early Stopping")
parser.add_argument('--seed', type=int, default=1111, help='Random seed value')
parser.add_argument('--warmup_steps', type=int, default=0, help="Warmup proportion")
parser.add_argument('--grad_steps', type=int, default=1, help="Gradient accumulating step")
parser.add_argument('--learning_rate', type=float, default=6e-4, help="learning rate")
parser.add_argument('--train_data_percentage', type=float, default=1.0, help="Percentage of training data to be used")
parser.add_argument('--adam_epsilon', type=float, default=1e-6, help="Epsilon value for adam optimizer")
parser.add_argument('--min_delta_change', type=float, default=0.01, help="Minimum change in delta in validation loss for Early Stopping")
parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay")
parser.add_argument('--augment_data', type=bool, default=False, help='Enables data augmentation')
parser.add_argument('--nb_augmented_samples', type=int, default=1, help='Number of augmented samples to be generated')
args = parser.parse_args()

# Setting seed value for reproducibility    
seed_everything(args.seed)

# Set matrix multiplication precision for mixed precision training
torch.set_float32_matmul_precision("high")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Creating directories
save_path = os.path.join(args.save_dir, args.city, "seed:" + str(args.seed), "lr-" + str(args.learning_rate))
Path(save_path).mkdir(parents=True, exist_ok=True)

# %% Load your DataFrame
data_dir = os.path.join(args.data_dir, args.city + ".csv")
args.image_dir = os.path.join("/workspace/persistent/HTClipper/data/IMAGES", args.city, "image", "image")
df = pd.read_csv(data_dir)

# Apply map_images_with_text
train_df = map_images_with_text(df).drop_duplicates()

# Identify and keep vendors with at least 2 instances
class_counts = df['VENDOR'].value_counts()
valid_classes = class_counts[class_counts >= 2].index
df_filtered = df[df['VENDOR'].isin(valid_classes)]

# Re-encode labels after filtering
label_encoder = LabelEncoder()
df_filtered = df_filtered[["TEXT", "IMAGES", "VENDOR"]].drop_duplicates()
df_filtered['VENDOR'] = label_encoder.fit_transform(df_filtered['VENDOR'])

# Split the data into train, validation, and test sets without mapping images to text yet
train_df, test_df = train_test_split(
    df_filtered, test_size=0.2, random_state=args.seed, stratify=df_filtered['VENDOR'], shuffle=True)

# Adjust the validation split size based on the number of unique vendors
min_val_size = len(df_filtered['VENDOR'].unique()) / len(train_df)
val_size = max(0.05, min_val_size)  # Choose a larger value if needed, e.g., 0.05 or 5%

train_df, val_df = train_test_split(
    train_df, test_size=val_size, random_state=args.seed, stratify=train_df['VENDOR']
)

# Replacing all the numbers in the training dataset with the letter "N"
train_df['TEXT'] = train_df['TEXT'].apply(lambda x: re.sub(r'\d', 'N', str(x)))

# %% Load the processor and model for ViLT
vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
vilt_model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")

# Define train, validation, and test datasets and loaders
train_dataset = ViLTMultimodalDataset(train_df, vilt_processor, label_encoder, image_dir=args.image_dir, augment=args.augment_data)
val_dataset = ViLTMultimodalDataset(val_df, vilt_processor, label_encoder, image_dir=args.image_dir, augment=False)
test_dataset = ViLTMultimodalDataset(test_df, vilt_processor, label_encoder, image_dir=args.image_dir, augment=False)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

num_training_steps = args.nb_epochs * len(train_dataloader)
# Setting the warmup steps to 1/10th the size of training data
warmup_steps = int(0.1 * num_training_steps)

model = ViLTClassifier(vilt_model=vilt_model, learning_rate=args.learning_rate, num_classes=len(label_encoder.classes_), weight_decay=args.weight_decay,
                    eps=args.adam_epsilon, warmup_steps=warmup_steps, num_training_steps=num_training_steps)

# Callbacks and logger
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=args.min_delta_change, patience=args.patience, mode="min")
wandb_logger = WandbLogger(save_dir=save_path, name=args.logged_entry_name, project="Multimodal-Baselines")

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

trainer.fit(model, train_dataloader, val_dataloader)
trainer.save_checkpoint(os.path.join(save_path, "final_model.ckpt"))
torch.save(model.state_dict(), os.path.join(save_path, "final_model.model"))

# Evaluate on test set
trainer.test(model=model, dataloaders=test_dataloader)