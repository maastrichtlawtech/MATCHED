"""
Python version: 3.10
Description: Fine-tunes image baselines for authorship identification task using VENDOR labels.
"""

# %% Loading libraries
import os
import sys
import argparse
import time
import datetime
import random

from PIL import Image
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

import torch

from sklearn.model_selection import train_test_split

import lightning.pytorch as pl
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import timm

# Custom library
sys.path.append('../../process/')
from imageUtilities import load_images_and_labels
from loadData import ImageDataModule

sys.path.append('../../architectures/')
from visionClassifierLayer import PreTrainedVisionModel

import warnings
warnings.filterwarnings('ignore')

# %% Setting up the Argparser
parser = argparse.ArgumentParser(description="Trains a image classifier to establish baselines for Authorship tasks on Backpage advertisements.")
parser.add_argument('--model_name_or_path', type=str, default="vgg16", help="Name of the model to be trained (can only be between distilbert-base-cased)")
parser.add_argument('--logged_entry_name', type=str, default="vgg16-seed:1111", help="Logged entry name visible on weights and biases")
parser.add_argument('--data_dir', type=str, default='/workspace/persistent/HTClipper/data/processed', help="""Data directory""")
parser.add_argument('--data_type', type=str, default="all", help="can be faces for the dataset with human faces, nofaces for body parts dataset, or all.")
parser.add_argument('--city', type=str, default='south', help="""Demography of data, can be only between midwest, northeast, south, west""")
parser.add_argument('--save_dir', type=str, default="/workspace/persistent/HTClipper/models/grouped-and-masked/image-baselines", help="""Directory for models to be saved""")
parser.add_argument('--model_dir_name', type=str, default=None, help="Save the model with the folder name as mentioned.")
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

# Making sure that the input variables are right
assert args.data_type in ["all"]
assert args.city in ["midwest", "northeast", "south", "west"]
assert args.model_name_or_path in ['vgg16', 'vgg19', "resnet50", "resnet101", "resnet152", "mobilenet", "mobilenetv2", "densenet121", "densenet169", 
                                "efficientnet-b0", "efficientnet-b1", "efficientnet-b2", "efficientnet-b3", "efficientnet-b4", "efficientnet-b5", "efficientnet-b6",
                                "efficientnet-b7", "efficientnetv2_rw_m", "efficientnetv2_rw_s", "efficientnetv2_rw_t", "convnext_tiny", "convnext_small", 
                                "convnext_base", "convnext_large", "convnext_xlarge", "vit_base_patch16_224", "vit_large_patch16_224", "vit_base_patch32_224", 
                                "vit_large_patch32_224", "inception_v3", "inception_resnet_v2" ]

# Creating directories
if args.model_dir_name == None:
    directory = os.path.join(args.save_dir, args.model_name_or_path.split("/")[-1], args.city, args.data_type, 
                            "seed:" + str(args.seed), "lr-" + str(args.learning_rate))
else:
    directory = os.path.join(args.save_dir, args.model_name_or_path.split("/")[-1], args.city, args.data_type, 
                            "seed:" + str(args.seed), "lr-" + str(args.learning_rate) + "-" + args.model_dir_name)
Path(directory).mkdir(parents=True, exist_ok=True)
Path(args.save_dir).mkdir(parents=True, exist_ok=True)

# %% Loading dataset
# Map city and data_type combinations to file paths
file_paths = {
    "chicago": {
        "faces": "chicago_faces.csv",
        "nofaces": "chicago_nofaces.csv",
        "all": "chicago_images.csv"
    },
    "all": {
        "faces": "all_faces.csv",
        "nofaces": "all_nofaces.csv",
        "all": "all_images.csv"
    },
    "south": {
        "all": "south_images.csv"
    }
}

# Construct the file path and read the CSV file
file_path = os.path.join(args.data_dir, file_paths[args.city][args.data_type])
df = pd.read_csv(file_path)

# Removing vendors that have less than 2 ads
vendors_of_interest = {k:v for k, v in dict(Counter(df.VENDOR)).items() if v>1}
df = df[df['VENDOR'].isin(list(vendors_of_interest.keys()))]

# Remapping new vendor ids
all_vendors = df.VENDOR.unique()
vendor_to_idx_dict = {vendor: idx for idx, vendor in enumerate(all_vendors)}
df["VENDOR"] = df["VENDOR"].replace(vendor_to_idx_dict)

num_classes = df.VENDOR.nunique()
assert df['VENDOR'].min() >= 0 and df['VENDOR'].max() < num_classes

# %% Load and preprocess images
# The target size is fixed to 224x224 for a fair comparison with the ViT models.
# Turn the augment parameter to True only if you want to perform augmentation for the entire dataset
# Otherwise, the augmentation to training data only is implemented in the ImageDataModule class
images, labels = load_images_and_labels(df, target_size=(224, 224), augment=False,
                                         num_augmented_samples=args.nb_augmented_samples)
assert images.shape[0] == labels.shape[0]

# %% Split data
# Split ratio is set to 0.20 between training and test data, and 0.05 between training and val data
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.20, random_state=1111, stratify=labels)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=1111, stratify=y_train)

# %% Instantiate DataModule and Model
if args.augment_data == True:
    data_module = ImageDataModule(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=args.batch_size, augment_data=args.augment_data, num_augmented_samples=args.nb_augmented_samples)
else:
    data_module = ImageDataModule(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=args.batch_size, augment_data=args.augment_data)

data_module.setup()

model = PreTrainedVisionModel(model_name=args.model_name_or_path, num_classes=num_classes, learning_rate=args.learning_rate)

# %%  Setting the trainer
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=args.min_delta_change, patience=args.patience, verbose=False, mode="min")
model_checkpoint = ModelCheckpoint(dirpath=directory, filename="{epoch}-{step}-{val_loss:2f}", save_last=True, save_top_k=3, monitor="val_loss",  
                                    mode="min", verbose=True)
wandb_logger = WandbLogger(save_dir=os.path.join(directory, "logs"), name=args.logged_entry_name, project="Multimodal-ImageBaselines")

trainer = Trainer(max_epochs=args.nb_epochs, devices=1, accelerator='gpu' if torch.cuda.is_available() else 'cpu', 
                    accumulate_grad_batches = args.grad_steps, # To run the backward step after n batches, helps to increase the batch size
                    benchmark = True, # Fastens the training process
                    deterministic=True, # Ensures reproducibility 
                    limit_train_batches=args.train_data_percentage, # trains on 10% of the data,
                    check_val_every_n_epoch = 1, # run val loop every 1 training epochs
                    # callbacks=[model_checkpoint, early_stop_callback], # Enables model checkpoint and early stopping
                    callbacks=[early_stop_callback], 
                    logger = wandb_logger,
                    precision='16-mixed' # Mixed Precision system
                    )

# %% Train the model
start_time = time.time()
trainer.fit(model, data_module)
print("Total training:", str(datetime.timedelta(seconds=time.time()-start_time)))
trainer.save_checkpoint(os.path.join(directory, "final_model.ckpt"))
torch.save(model.state_dict(), os.path.join(directory, "final_model.model"))

# %% Test the model
trainer.test(model, data_module)

