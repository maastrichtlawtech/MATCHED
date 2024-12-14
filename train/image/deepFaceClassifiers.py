"""
Python version: 3.10
Description: Fine-tunes DeepFace models for authorship identification task using VENDOR labels.
"""


# %% Loading libraries
import os
import sys
import argparse
import time
import datetime
import random

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

# Custom library
sys.path.append('../../process/')
from imageUtilities import load_images_and_labels
from loadData import DeepFaceImageDataModule

sys.path.append('../../architectures/')
from deepFaceLayer import DeepFaceModel

import warnings
warnings.filterwarnings('ignore')

# %% Setting up the Argparser
parser = argparse.ArgumentParser(description="Trains a image classifier to establish baselines for Authorship tasks on Backpage advertisements.")
parser.add_argument('--model_name_or_path', type=str, default="Facenet512", help="Name of the model to be trained (can only be between distilbert-base-cased)")
parser.add_argument('--logged_entry_name', type=str, default="Facenet512-seed:1111", help="Logged entry name visible on weights and biases")
parser.add_argument('--data_dir', type=str, default='/workspace/persistent/HTClipper/data/processed', help="""Data directory""")
parser.add_argument('--data_type', type=str, default="faces", help="can be faces for the dataset with human faces or nofaces for body parts dataset")
parser.add_argument('--geography', type=str, default='chicago', help="""geography of data, can be only between chicago, atlanta, houston, dallas, detroit, ny, or sf""")
parser.add_argument('--save_dir', type=str, default="/workspace/persistent/HTClipper/models/image-baselines/end-to-end", help="""Directory for models to be saved""")
parser.add_argument('--batch_size', type=int, default=32, help="Batch Size")
parser.add_argument('--nb_epochs', type=int, default=40, help="Number of Epochs")
parser.add_argument('--patience', type=int, default=3, help="Patience for Early Stopping")
parser.add_argument('--seed', type=int, default=1111, help='Random seed value')
parser.add_argument('--warmup_steps', type=int, default=0, help="Warmup proportion")
parser.add_argument('--grad_steps', type=int, default=4, help="Gradient accumulating step")
parser.add_argument('--learning_rate', type=float, default=6e-4, help="learning rate")
parser.add_argument('--train_data_percentage', type=float, default=1.0, help="Percentage of training data to be used")
parser.add_argument('--adam_epsilon', type=float, default=1e-6, help="Epsilon value for adam optimizer")
parser.add_argument('--min_delta_change', type=float, default=0.01, help="Minimum change in delta in validation loss for Early Stopping")
parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay")
parser.add_argument('--augment_data', type=bool, default=False, help='Enables data augmentation')
parser.add_argument('--nb_augmented_samples', type=int, default=5, help='Number of augmented samples to be generated')
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
assert args.data_type in ["faces", "nofaces"]
assert args.model_name_or_path in ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace", "GhostFaceNet"]

# Creating directories
directory = os.path.join(args.save_dir, args.model_name_or_path.split("/")[-1], args.geography, args.data_type, "seed:" + str(args.seed), "lr-" + str(args.learning_rate))
Path(directory).mkdir(parents=True, exist_ok=True)
Path(args.save_dir).mkdir(parents=True, exist_ok=True)

# %% Loading dataset
if args.data_type == "faces":
    df = pd.read_csv(os.path.join(args.data_dir, "chicago_faces.csv")) 
else:
    df = pd.read_csv(os.path.join(args.data_dir, "chicago_nofaces.csv")) 

# Removing vendors that have less than 2 ads
vendors_of_interest = {k:v for k, v in dict(Counter(df.VENDOR)).items() if v>1}
df = df[df['VENDOR'].isin(list(vendors_of_interest.keys()))]

# Remapping new vendor ids
all_vendors = df.VENDOR.unique()
vendor_to_idx_dict =  {}
for vendor in all_vendors:
    if vendor not in vendor_to_idx_dict.keys():
        vendor_to_idx_dict[vendor] = len(vendor_to_idx_dict)
        
df["VENDOR"] = df["VENDOR"].replace(vendor_to_idx_dict)

# %% Load and preprocess images
# The target size is fixed to 224x224 for a fair comparison with the ViT models.
images, labels = load_images_and_labels(df, target_size=(224, 224), augment=args.augment_data,
                                         num_augmented_samples=args.nb_augmented_samples)
assert images.shape[0] == labels.shape[0]

# %% Split data
# Split ratio is set to 0.20 between training and test data, and 0.05 between training and val data
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.20, random_state=1111)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=1111)

# %% Instantiate DataModule and Model
num_classes = df.VENDOR.nunique()
model = DeepFaceModel(model_name=args.model_name_or_path, num_classes=num_classes, learning_rate=args.learning_rate)
data_module = DeepFaceImageDataModule(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=args.batch_size, augment=args.augment_data)

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
                    callbacks=[model_checkpoint, early_stop_callback], # Enables model checkpoint and early stopping
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