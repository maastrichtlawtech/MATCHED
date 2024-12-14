""" 
Python version: 3.9
Description: Contains helper classes and functions to load the data into the LightningDataModule.
"""

# %% Importing libraries
import os
import re
import random

from tqdm import tqdm
from collections import defaultdict

from PIL import Image

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, random_split, TensorDataset, Sampler
from datasets import load_dataset
import lightning.pytorch as pl

from torchvision import transforms

from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from transformers import RobertaTokenizerFast, RobertaModel
from transformers import ViTModel

# Custom Library
from imageUtilities import get_augmentation_pipeline

import warnings

# Define a custom dataset class for creating batches of sentence pairs
class HTDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_length):
        # Constructor method that takes in a list of input pairs (data) and a tokenizer object
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        # Returns the length of the input data
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieves the input pair at the given index in the data list
        input_pair = self.data[idx]
        # Tokenizes the input pair using the given tokenizer object, and returns the result as a PyTorch tensor
        encoded_pair = self.tokenizer(input_pair[0], input_pair[1], padding='max_length', truncation=True, max_length=self.max_seq_length, return_tensors='pt')
        return encoded_pair


class HTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, model_name_or_dir: str, demography: str = "north", seed: int = 1111,  max_seq_length: int = 512, 
                train_batch_size: int = 8, eval_batch_size: int = 1, split_ratio: float = 0.20, **kwargs):
        super().__init__()

        # Initialize the class attributes
        self.data_dir = data_dir
        self.demography = demography
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.split_ratio = split_ratio
        self.seed = seed
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_dir, use_fast=True)
        self.nb_pos = 1
        self.nb_neg = train_batch_size - 1

        self.train_data = None
        self.val_data = None
        self.test_data = None

    def load_and_split_data(self):
        # Load the data from a CSV file
        data_df = pd.read_csv(os.path.join(self.data_dir, self.demography + '.csv'), error_bad_lines=False, warn_bad_lines=False)

        # Identify and keep vendors with at least 2 instances
        class_counts = self.data_df['VENDOR'].value_counts()
        valid_classes = class_counts[class_counts >= 2].index
        self.data_df = self.data_df[self.data_df['VENDOR'].isin(valid_classes)][["TEXT", "VENDOR"]].drop_duplicates()

        # Initialize lists to hold the source texts, target texts, and labels
        source_texts, target_texts, label_list = ([] for i in range(3))

        # Get the unique vendors from the data
        all_vendors = data_df.VENDOR.unique()

        # Iterate through the data to find anchors, positives, and labels
        pbar = tqdm(total=len(all_vendors))
        for vendor in all_vendors:
            df = data_df[data_df["VENDOR"]==vendor]
            text_data = df.TEXT.to_list()
            
            # Find all possible pairs of source and target texts
            text_data = [(a, b) for idx, a in enumerate(text_data) for b in text_data[idx + 1:]]
            
            # Add the source texts, target texts, and labels to their respective lists
            source_texts.append([data[0] for data in text_data])
            target_texts.append([data[1] for data in text_data])
            label_list.append([vendor] * len(text_data))
            
            pbar.update(1)
        pbar.close()

        # Flatten the lists of source texts, target texts, and labels
        source_texts = [item for sublist in source_texts for item in sublist]
        target_texts = [item for sublist in target_texts for item in sublist]
        label_list = [item for sublist in label_list for item in sublist]

        data = list(zip(source_texts, target_texts, label_list))

        # Splitting the data
        train_data, test_data = train_test_split(data, test_size=self.split_ratio, random_state = self.seed, shuffle=True)
        train_data, val_data = train_test_split(train_data, test_size=0.05, random_state = self.seed, shuffle=True)
        return train_data, val_data, test_data 
    
    def get_positive_pairs(self, data, author):
        # Returns a generator expression that yields all positive pairs for the given author in the data
        return ((x[0], x[1]) for x in data if x[2] == author)

    def get_negative_pairs(self, data, author):
        author_dict = defaultdict(list)
        # Creates a dictionary with keys being all authors that are not the given author, 
        # and values being lists of negative pairs for each author
        for x in data:
            if x[2] != author:
                author_dict[x[2]].append((x[0], x[1]))
        return author_dict

    def get_batches(self, data):
        authors = set(x[2] for x in data)
        all_batches = []
        for author in authors:
            # Generates positive pairs for the current author
            positive_pairs = self.get_positive_pairs(data, author)
            # Generates negative pairs for all other authors
            negative_author_pairs = self.get_negative_pairs(data, author)
            for pos_pair in positive_pairs:
                current_batch = []
                current_batch.append(pos_pair)
                # Selects a random subset of authors to sample negative pairs from
                negative_authors = set(negative_author_pairs.keys())
                while len(current_batch) < self.nb_pos + self.nb_neg:
                    # Randomly selects an author to sample a negative pair from
                    random_negative_author = random.choice(list(negative_authors))
                    current_batch.append(random.choice(negative_author_pairs[random_negative_author]))
                    # Removes the selected author from the set of negative authors so it won't be selected again
                    negative_authors.remove(random_negative_author)
                all_batches.append(current_batch)
        return all_batches
    
    def setup(self, stage=None):
        # Load and split the data
        self.train_data, self.val_data, self.test_data = self.load_and_split_data()
        if stage == 'fit' or stage is None:
            self.train_batches = self.get_batches(self.train_data)
            self.val_batches = self.get_batches(self.val_data)

    # Returning the pytorch-lightning default training DataLoader 
    def train_dataloader(self):
        dataset = HTDataset(self.train_batches, self.tokenizer, self.max_seq_length)
        return DataLoader(dataset, batch_size=self.train_batch_size, shuffle=True)

    # Returning the pytorch-lightning default validation DataLoader
    def val_dataloader(self):
        dataset = HTDataset(self.val_batches, self.tokenizer, self.max_seq_length)
        return DataLoader(dataset, batch_size=self.eval_batch_size)

    # Returning the pytorch-lightning default test DataLoader
    def test_dataloader(self):
        test_batches = self.get_batches(self.test_data)
        dataset = HTDataset(test_batches, self.tokenizer, self.max_seq_length)
        return DataLoader(dataset, batch_size=self.eval_batch_size)
    
class TokenizeDataset(Dataset):
    def __init__(self, x, y, tokenizer, length=128, return_idx=False):
        super(TokenizeDataset, self).__init__()
        self.tokenizer = tokenizer
        self.length = length
        self.x = x
        self.return_idx = return_idx
        self.y = torch.tensor(y)
        self.tokens_cache = {}

    def tokenize(self, x):
        dic = self.tokenizer.batch_encode_plus(
            [x],  # input must be a list
            max_length=self.length,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            return_tensors="pt"
        )
        return [x[0] for x in dic.values()]  # get rid of the first dim

    def __getitem__(self, idx):
        int_idx = int(idx)
        assert idx == int_idx
        idx = int_idx
        if idx not in self.tokens_cache:
            self.tokens_cache[idx] = self.tokenize(self.x[idx])
        input_ids, token_type_ids, attention_mask = self.tokens_cache[idx]
        if self.return_idx:
            return input_ids, token_type_ids, attention_mask, self.y[idx], idx, self.x[idx]
        return input_ids, token_type_ids, attention_mask, self.y[idx]

    def __len__(self):
        return len(self.y)
    
class TrainSamplerMultiClassUnit(Sampler):
    def __init__(self, dataset, sample_unit_size):
        super().__init__(None)
        self.x = dataset.x
        self.y = dataset.y
        self.sample_unit_size = sample_unit_size
        # print(f'train sampler with sample unit size {sample_unit_size}')
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        indices = list(range(len(self.y)))
        label_cluster = {}
        for i in indices:
            label = self.y[i].item()
            if label not in label_cluster:
                label_cluster[label] = []
            label_cluster[label].append(i)

        dataset_matrix = []
        for key, value in label_cluster.items():
            random.shuffle(value)
            num_valid_samples = len(value) // self.sample_unit_size * self.sample_unit_size
            dataset_matrix.append(torch.tensor(value[:num_valid_samples]).view(self.sample_unit_size, -1))

        tuples = torch.cat(dataset_matrix, dim=1).transpose(1, 0).split(1, dim=0)
        tuples = [x.flatten().tolist() for x in tuples]
        random.shuffle(tuples)
        all = sum(tuples, [])

        # print(f'from dataset sampler: original dataset size {len(self.y)}, resampled dataset size {len(all)}. 'f'sample unit size {self.sample_unit_size}')

        return iter(all)

    def __len__(self):
        return self.length

class HTClassifierDataModule(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        # Initialize the class attributes
        if isinstance(args, tuple) and len(args) > 0: 
            self.args = args[0]

        # Handling the padding token in distilgpt2 by substituting it with eos_token_id
        if self.args.tokenizer_name_or_path == "distilgpt2":
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_name_or_path, use_fast=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_name_or_path, use_fast=True)
    
    def setup(self, stage=None):        
        # Load the dataset into a pandas dataframe.
        data_df = pd.read_csv(os.path.join(self.args.data_dir, self.args.demography + '.csv'))[["TEXT", "VENDOR"]].drop_duplicates()
        # Identify and keep vendors with at least 2 instances
        class_counts = data_df['VENDOR'].value_counts()
        valid_classes = class_counts[class_counts >= 2].index
        data_df = data_df[data_df['VENDOR'].isin(valid_classes)][["TEXT", "VENDOR"]].drop_duplicates()

        # Randomly shuffle the data
        data_df = data_df.sample(frac=1)

        text = data_df.TEXT.values.tolist()
        vendors = data_df.VENDOR.values.tolist()

        # Tokenizing the data with padding and truncation
        # Note: Tokenization is initially done before altering the training texts.
        encodings = self.tokenizer(text, add_special_tokens=True, max_length=512, padding='max_length', return_token_type_ids=False, truncation=True, 
                                   return_attention_mask=True, return_tensors='pt') 

        # Convert the lists into tensors.
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']

        # Since the vendor IDs are not the current representations of the class labels, we remap these label IDs
        vendors_dict = {}
        i = 0
        for vendor in vendors:
            if vendor not in vendors_dict.keys():
                vendors_dict[vendor] = i
                i += 1
        vendors = [vendors_dict[vendor] for vendor in vendors]
        labels = torch.tensor(vendors)

        # Combine the inputs into a TensorDataset.
        dataset = TensorDataset(input_ids, attention_mask, labels)

        # Splitting the dataset into training, validation, and testing sets
        total_length = len(dataset)
        train_size = int(0.75 * total_length)
        test_size = int(0.20 * total_length)
        val_size = total_length - train_size - test_size

        train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size], generator=torch.Generator().manual_seed(self.args.seed))

        # Only replace numbers with 'N' in the training dataset
        train_texts = [re.sub(r'\d', 'N', text) for text in data_df.iloc[train_dataset.indices]['TEXT']]
        train_vendors = data_df.iloc[train_dataset.indices]['VENDOR'].tolist()

        # Re-tokenize the modified training data
        train_encodings = self.tokenizer(train_texts, add_special_tokens=True, max_length=512, padding='max_length', return_token_type_ids=False, truncation=True, 
                                         return_attention_mask=True, return_tensors='pt')
        train_input_ids = train_encodings['input_ids']
        train_attention_mask = train_encodings['attention_mask']
        train_labels = torch.tensor([vendors_dict[vendor] for vendor in train_vendors])

        # Recreate the training dataset with modified texts
        self.train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset

    # Returning the pytorch-lightning default training DataLoader 
    def train_dataloader(self):
        return DataLoader(self.train_dataset, sampler=RandomSampler(self.train_dataset), batch_size=self.args.batch_size) 

    # Returning the pytorch-lightning default val DataLoader 
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.args.batch_size) 
         
    # Returning the pytorch-lightning default test DataLoader 
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.args.batch_size) 

class LMDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer_name_or_path, train_file, validation_file, line_by_line, pad_to_max_length,
                 preprocessing_num_workers, overwrite_cache, max_seq_length, mlm_probability,
                 train_batch_size, val_batch_size, dataloader_num_workers):
        super().__init__()
        self.train_file = train_file
        self.validation_file = validation_file
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.line_by_line = line_by_line
        self.pad_to_max_length = pad_to_max_length
        self.preprocessing_num_workers = preprocessing_num_workers
        self.overwrite_cache = overwrite_cache
        self.max_seq_length = max_seq_length
        self.mlm_probability = mlm_probability
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.dataloader_num_workers = dataloader_num_workers

    def setup(self, stage=None):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)
        extension = self.train_file.split(".")[-1]
        if extension in ("txt", "raw"):
            extension = "text"

        data_files = {}
        data_files["train"] = self.train_file
        data_files["validation"] = self.validation_file
        datasets = load_dataset(extension, data_files=data_files)

        column_names = datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        if self.line_by_line:
            # When using line_by_line, we just tokenize each nonempty line.
            padding = "max_length" if self.pad_to_max_length else False

            def tokenize_function(examples):
                # Remove empty lines
                examples["text"] = [line for line in examples["text"]
                                    if len(line) > 0 and not line.isspace()]
                return tokenizer(examples["text"], padding=padding, truncation=True, max_length=self.max_seq_length,
                    # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                    # receives the `special_tokens_mask`.
                    return_special_tokens_mask=True,
                )

            tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=self.preprocessing_num_workers,
                                                remove_columns=[text_column_name], load_from_cache_file=not self.overwrite_cache,
                                                )
        else:
            # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
            # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
            # efficient when it receives the `special_tokens_mask`.
            def tokenize_function(examples):
                return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

            tokenized_datasets = datasets.map(
                tokenize_function,
                batched=True,
                num_proc=self.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.overwrite_cache,
            )

            if self.max_seq_length is None:
                self.max_seq_length = tokenizer.model_max_length
            else:
                if self.max_seq_length > tokenizer.model_max_length:
                    warnings.warn(
                        f"The max_seq_length passed ({self.max_seq_length}) is larger than the maximum length for the"
                        f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
                    )
                self.max_seq_length = min(self.max_seq_length, tokenizer.model_max_length)

            # Main data processing function that will concatenate all texts from our dataset and generate chunks of
            # max_seq_length.
            def group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {
                    k: sum(examples[k], []) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
                # customize this part to your needs.
                total_length = (total_length // self.max_seq_length) * self.max_seq_length
                # Split by chunks of max_len.
                result = {
                    k: [t[i: i + self.max_seq_length]
                        for i in range(0, total_length, self.max_seq_length)]
                    for k, t in concatenated_examples.items()
                }
                return result

            # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
            # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
            # might be slower to preprocess.
            #
            # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
            tokenized_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=self.preprocessing_num_workers,
                load_from_cache_file=not self.overwrite_cache,
            )

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=self.mlm_probability)

        train_dataset = tokenized_datasets["train"]
        eval_dataset = tokenized_datasets["validation"]

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, collate_fn=self.data_collator, 
                        num_workers=self.dataloader_num_workers)

    def val_dataloader(self):
        return DataLoader(self.eval_dataset, batch_size=self.val_batch_size, collate_fn=self.data_collator, 
                            num_workers=self.dataloader_num_workers)

class HTContraDataModule(pl.LightningDataModule):
    def __init__(self, file_dir: str = "../data/all.csv", 
                tokenizer_name_or_path: str = "johngiorgi/declutr-small", 
                seed: int = 1111,  max_seq_length: int = 512, sample_unit_size: int = 2,
                train_batch_size: int = 32, eval_batch_size: int = 32, 
                nr_workers: int = 4, **kwargs):
        super().__init__()

        # Initialize the class attributes
        self.file_dir = file_dir
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.seed = seed
        self.sample_unit_size = sample_unit_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.nr_workers = nr_workers
        
    def remap_vendor_ids(self):
        text = self.data_df.TEXT.values.tolist()
        vendors = self.data_df.VENDOR.values.tolist()
        
        vendors_dict = {}
        i = 0
        for vendor in vendors:
            if vendor not in vendors_dict.keys():
                vendors_dict[vendor] = i
                i += 1
                
        self.data_df = self.data_df.replace({"VENDOR": vendors_dict})
        return self.data_df

    def load_and_split_data(self):
        # Load the data from a CSV file
        self.data_df = pd.read_csv(self.file_dir)[["TEXT", "VENDOR"]].drop_duplicates()

        # Identify and keep vendors with at least 2 instances
        class_counts = self.data_df['VENDOR'].value_counts()
        valid_classes = class_counts[class_counts >= 2].index
        self.data_df = self.data_df[self.data_df['VENDOR'].isin(valid_classes)][["TEXT", "VENDOR"]].drop_duplicates()
        
        # Randomly shuffle the data
        self.data_df = self.data_df.sample(frac=1)
        
        # Since the vendor IDs are not the current representations of the class labels, we remap these label IDs to 
        # avoid falling into out-of-bounds problem
        self.data_df = self.remap_vendor_ids()
        
        # splitting data
        train_df, test_df = train_test_split(self.data_df, test_size=0.20, random_state=self.seed)
        train_df, val_df = train_test_split(train_df, test_size=0.05, random_state=self.seed)
        return train_df, val_df, test_df 
    
    def setup(self, stage='fit'):
        # Load and split the data
        train_df, val_df, test_df = self.load_and_split_data()

        # Only replace numbers with 'N' in the training dataset
        train_texts = [re.sub(r'\d', 'N', text) for text in train_df['TEXT']]
        train_vendors = train_df['VENDOR'].tolist()
        val_texts = val_df['TEXT'].tolist()
        val_vendors = val_df['VENDOR'].tolist()
        test_texts = test_df['TEXT'].tolist()
        test_vendors = test_df['VENDOR'].tolist()

        # Create the datasets
        self.train_set = TokenizeDataset(train_texts, train_vendors, self.tokenizer, length=self.max_seq_length)
        self.val_set = TokenizeDataset(val_texts, val_vendors, self.tokenizer, length=self.max_seq_length)
        self.test_set = TokenizeDataset(test_texts, test_vendors, self.tokenizer, length=self.max_seq_length)

    # Returning the pytorch-lightning default training DataLoader 
    def train_dataloader(self):
        train_sampler = TrainSamplerMultiClassUnit(self.train_set, sample_unit_size=self.sample_unit_size)
        dataloader = DataLoader(self.train_set, batch_size=self.train_batch_size, sampler=train_sampler, shuffle=False, 
                                num_workers=self.nr_workers, pin_memory=True, drop_last=True)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(self.val_set, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.nr_workers, 
                                pin_memory=True, drop_last=True)
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(self.test_set, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.nr_workers, 
                                pin_memory=True, drop_last=True)
        return dataloader

class ImageDataset(Dataset):
    def __init__(self, x_data, y_data, transform=None, augment_data=False, num_augmented_samples=1):
        self.x_data = x_data
        self.y_data = y_data
        self.transform = transform
        self.augment_data = augment_data
        self.num_augmented_samples = num_augmented_samples

    def __len__(self):
        if self.augment_data:
            return len(self.x_data) * self.num_augmented_samples
        else:
            return len(self.x_data)

    def __getitem__(self, idx):
        if self.augment_data:
            original_idx = idx // self.num_augmented_samples
        else:
            original_idx = idx
        
        image = self.x_data[original_idx]
        label = self.y_data[original_idx]

        # Ensure image is a NumPy array for Albumentations
        if isinstance(image, Image.Image):  # If it's a PIL Image, convert to NumPy array
            image = np.array(image)

        if self.augment_data:
            augmented = get_augmentation_pipeline()(image=image)
            image = augmented['image']
        
        # Convert back to PIL Image for torchvision transforms
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)
        
        return image, label

class ImageDataModule(pl.LightningDataModule):
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test, batch_size=32, augment_data=False, num_augmented_samples=1):
        super().__init__()
        self.batch_size = batch_size
        self.x_train, self.y_train = x_train, y_train
        self.x_val, self.y_val = x_val, y_val
        self.x_test, self.y_test = x_test, y_test
        self.augment_data = augment_data
        self.num_augmented_samples = num_augmented_samples

        # Define transformations for training, validation, and test sets
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.test_val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        self.train_dataset = ImageDataset(self.x_train, self.y_train, transform=self.train_transform, augment_data=self.augment_data, num_augmented_samples=self.num_augmented_samples)
        self.val_dataset = ImageDataset(self.x_val, self.y_val, transform=self.test_val_transform, augment_data=False)
        self.test_dataset = ImageDataset(self.x_test, self.y_test, transform=self.test_val_transform, augment_data=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True)


# Custom DeepFace Dataset
class DeepFaceDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        if self.transform:
            image_path = self.transform(image_path)
        return image_path, label

# DeepFace Lightning Image DataModule
class DeepFaceImageDataModule(pl.LightningDataModule):
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test, batch_size=32, augment=False):
        super().__init__()
        self.batch_size = batch_size
        self.x_train, self.y_train = x_train, y_train
        self.x_val, self.y_val = x_val, y_val
        self.x_test, self.y_test = x_test, y_test
        self.augment = augment

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if self.augment:
            self.augmentation_pipeline = get_augmentation_pipeline()

    def setup(self, stage=None):
        self.train_dataset = DeepFaceDataset(self.x_train, self.y_train, transform=self.transform)
        self.val_dataset = DeepFaceDataset(self.x_val, self.y_val, transform=self.transform)
        self.test_dataset = DeepFaceDataset(self.x_test, self.y_test, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True)


class MultimodalDataset(Dataset):
    def __init__(self, dataframe, text_tokenizer, image_processor, label_encoder, image_dir, augment=False, image_size=(224, 224)):
        self.dataframe = dataframe
        self.text_tokenizer = text_tokenizer
        self.image_processor = image_processor
        self.label_encoder = label_encoder
        self.augment = augment
        self.augmentation_pipelines = get_augmentation_pipeline() if augment else None
        self.image_size = image_size
        self.image_dir = image_dir
        
        # Remove rows with missing image files
        self.dataframe = self.dataframe[self.dataframe['IMAGES'].apply(lambda x: os.path.exists(os.path.join(self.image_dir, x)))]
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row['TEXT']
        image_path = row['IMAGES']
        label = row['VENDOR']
        
        text_inputs = self.text_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        
        try:
            image = Image.open(os.path.join(self.image_dir, image_path)).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a default image or handle the error as needed
            image = Image.new('RGB', self.image_size, (255, 255, 255))

        # Resize the image to a consistent size
        image = image.resize(self.image_size)
        image_array = np.array(image)

        # Ensure the image array has the correct dimensions (H, W, C)
        if image_array.shape[-1] != 3:
            image_array = np.stack((image_array,) * 3, axis=-1)
        
        if self.augment and 'AUGMENT' in row and row['AUGMENT'] >= 0:
            augment_idx = row['AUGMENT']
            augmented = self.augmentation_pipelines[augment_idx](image=image_array)
            image_array = augmented['image']
        
        # Ensure image dimensions are (C, H, W)
        if image_array.shape[-1] == 3:
            image_array = np.transpose(image_array, (2, 0, 1))
        
        image_tensor = torch.tensor(image_array, dtype=torch.float)
        image_tensor = self.image_processor(images=image_tensor, return_tensors="pt")['pixel_values'].squeeze(0)
        
        input_ids = text_inputs['input_ids'].squeeze(0)
        attention_mask = text_inputs['attention_mask'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': image_tensor,
            'label': torch.tensor(label, dtype=torch.long)
        }

# Create the datasets and dataloaders
class ViLTMultimodalDataset(Dataset):
    def __init__(self, dataframe, processor, label_encoder, image_dir, augment=False, image_size=(384, 384)):
        self.dataframe = dataframe
        self.processor = processor
        self.label_encoder = label_encoder
        self.image_dir = image_dir
        self.augment = augment
        self.image_size = image_size  # Set the fixed image size
        
        # Remove rows with missing image files
        self.dataframe = self.dataframe[self.dataframe['IMAGES'].apply(lambda x: os.path.exists(os.path.join(self.image_dir, x)))]
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row['TEXT']
        image_path = row['IMAGES']
        label = row['VENDOR']
        
        try:
            image = Image.open(os.path.join(self.image_dir, image_path)).convert('RGB')
            image = image.resize(self.image_size)  # Resize the image to the fixed size
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', self.image_size, (255, 255, 255))  # Use a blank image if loading fails

        # Use ViLT processor to process text and image together
        inputs = self.processor(text=[text], images=image, return_tensors="pt", padding="max_length", truncation=True)
        
        # Extract necessary fields
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        pixel_values = inputs['pixel_values'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'label': torch.tensor(label, dtype=torch.long)
        }

class CLIPDataset(Dataset):
    def __init__(self, df, text_tokenizer, image_processor, num_negatives=5, pairing_mode='associated'):
        self.df = df
        self.text_tokenizer = text_tokenizer
        self.image_processor = image_processor
        self.num_negatives = num_negatives
        self.pairing_mode = pairing_mode

        # Assume that 'region' is a column in your DataFrame indicating the region of the text-image pair
        self.region_groups = df.groupby('region')

        # Define a transform to resize the image to 224x224
        # self.image_transform = transforms.Compose([
        #    transforms.Resize((224, 224)),  # Resize image to 224x224
        #    transforms.ToTensor(),  # Convert to tensor
        #])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Select a positive text-image pair
        pos_row = self.df.iloc[idx]
        pos_text = pos_row['TEXT']
        pos_image_path = pos_row['IMAGES']
        pos_region = pos_row['region']

        # Tokenize the text and process the positive image
        pos_text_inputs = self.text_tokenizer(pos_text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        pos_image = self._load_image(pos_image_path)

        # Sample negatives (based on pairing_mode)
        if self.pairing_mode == 'associated':
            neg_texts, neg_image_paths = self._get_associated_negative_samples(pos_region)
        elif self.pairing_mode == 'non-associated':
            neg_texts, neg_image_paths = self._get_non_associated_negative_samples(pos_region)
        else:
            raise ValueError("pairing_mode must be either 'associated' or 'non-associated'")

        neg_text_inputs = [self.text_tokenizer(neg_text, return_tensors="pt", padding="max_length", truncation=True, max_length=512) for neg_text in neg_texts]
        neg_images = [self._load_image(neg_image_path) for neg_image_path in neg_image_paths]

        neg_input_ids = torch.stack([neg_text['input_ids'].squeeze(0) for neg_text in neg_text_inputs])
        neg_attention_mask = torch.stack([neg_text['attention_mask'].squeeze(0) for neg_text in neg_text_inputs])
        neg_pixel_values = torch.stack(neg_images)

        return {
            'pos_input_ids': pos_text_inputs['input_ids'].squeeze(0),
            'pos_attention_mask': pos_text_inputs['attention_mask'].squeeze(0),
            'pos_pixel_values': pos_image,
            'neg_input_ids': neg_input_ids,
            'neg_attention_mask': neg_attention_mask,
            'neg_pixel_values': neg_pixel_values
        }

    def _get_associated_negative_samples(self, pos_region):
        # Sample negatives from another region (associated text-image pairs)
        other_regions = [region for region in self.region_groups.groups.keys() if region != pos_region]
        neg_region = random.choice(other_regions)
        neg_df = self.region_groups.get_group(neg_region)

        neg_text_indices = torch.randint(0, len(neg_df), (self.num_negatives,))
        neg_texts = neg_df.iloc[neg_text_indices]['TEXT'].values
        neg_image_paths = neg_df.iloc[neg_text_indices]['IMAGES'].values

        return neg_texts, neg_image_paths

    def _get_non_associated_negative_samples(self, pos_region):
        # Sample negatives from another region (non-associated text-image pairs)
        other_regions = [region for region in self.region_groups.groups.keys() if region != pos_region]
        neg_region = random.choice(other_regions)
        neg_df = self.region_groups.get_group(neg_region)

        neg_text_indices = torch.randint(0, len(neg_df), (self.num_negatives,))
        neg_image_indices = torch.randint(0, len(neg_df), (self.num_negatives,))

        neg_texts = neg_df.iloc[neg_text_indices]['TEXT'].values
        neg_image_paths = neg_df.iloc[neg_image_indices]['IMAGES'].values

        return neg_texts, neg_image_paths

    def _load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')  # Ensure 3 channels
        # Resize image to 224x224
        # image = self.image_transform(image)
        image = self.image_processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        return image
    
class BLIP2Dataset(Dataset):
    def __init__(self, df, text_tokenizer, t5_tokenizer, image_processor, num_negatives=5, pairing_mode='associated'):
        self.df = df
        self.text_tokenizer = text_tokenizer
        self.image_processor = image_processor
        self.num_negatives = num_negatives
        self.pairing_mode = pairing_mode
        self.t5_tokenizer = t5_tokenizer

        # Assume that 'region' is a column in your DataFrame indicating the region of the text-image pair
        self.region_groups = df.groupby('region')

        # Define a transform to resize the image to 224x224
        # self.image_transform = transforms.Compose([
        #    transforms.Resize((224, 224)),  # Resize image to 224x224
        #    transforms.ToTensor(),  # Convert to tensor
        #])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Select a positive text-image pair
        pos_row = self.df.iloc[idx]
        pos_text = pos_row['TEXT']
        pos_image_path = pos_row['IMAGES']
        pos_region = pos_row['region']

        # Tokenize the text and process the positive image
        pos_text_inputs = self.text_tokenizer(pos_text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        pos_text_inputs_t5 = self.t5_tokenizer(pos_text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)

        pos_image = self._load_image(pos_image_path)

        # Sample negatives (based on pairing_mode)
        if self.pairing_mode == 'associated':
            neg_texts, neg_image_paths = self._get_associated_negative_samples(pos_region)
        elif self.pairing_mode == 'non-associated':
            neg_texts, neg_image_paths = self._get_non_associated_negative_samples(pos_region)
        else:
            raise ValueError("pairing_mode must be either 'associated' or 'non-associated'")

        neg_text_inputs = [self.text_tokenizer(neg_text, return_tensors="pt", padding="max_length", truncation=True, max_length=512) for neg_text in neg_texts]
        neg_images = [self._load_image(neg_image_path) for neg_image_path in neg_image_paths]

        neg_input_ids = torch.stack([neg_text['input_ids'].squeeze(0) for neg_text in neg_text_inputs])
        neg_attention_mask = torch.stack([neg_text['attention_mask'].squeeze(0) for neg_text in neg_text_inputs])
        neg_pixel_values = torch.stack(neg_images)

        return {
            'pos_input_ids': pos_text_inputs['input_ids'].squeeze(0),
            'pos_attention_mask': pos_text_inputs['attention_mask'].squeeze(0),
            'pos_pixel_values': pos_image,
            'neg_input_ids': neg_input_ids,
            'neg_attention_mask': neg_attention_mask,
            'neg_pixel_values': neg_pixel_values,
            'pos_input_ids_t5': pos_text_inputs_t5['input_ids'].squeeze(0),
            'pos_attention_mask_t5': pos_text_inputs_t5['attention_mask'].squeeze(0)
        }

    def _get_associated_negative_samples(self, pos_region):
        # Sample negatives from another region (associated text-image pairs)
        other_regions = [region for region in self.region_groups.groups.keys() if region != pos_region]
        neg_region = random.choice(other_regions)
        neg_df = self.region_groups.get_group(neg_region)

        neg_text_indices = torch.randint(0, len(neg_df), (self.num_negatives,))
        neg_texts = neg_df.iloc[neg_text_indices]['TEXT'].values
        neg_image_paths = neg_df.iloc[neg_text_indices]['IMAGES'].values

        return neg_texts, neg_image_paths

    def _get_non_associated_negative_samples(self, pos_region):
        # Sample negatives from another region (non-associated text-image pairs)
        other_regions = [region for region in self.region_groups.groups.keys() if region != pos_region]
        neg_region = random.choice(other_regions)
        neg_df = self.region_groups.get_group(neg_region)

        neg_text_indices = torch.randint(0, len(neg_df), (self.num_negatives,))
        neg_image_indices = torch.randint(0, len(neg_df), (self.num_negatives,))

        neg_texts = neg_df.iloc[neg_text_indices]['TEXT'].values
        neg_image_paths = neg_df.iloc[neg_image_indices]['IMAGES'].values

        return neg_texts, neg_image_paths

    def _load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')  # Ensure 3 channels
        # Resize image to 224x224
        # image = self.image_transform(image)
        image = self.image_processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        return image

class BLIP2ConditionalDataset(Dataset):
    def __init__(self, df, text_tokenizer, t5_tokenizer, image_processor, num_negatives=5, pairing_mode='non-associated'):
        self.df = df
        self.text_tokenizer = text_tokenizer
        self.image_processor = image_processor
        self.num_negatives = num_negatives
        self.pairing_mode = pairing_mode
        self.t5_tokenizer = t5_tokenizer

        # Assume that 'region' is a column in your DataFrame indicating the region of the text-image pair
        self.region_groups = df.groupby('region')

        # Define a transform to resize the image to 224x224
        # self.image_transform = transforms.Compose([
        #    transforms.Resize((224, 224)),  # Resize image to 224x224
        #    transforms.ToTensor(),  # Convert to tensor
        #])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Select a positive text-image pair
        pos_row = self.df.iloc[idx]
        full_text = pos_row['TEXT']
        pos_image_path = pos_row['IMAGES']
        pos_region = pos_row['region']

        # Split the text on [SEP] for text generation
        if '[SEP]' in full_text:
            conditional_text, target_text = full_text.split('[SEP]', 1)
            conditional_text = conditional_text.strip()
            target_text = target_text.strip()
        else:
            # If [SEP] is not present, handle accordingly
            conditional_text = ''
            target_text = full_text.strip()

        # Tokenize the full text for CLIP and ITM losses
        pos_text_inputs = self.text_tokenizer(
            full_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )

        # Tokenize the conditional text (text prompt) for text generation
        conditional_text_inputs = self.t5_tokenizer(
            conditional_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )

        # Tokenize the target text for text generation
        target_text_inputs = self.t5_tokenizer(
            target_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )

        # Process the positive image
        pos_image = self._load_image(pos_image_path)

        # Sample negatives (based on pairing_mode)
        if self.pairing_mode == 'associated':
            neg_texts, neg_image_paths = self._get_associated_negative_samples(pos_region)
        elif self.pairing_mode == 'non-associated':
            neg_texts, neg_image_paths = self._get_non_associated_negative_samples(pos_region)
        else:
            raise ValueError("pairing_mode must be either 'associated' or 'non-associated'")

        # Tokenize negative texts
        neg_text_inputs = [self.text_tokenizer(
            neg_text, return_tensors="pt", padding="max_length", truncation=True, max_length=512
        ) for neg_text in neg_texts]

        # Load negative images
        neg_images = [self._load_image(neg_image_path) for neg_image_path in neg_image_paths]

        # Prepare negative inputs
        neg_input_ids = torch.stack([neg_text['input_ids'].squeeze(0) for neg_text in neg_text_inputs])
        neg_attention_mask = torch.stack([neg_text['attention_mask'].squeeze(0) for neg_text in neg_text_inputs])
        neg_pixel_values = torch.stack(neg_images)

        return {
            # For CLIP and ITM losses (full text)
            'pos_input_ids': pos_text_inputs['input_ids'].squeeze(0),
            'pos_attention_mask': pos_text_inputs['attention_mask'].squeeze(0),
            # For text generation loss (conditional text and target text)
            'conditional_input_ids': conditional_text_inputs['input_ids'].squeeze(0),
            'conditional_attention_mask': conditional_text_inputs['attention_mask'].squeeze(0),
            'target_input_ids': target_text_inputs['input_ids'].squeeze(0),
            'target_attention_mask': target_text_inputs['attention_mask'].squeeze(0),
            # Positive image
            'pos_pixel_values': pos_image,
            # Negative samples
            'neg_input_ids': neg_input_ids,
            'neg_attention_mask': neg_attention_mask,
            'neg_pixel_values': neg_pixel_values,
        }


    def _get_associated_negative_samples(self, pos_region):
        # Sample negatives from another region (associated text-image pairs)
        other_regions = [region for region in self.region_groups.groups.keys() if region != pos_region]
        neg_region = random.choice(other_regions)
        neg_df = self.region_groups.get_group(neg_region)

        neg_text_indices = torch.randint(0, len(neg_df), (self.num_negatives,))
        neg_texts = neg_df.iloc[neg_text_indices]['TEXT'].values
        neg_image_paths = neg_df.iloc[neg_text_indices]['IMAGES'].values

        return neg_texts, neg_image_paths

    def _get_non_associated_negative_samples(self, pos_region):
        # Sample negatives from another region (non-associated text-image pairs)
        other_regions = [region for region in self.region_groups.groups.keys() if region != pos_region]
        neg_region = random.choice(other_regions)
        neg_df = self.region_groups.get_group(neg_region)

        neg_text_indices = torch.randint(0, len(neg_df), (self.num_negatives,))
        neg_image_indices = torch.randint(0, len(neg_df), (self.num_negatives,))

        neg_texts = neg_df.iloc[neg_text_indices]['TEXT'].values
        neg_image_paths = neg_df.iloc[neg_image_indices]['IMAGES'].values

        return neg_texts, neg_image_paths

    def _load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')  # Ensure 3 channels
        # Resize image to 224x224
        # image = self.image_transform(image)
        image = self.image_processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        return image

def clip_collate_fn(batch):
    texts = [item['text'] for item in batch]
    images = [item['image'] for item in batch]
    return {'texts': texts, 'images': images}

def blip2_collate_fn(batch):
    images = [item['image'] for item in batch]
    text_inputs = [item['text_input'] for item in batch]
    target_texts = [item['target_text'] for item in batch]
    return {
        'images': images,
        'text_inputs': text_inputs,
        'target_texts': target_texts
    }

class CLIPCheckpointDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row['TEXT']
        image_path = row['IMAGES']
        image = Image.open(image_path).convert('RGB')
        return {'text': text, 'image': image}

class BLIP2CheckpointDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(row['IMAGES']).convert('RGB')
        target_text = row['TEXT'] 
        
        sample = {
            'image': image,  # Pass the PIL image directly
            'text_input': target_text.split("[SEP]")[0],  # e.g., "Describe the image."
            'target_text': target_text.split("[SEP]")[1]
        }
        return sample

# Create the datasets
class FineTuneCLIPstyleModelDataset(Dataset):
    def __init__(self, df, text_tokenizer, image_processor):
        self.df = df.reset_index(drop=True)
        self.text_tokenizer = text_tokenizer
        self.image_processor = image_processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row["TEXT"]
        image_path = row["IMAGES"]
        label = row["VENDOR"]

        # Tokenize the text
        text_inputs = self.text_tokenizer(
            text, return_tensors="pt", padding="max_length", truncation=True, max_length=512
        )

        # Process the image
        image = Image.open(image_path).convert("RGB")
        image = self.image_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        return {
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
            "pixel_values": image,
            "labels": torch.tensor(label, dtype=torch.long),
        }
