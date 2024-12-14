# %% Importing libraries
import random
from PIL import Image
import concurrent.futures

import numpy as np

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define the augmentation pipeline
def get_augmentation_pipeline():
    return A.Compose([
        A.HorizontalFlip(p=random.uniform(0.3, 0.7)),
        A.VerticalFlip(p=random.uniform(0.3, 0.7)),    # Vertical flip with random probability
        A.RandomResizedCrop(height=224, width=224, scale=(random.uniform(0.6, 0.8), random.uniform(0.8, 1.0)), ratio=(random.uniform(0.7, 0.8), random.uniform(1.2, 1.4)), p=1.0),
        A.Rotate(limit=random.randint(10, 20), p=random.uniform(0.4, 0.6)),
        A.ShiftScaleRotate(shift_limit=random.uniform(0.05, 0.15), scale_limit=random.uniform(0.05, 0.15), rotate_limit=random.uniform(-10, 10), p=random.uniform(0.4, 0.6)),
        A.GaussNoise(var_limit=(random.uniform(10.0, 30.0), random.uniform(30.0, 50.0)), p=random.uniform(0.1, 0.3))
    ])

def image_to_numpy_array(image_path, target_size=(224, 224), augment=True, num_augmented_samples=5):
    augmented_images = []
    try:
        # Open the image file
        with Image.open(image_path) as img:
            # Convert the image to RGB (in case it is in another mode)
            img = img.convert('RGB')
            # Convert the image to a NumPy array
            img_array = np.array(img)

            # Resize the image
            img_array = cv2.resize(img_array, target_size)

            if augment:
                # Apply augmentations multiple times
                augmentation_pipeline = get_augmentation_pipeline()
                for _ in range(num_augmented_samples):
                    augmented_image = augmentation_pipeline(image=img_array)['image']
                    augmented_images.append(augmented_image)
            else:
                augmented_images.append(img_array)

        return augmented_images
    except (IOError, SyntaxError) as e:
        # Handle invalid image files
        return []

def load_images_and_labels(df, target_size=(224, 224), augment=True, num_augmented_samples=1):
    image_paths = df['IMAGE'].to_list()
    labels = df['VENDOR'].to_list()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map the image_to_numpy_array function to the list of image paths with target_size and augment
        results = list(executor.map(lambda p: image_to_numpy_array(p, target_size, augment, num_augmented_samples), image_paths))
    
    if augment:
        valid_images = [img for sublist in results for img in sublist]
        valid_labels = [[labels[i]] * num_augmented_samples for i, sublist in enumerate(results) if sublist]
        valid_labels = [label for sublist in valid_labels for label in sublist]
    else:
        valid_images = [img[0] for img in results if img]
        valid_labels = [labels[i] for i, img in enumerate(results) if img]

    return np.array(valid_images), np.array(valid_labels)