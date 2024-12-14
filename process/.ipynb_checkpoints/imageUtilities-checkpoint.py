# %% Importing libraries
from PIL import Image
import concurrent.futures

import numpy as np

import cv2
import albumentations as A

# Define the augmentation pipeline
def get_augmentation_pipeline():
    return A.Compose([
        A.HorizontalFlip(),
        # A.RandomBrightnessContrast(p=0.2),
        A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0), ratio=(0.75, 1.333), p=1.0),
        A.Rotate(limit=15, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.5),
        A.GaussNoise(p=0.2),
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