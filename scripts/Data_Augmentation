"""
Script to create augmented data to increase training set size.
Random images are taken from the existing training pool, with combinations of augmentation techniques applied:
- Image tilt between 25% left/right
- Image inversion
- Addition of noise
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import os
import glob
import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util

# Creating local functions to manipulate images
def add_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.add_noise(image_array)

def flip_horizontal(image_array: ndarray):
    return image_array[:, ::-1]

def tilt_image(image_array: ndarray):
    # Random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

# Creating a dictionary of the new functions
available_transformations = {
    'rotate': tilt_image,
    'noise': add_noise,
    'flip_horizontal': flip_horizontal
}

############### Creating new images of "Normal" chest x-rays ###############

# Define the location of the current "Normal" training images
folder_path_normal = r'.//dmml2_deep_learning//scripts//cleaned_data//train//normal'

# Create a limit of 2000 new images
num_files_desired = 2000

# Find all images of "Normal" x-rays
images = [os.path.join(folder_path_normal, f) for f in os.listdir(folder_path_normal) if os.path.isfile(os.path.join(folder_path_normal, f))]

# Define a counter for the while loop
num_generated_files = 0

# Iterate through the images, randomly selecting those to be augmented and saved as a new image
while num_generated_files <= num_files_desired:
    # Read a randomly selected image
    image_path = random.choice(images)
    image_to_transform = sk.io.imread(image_path)
    # Select a random number of augmentation steps to apply
    num_transformations_to_apply = random.randint(1, len(available_transformations))
    # Define a counter for the while loop
    num_transformations = 0
    transformed_image = None
    while num_transformations <= num_transformations_to_apply:
        # For each number of transformations to be made, randomly select which type to apply from the dictionary.
        key = random.choice(list(available_transformations))
        # Manipulate the image
        transformed_image = available_transformations[key](image_to_transform)
        # Increase the counter
        num_transformations += 1
    # Define location of new data
    new_folder = r'.//dmml2_deep_learning//scripts//Augmented_Data//normal'
    new_file_path = '%s/augmented_image_%s.jpg' % (new_folder, num_generated_files)
    # Save new image
    sk.io.imsave(new_file_path, transformed_image)
    # Reopen image
    im = Image.open(new_file_path)
    # Get image dimensions
    width, height = im.size
    # Reduce size of images by a factor of 4
    new_width = int(round(width) / 4)
    new_height = int(round(height) / 4)
    im = im.resize((new_width, new_height), Image.ANTIALIAS)
    # Find dimensions in order to crop from centre of image
    left = (new_width - 224) / 2
    top = (new_height - 224) / 2
    right = (new_width + 224) / 2
    bottom = (new_height + 224) / 2
    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    # Resave images
    im.save(new_file_path)
    print("Saving ", num_generated_files)
    # Increment counter
    num_generated_files += 1

############### Creating new images of "Viral" chest x-rays ###############

# Define the location of the current "Viral" training images
folder_path_viral = r'.//dmml2_deep_learning//scripts//cleaned_data//train//viral'

# Create a limit of 2000 new images
num_files_desired = 2000

# Find all images of "Normal" x-rays
images = [os.path.join(folder_path_viral, f) for f in os.listdir(folder_path_viral) if os.path.isfile(os.path.join(folder_path_viral, f))]

# Define a counter for the while loop
num_generated_files = 0

# Iterate through the images, randomly selecting those to be augmented and saved as a new image
while num_generated_files <= num_files_desired:
    # Read a randomly selected image
    image_path = random.choice(images)
    image_to_transform = sk.io.imread(image_path)
    # Select a random number of augmentation steps to apply
    num_transformations_to_apply = random.randint(1, len(available_transformations))
    # Define a counter for the while loop
    num_transformations = 0
    transformed_image = None
    while num_transformations <= num_transformations_to_apply:
        # For each number of transformations to be made, randomly select which type to apply from the dictionary.
        key = random.choice(list(available_transformations))
        # Manipulate the image
        transformed_image = available_transformations[key](image_to_transform)
        # Increase the counter
        num_transformations += 1
    # Define location of new data
    new_folder = r'.//dmml2_deep_learning//scripts//Augmented_Data//viral'
    new_file_path = '%s/augmented_image_%s.jpg' % (new_folder, num_generated_files)
    # Save new image
    sk.io.imsave(new_file_path, transformed_image)
    # Reopen image
    im = Image.open(new_file_path)
    # Get image dimensions
    width, height = im.size
    # Reduce size of images by a factor of 4
    new_width = int(round(width) / 4)
    new_height = int(round(height) / 4)
    im = im.resize((new_width, new_height), Image.ANTIALIAS)
    # Find dimensions in order to crop from centre of image
    left = (new_width - 224) / 2
    top = (new_height - 224) / 2
    right = (new_width + 224) / 2
    bottom = (new_height + 224) / 2
    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    # Resave images
    im.save(new_file_path)
    print("Saving ", num_generated_files)
    # Increment counter
    num_generated_files += 1

############### Transfer original images into new file location ###############

images = [os.path.join(folder_path_normal, f) for f in os.listdir(folder_path_normal) if os.path.isfile(os.path.join(folder_path_normal, f))]

num_moved_files = 0

new_folder = r'.//dmml2_deep_learning//scripts//Augmented_Data//normal'

while num_moved_files <= len(images):
    new_file_path = '%s/original_image_%s.jpg' % (new_folder, num_moved_files)
    im = Image.open(images[num_moved_files])
    width, height = im.size  # Get dimensions
    new_width = 224
    new_height = 224
    im = im.resize((new_width, new_height), Image.ANTIALIAS)
    im.save(new_file_path)
    print("Saving ",num_moved_files)
    num_moved_files += 1

images = [os.path.join(folder_path_viral, f) for f in os.listdir(folder_path_viral) if os.path.isfile(os.path.join(folder_path_viral, f))]

num_moved_files = 0

new_folder = r'.//dmml2_deep_learning//scripts//Augmented_Data//viral'

while num_moved_files <= len(images):
    new_file_path = '%s/original_image_%s.jpg' % (new_folder, num_moved_files)
    im = Image.open(images[num_moved_files])
    width, height = im.size  # Get dimensions
    new_width = 224
    new_height = 224
    im = im.resize((new_width, new_height), Image.ANTIALIAS)
    im.save(new_file_path)
    print("Saving ",num_moved_files)
    num_moved_files += 1
