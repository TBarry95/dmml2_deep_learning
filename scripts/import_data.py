#########################################################
# DES: Import raw dataset from '.\scripts\input_files' and export target dataset into 'cleaned_data' folder.
#      Removing all bacterial pneumonia images for the following reasons:
#      - Balance the dataset 50:50 between pneumonia / normal
#      - COVID is viral
# BY: Tiernan Barry
#########################################################

import os
import glob
import re
import shutil
import cv2
from sklearn.model_selection import train_test_split

#########################################################
# Set Working Directory:
# - Ensure RELATIVE working directory (so it can be replicated by any user)
# - Ensure users can read data using either Windows or UNIX folders
# - Working directory should be '.\scripts' for windows or './scripts' for UNIX
#########################################################

current_dir = os.getcwd()
print("Current directory: ", current_dir)

if current_dir[len(current_dir)-7:len(current_dir)] != 'scripts':
    try:
        os.chdir(r".\scripts")
        print("Changing working directory to: ", os.getcwd())
        print("New working directory: ", os.getcwd())
    except:
        print(r"Can't find .\scripts folder, will try '/scripts' instead (Windows v UNIX) ")

        try:
            os.chdir(r"./scripts")
            print("Changing working directory to: ", os.getcwd())
            print("New working directory: ", os.getcwd())
        except:
            print(r"Still can't find correct directory, continuing script anyway")
else:
    print("Working directory already correct: ", os.getcwd())

working_dir = os.getcwd()

#########################################################
# Extract data:
#########################################################

#######################################
# Extract pictures into lists:
#######################################

if working_dir[len(working_dir)-8:len(working_dir)] == '/scripts':
    # UNIX: Get list of all files in each directory:
    all_files_train_norm = glob.glob(r"./input_files/train/NORMAL/*g")
    all_files_train_pneu = glob.glob(r"./input_files/train/PNEUMONIA/*g")
    all_files_test_norm = glob.glob(r"./input_files/test/NORMAL/*g")
    all_files_test_pneu = glob.glob(r"./input_files/test/PNEUMONIA/*g")
else:
    # Windows: Get list of all files in each directory:
    all_files_train_norm = glob.glob(r".\input_files\train\NORMAL\*g")
    all_files_train_pneu = glob.glob(r".\input_files\train\PNEUMONIA\*g")
    all_files_test_norm = glob.glob(r".\input_files\test\NORMAL\*g")
    all_files_test_pneu = glob.glob(r".\input_files\test\PNEUMONIA\*g")

# combine all normal images
all_normal_imgs = [all_files_train_norm, all_files_test_norm]
all_penu_imgs = [all_files_train_pneu, all_files_test_pneu]

# combine into 2 lists: Normal and Pneumonia
def get_all_images(all_imgs):
    new_list = []
    for i in all_imgs:
        for ii in i:
            new_list.append(ii)
    return new_list

all_normal_img_list = get_all_images(all_normal_imgs)
all_penu_img_list = get_all_images(all_penu_imgs)

print("Number of NORMAL images: ", len(set(all_normal_img_list)))
print("Number of PNEUMONIA images: ", len(set(all_penu_img_list)))

#######################################
# Filter for just VIRAL PNEUMONIA:
# - Balances dataset
# - Makes problem 2 way classification, not 3
#######################################

# filter pneumonia files for JUST viral??
viral_pattern = re.compile('virus')
virus_list = [i for i in all_penu_img_list if viral_pattern.search(i) ]
print("Number of PNEUMONIA/VIRUS images: ", len(set(virus_list)))

#######################################
# Export new folder: NORMAL and VIRAL
#######################################

normal_images = all_normal_img_list
viral_images = virus_list

normal_train, normal_test= train_test_split(normal_images, test_size=0.2, random_state=0)
viral_train, viral_test= train_test_split(viral_images, test_size=0.2, random_state=0)

for i in normal_train:
    shutil.copy(i, r'.\cleaned_data\train\normal')

for i in normal_test:
    shutil.copy(i, r'.\cleaned_data\test\normal')

for i in viral_train:
    shutil.copy(i, r'.\cleaned_data\train\viral')

for i in viral_test:
    shutil.copy(i, r'.\cleaned_data\test\viral')

print("Filtered raw dataset to new folder: \cleaned_data ")

#########################################################
# Transform data:
# - Convert images to np arrays
#########################################################

# List comprehension to get all files into list variables:
def parse_images(image_list):
    pics = []
    print("Parsing images into numpy matrices, this takes a few minutes")
    for pic in image_list:
        img = cv2.imread(pic)
        pics.append(img)
    return pics

# normal_pics = parse_images(all_normal_img_list)
# pneu_vir_pics = parse_images(virus_list)

# Test data:
# print(normal_pics[0])
# print(pneu_vir_pics[0])
