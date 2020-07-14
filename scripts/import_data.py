#########################################################
# DES: Read in pictures from directory
# BY: Tiernan Barry
#########################################################

import pandas as pd
import numpy as np
#from sklearn import model_selection
#from sklearn.model_selection import train_test_split
#from sklearn.neural_network import MLPClassifier
#from sklearn.linear_model import Perceptron
#from sklearn.metrics import accuracy_score
#import matplotlib.pyplot as plt
import random
#from PIL import Image
##from numpy import asarray
import cv2
import os
import glob
import re

#########################################################
# Set RELATIVE working directory (do not use absolute paths!)
# Working directory should be '.\scripts' assuming using windows OS
#########################################################

current_dir = os.getcwd()
print("Current directory: ", current_dir)

if current_dir[len(current_dir)-7:len(current_dir)] != 'scripts':
    try:
        os.chdir(r".\scripts")
        print("Changing working directory to: ", os.getcwd())
        print("New working directory: ", os.getcwd())
    except:
        print("Not in correct repo, continuing script anyway")
else:
    print("Working directory already correct: ", os.getcwd())

#########################################################
# Extract data:
#########################################################

#######################################
# Extract pictures into lists:
#######################################

# Get list of all files in each directory:
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

# filter pneumonia files for JUST viral??
viral_pattern = re.compile('virus')
virus_list = [i for i in all_penu_img_list if viral_pattern.search(i) ]
print("Number of PNEUMONIA/VIRUS images: ", len(set(virus_list)))

# List comprehension to get all files into list variables:
def parse_images(image_list):
    pics = []
    print("Parsing images into numpy matrices, this takes a few minutes")
    for pic in image_list:
        img = cv2.imread(pic)
        pics.append(img)
    return pics

normal_pics = parse_images(all_normal_img_list)
pneu_vir_pics = parse_images(virus_list)

# Test data:
print(normal_pics[0])
print(pneu_vir_pics[0])

img = normal_pics[0]
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()