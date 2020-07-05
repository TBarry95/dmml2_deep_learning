# DES: Read in pictures from directory
# BY: Tiernan Barry

import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random
from PIL import Image
#from numpy import asarray
import cv2
import os
import glob

#########################################################
# Set RELATIVE working directory (do not use absolute paths!)
# Working directory should be '.\scripts'
#########################################################

os.chdir(r".\scripts")
print("Checking working directory: ", os.getcwd())

#########################################################
# Get data:
#########################################################

#######################################
# Extract pictures into lists:
#######################################

# Locations of each dataset (test, train for both Normal and Pneumonia)
path_train_normal = "./input_files/train/NORMAL/*g"
path_train_pneumonia = "./input_files/train/PNEUMONIA/*g"
path_test_normal = "./input_files/test/NORMAL/*g"
path_test_pneumonia = "./input_files/test/PNEUMONIA/*g"

# Get list of all files in each directory:
all_files_train_norm = glob.glob(path_train_normal)
all_files_train_pneu = glob.glob(path_train_pneumonia)
all_files_test_norm = glob.glob(path_test_normal)
all_files_test_pneu = glob.glob(path_test_pneumonia)

# filter pneumonia files for JUST viral?? balance the dataset


# List comprehension to get all files into list variables:
normal_train_pics = []
pneu_train_pics = []
normal_test_pics = []
pneu_test_pics = []

for pic in all_files_train_norm:
    img = cv2.imread(pic)
    normal_train_pics.append(img)

for pic in all_files_train_pneu:
    img = cv2.imread(pic)
    pneu_train_pics.append(img)

for pic in all_files_test_norm:
    img = cv2.imread(pic)
    normal_test_pics.append(img)

for pic in all_files_test_pneu:
    img = cv2.imread(pic)
    pneu_test_pics.append(img)

# Test data:
print(normal_train_pics[0])
print(pneu_train_pics[0])
print(normal_test_pics[0])
print(pneu_test_pics[0])







