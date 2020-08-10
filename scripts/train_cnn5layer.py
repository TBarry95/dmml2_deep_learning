#########################################################
# DES: Run multiple CNNs and find optimal results:
# BY: Tiernan Barry
#########################################################

import os
import scripts.set_working_dir as set_wd

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing import image
from PIL import Image
from IPython.display import display
from itertools import product

#########################################################
# Set Working Directory:
# - Ensure RELATIVE working directory (so it can be replicated by any user)
# - Ensure users can read data using either Windows or UNIX folders
# - Working directory should be '.\scripts' for windows or './scripts' for UNIX
#########################################################

working_dir = set_wd.set_correct_working_dir()

#########################################################
# Import functions
#########################################################

import scripts.cnn_function as cnn_fns

############################################################
# Define model:
# - Parameters to change:
# - - Optimisation
# - - Loss function
############################################################

#########################################
# Define combinations of paramters:
#########################################

# Loss functions
loss_fns = ['binary_crossentropy', 'mean_squared_error', 'mean_squared_logarithmic_error']

# Optimisation for SGD learning rate:
opts = [0.1,  0.01, 0.001]

# combinations:
combos = list(product(loss_fns, opts))

#########################################
# Define X models (X = len(loss_fns)*len(opts)
#########################################

model_paths = []

# CNN is defined in cnn_function.py
for i in combos:
    model_path = cnn_fns.cnn_5_layers(i[0], i[1],  activation = 'relu')
    model_paths.append(model_path)






