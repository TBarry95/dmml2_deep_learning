#########################################################
# DES:
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
from itertools import product


#########################################################
# Define model parameter options:
#########################################################

# Loss function:
loss_functions = ['binary_crossentropy', 'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error']

# Optimisation:
sgd_1 = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD')
opts = [sgd_1]

# combination of 2:
combos = list(product(loss_functions, opts))

#########################################################
# For each combination, run CNN model:
#########################################################

accuracy = []

for i in combos:
    model_results = cnn_fns.cnn_5_layers(loss = i[0], optimizer= i[1], activation = 'relu')
    accuracy.append(model_results)
    print("Finished model")


