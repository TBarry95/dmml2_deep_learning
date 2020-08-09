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

#########################################################
# Run CNN model 1:
# loss = binary_crossentropy
# optimiser = RMSprop(lr=0.001)
# activation fn = relu
#########################################################

opt = RMSprop(lr=0.001)
cnn_model1 = cnn_fns.cnn_5_layers(loss = "binary_crossentropy",
                                  optimizer = opt,
                                  activation = 'relu'
                                  )

#########################################################
# Run CNN model 2:
# loss = categorical_crossentropy
# optimiser = RMSprop(lr=0.001)
# activation fn = relu
#########################################################

opt = RMSprop(lr=0.001)
cnn_model2 = cnn_fns.cnn_5_layers(loss = "categorical_crossentropy",
                                  optimizer = opt,
                                  activation = 'relu'
                                  )

#########################################################
# Run CNN model 3: LeNet model:
# Import trained LeNet model and run validation:
#########################################################




