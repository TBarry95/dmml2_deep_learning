#########################################################
# DES: Run multiple CNNs and find optimal results:
# BY: Tiernan Barry
#########################################################

import os
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
# Run CNN model 3:
# loss = x
# optimiser = RMSprop(lr=0.001)
# activation fn = relu
#########################################################

