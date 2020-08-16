# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 09:15:07 2020

@author: Frank
"""


############################################################
# DES: Define VGG16 CNN and train model.
#      Once trained, export model to working directory.
############################################################

############################################################
# Libraries:
############################################################

import os
#import scripts.set_working_dir as set_wd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import math
from itertools import product

#########################################################
# Set Working Directory:
# - Ensure RELATIVE working directory (so it can be replicated by any user)
# - Ensure users can read data using either Windows or UNIX folders
# - Working directory should be '.\scripts' for windows or './scripts' for UNIX
#########################################################

#working_dir = set_wd.set_correct_working_dir()

############################################################
# Define LeNet model:
# - Parameters to change:
# - - Optimisation
# - - Loss function
############################################################

#########################################
# Define combinations of paramters:
#########################################

# Loss functions
loss_fns = ['binary_crossentropy', 'mean_squared_error', 'mean_squared_logarithmic_error', 'sparse_categorical_crossentropy']

# Optimisation for SGD learning rate:
opts = [0.1,  0.01, 0.001]

# combinations:
combos = list(product(loss_fns, opts))

i = combos[1]

#########################################
# Define X models (X = len(loss_fns)*len(opts)
#########################################

for i in combos:

    VGG16_cnn_model = tf.keras.models.Sequential()
    VGG16_cnn_model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    VGG16_cnn_model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    VGG16_cnn_model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    VGG16_cnn_model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    VGG16_cnn_model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    VGG16_cnn_model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    VGG16_cnn_model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    VGG16_cnn_model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    VGG16_cnn_model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    VGG16_cnn_model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    VGG16_cnn_model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    VGG16_cnn_model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    VGG16_cnn_model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    VGG16_cnn_model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    VGG16_cnn_model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    VGG16_cnn_model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    VGG16_cnn_model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    VGG16_cnn_model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    
    model.add(Flatten())
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=4096,activation="relu"))
    #model.add(Dense(units=2, activation="softmax"))
    model.add(Dense(units=1, activation="softmax"))
    
    #VGG16_cnn_model.compile(optimizer='adam',
    #              loss='sparse_categorical_crossentropy',
    #              metrics=['accuracy'])


    #model_summary = VGG16_cnn_model.summary()
    #print(model_summary)

    sgd = tf.keras.optimizers.SGD(learning_rate= i[1], momentum=0.0, nesterov=False, name='SGD')

    VGG16_cnn_model.compile(loss = i[0],
                            optimizer = sgd,
                            metrics = ['accuracy'])
    
    model_summary = VGG16_cnn_model.summary()
    print(model_summary)


    ############################################################
    # Train Model:
    ############################################################

    batch_size = 128
    training_size = 2148
    testing_size = 538
    epochs = 5

    fn_steps_per_epoch = lambda x: int(math.ceil(1. * x / batch_size))
    steps_per_epoch = fn_steps_per_epoch(training_size)
    test_steps = fn_steps_per_epoch(testing_size)

    # Extract dataset from folder:
    train_datagen = ImageDataGenerator(rescale = 1/255)
    test_datagen = ImageDataGenerator(rescale = 1/255)

    # get training images
    train_gen = train_datagen.flow_from_directory(
        r'.\cleaned_data\train',
        target_size = (32, 32),
        batch_size = batch_size,
        class_mode = 'binary'
    )

    # get testing images
    test_gen = test_datagen.flow_from_directory(
        r'.\cleaned_data\test',
        target_size = (32, 32),
        batch_size  = batch_size,
        class_mode = 'binary'
    )

    # train model
    history = VGG16_cnn_model.fit(
        train_gen,
        steps_per_epoch = steps_per_epoch,
        epochs = epochs,
        validation_data = test_gen,
        validation_steps = test_steps
    )

    ############################################################
    # Export Model to working Directory:
    ############################################################

    model_name_loc = r".\saved_models\VGG16_" + str(i[0]) + str(i[1])
    model_weights_loc = r".\saved_models\VGG16_" + str(i[0]) + str(i[1] + "_weights")

    VGG16_cnn_model.save(model_name_loc)
    VGG16_cnn_model.save_weights(model_weights_loc) 

