# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 19:03:44 2020

@author: Frank
"""


##Repeat with the cleaned test/training dataset
import tensorflow as tf
import os
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import math

#(x_train, y_train),(x_test, y_test) = mnist.load_data()

# Extract dataset from folder:
train_datagen = ImageDataGenerator(rescale = 1/255)
test_datagen = ImageDataGenerator(rescale = 1/255)

# Batch Gradient Descent. Batch Size = Size of Training Set
# Stochastic Gradient Descent. Batch Size = 1
# Mini-Batch Gradient Descent. 1 < Batch Size < Size of Training Set

# get training images
train_gen = train_datagen.flow_from_directory(
    r'.\cleaned_data\train',
    target_size = (32, 32),
    batch_size = 128,
    class_mode='binary'
    #classes = ['normal','viral']
)

# get testing images
test_gen = test_datagen.flow_from_directory(
    r'.\cleaned_data\test',
    target_size = (32, 32), #FK: Target size is the height and width of the images...
    batch_size  = 128,
    class_mode='binary'
    #classes = ['normal','viral']
)

imgs, labels = next(train_gen)

import matplotlib.pyplot as plt
print(imgs[0])
plt.imshow(imgs[0],cmap=plt.cm.binary)
plt.show()

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_gen,
        steps_per_epoch=10,
        epochs=5,
        validation_data=test_gen
    )

#val_loss, val_acc = model.evaluate(train_gen, test_gen)
#print(val_loss)
#print(val_acc)

# predictions = model.predict(x_test)
# print(predictions)
# import numpy as np
# print(np.argmax(predictions[0]))
# plt.imshow(x_test[0],cmap=plt.cm.binary)
# plt.show()













###########################################################Number dataset...
# import tensorflow as tf

# mnist = tf.keras.datasets.mnist
# (x_train, y_train),(x_test, y_test) = mnist.load_data()

# print(x_train[0])

# import matplotlib.pyplot as plt

# plt.imshow(x_train[0],cmap=plt.cm.binary)
# plt.show()
# print(y_train[0])


# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

# print(x_train[0])

# plt.imshow(x_train[0],cmap=plt.cm.binary)
# plt.show()

# model = tf.keras.models.Sequential()

# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=3)

# val_loss, val_acc = model.evaluate(x_test, y_test)
# print(val_loss)
# print(val_acc)

# predictions = model.predict(x_test)
# print(predictions)
# import numpy as np
# print(np.argmax(predictions[0]))
# plt.imshow(x_test[0],cmap=plt.cm.binary)
# plt.show()
###################################################END OF NUMBER DATASET







