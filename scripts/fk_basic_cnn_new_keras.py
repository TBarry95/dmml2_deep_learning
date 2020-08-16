# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 19:03:44 2020

@author: Frank
"""


##Repeat with the cleaned test/training dataset
import tensorflow as tf
import random as rn
import os
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import math

print(keras.__version__)

from numpy.random import seed
seed(42)# keras seed fixing
import tensorflow as tf
tf.random.set_seed(42)# tensorflow seed fixing


#(x_train, y_train),(x_test, y_test) = mnist.load_data()

print("Current Working Directory " , os.getcwd())
os.chdir("C:\\Users\\Frank\\dmml2_deep_learning\\scripts")
print("Current Working Directory " , os.getcwd())

# Extract dataset from folder:
train_datagen = ImageDataGenerator(rescale = 1/255)
test_datagen = ImageDataGenerator(rescale = 1/255)

# Batch Gradient Descent. Batch Size = Size of Training Set
# Stochastic Gradient Descent. Batch Size = 1
# Mini-Batch Gradient Descent. 1 < Batch Size < Size of Training Set

# get training images
train_gen = train_datagen.flow_from_directory(
    r'.\cleaned_data\train',
    target_size = (224, 224), # target_size = (32, 32),
    batch_size = 128,
    class_mode='binary'
    #classes = ['normal','viral']
)

# get testing images
test_gen = test_datagen.flow_from_directory(
    r'.\cleaned_data\test',
    target_size = (224, 224), #FK: Target size is the height and width of the images...
    batch_size  = 128,
    class_mode='binary'
    #classes = ['normal','viral']
)

imgs, labels = next(train_gen)

import matplotlib.pyplot as plt
print(imgs[0])
print(labels[0])
plt.imshow(imgs[0],cmap=plt.cm.binary)
plt.show()

print(labels[2])
plt.imshow(imgs[2],cmap=plt.cm.binary)
plt.show()

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) #FK: Added another layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) #FK: Added another layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) #FK: Added another layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) #FK: Added another layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) #FK: Added another layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) #FK: Added another layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) #FK: Added another layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) #FK: Added another layer
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_gen,
        steps_per_epoch=17, #NUMBER OF SAMPLES IN TRAINING/BATCH_SIZE : https://datascience.stackexchange.com/questions/47405/what-to-set-in-steps-per-epoch-in-keras-fit-generator
        epochs=10, 
        validation_data=test_gen
    )

#val_loss, val_acc = model.evaluate(train_gen, test_gen)
#print(val_loss)
#print(val_acc)

############################################################
# Validate Model: get final results
############################################################


####Save the model down....




# Save the entire model as a SavedModel.
#!mkdir -p saved_models 

model.save('saved_models/frank_cnn_10layer_224_224') 
model.save_weights('saved_models/frank_cnn_10layer_224_224/weights') 





# load new unseen validate dataset
validation_datagen = ImageDataGenerator(rescale = 1 / 255)

val_generator = validation_datagen.flow_from_directory(
    r'.\cleaned_data\validate',
    target_size = (224, 224),
    batch_size = 128, #Should be 18??
    class_mode = 'binary' #,
    #shuffle = False # see: https://github.com/keras-team/keras/issues/4875
)

eval_result = model.evaluate_generator(val_generator) #eval_result = model.evaluate_generator(val_generator, 300)
print('Loss rate for validation: ', eval_result[0])
print('Accuracy rate for validation: ', eval_result[1])


predictions = model.predict(val_generator)
print(predictions)
import numpy as np

for i in range(1,100):
    print(np.argmax(predictions[i]))

imgs, labels = next(val_generator) 

for i in range(1,100):
    print(labels[i])

plt.imshow(imgs[0],cmap=plt.cm.binary)
plt.show()


#Load the saved model
new_model = tf.keras.models.load_model('saved_models/frank_cnn_10layer_224_224')
new_weights = new_model.load_weights('saved_models/frank_cnn_10layer_224_224/weights')

new_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Check its architecture
new_model.summary()

eval_result = new_model.evaluate_generator(val_generator) #eval_result = model.evaluate_generator(val_generator, 300)
print('Loss rate for validation: ', eval_result[0])
print('Accuracy rate for validation: ', eval_result[1])



print(tf.__version__)























#######################################################Save Model attempt 1##################
# from tensorflow.keras.models import model_from_json

# # serialize model to json
# json_model = model.to_json()

# #save the model architecture to JSON file
# with open('fashionmnist_model.json', 'w') as json_file:
#     json_file.write(json_model)
    
# #saving the weights of the model
# model.save_weights('FashionMNIST_weights.h5')


# from tensorflow.keras.initializers import glorot_uniform
# #Reading the model from JSON file
# with open('fashionmnist_model.json', 'r') as json_file:
#     json_savedModel= json_file.read()

# #load the model architecture 
# model_j = tf.keras.models.model_from_json(json_savedModel)
# model_j.summary()











# ######################################Save Model attempt 2##################
# with open('saved_models/frank_cnn_5layer_32_32_attempt2.json', 'w') as f:
#     f.write(model.to_json())

# model.save_weights('saved_models/frank_cnn_5layer_32_32_attempt2_weights.h5')


# with open('saved_models/frank_cnn_5layer_32_32_attempt2.json', 'w') as f:
#     attempt2_model = tf.keras.models.model_from_json('saved_models/frank_cnn_5layer_32_32_attempt2.json')

# model.load_weights('saved_models/frank_cnn_5layer_32_32_attempt2_weights.h5')






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







