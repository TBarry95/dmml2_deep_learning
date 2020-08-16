# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 12:06:43 2020

@author: Frank
"""


import tensorflow as tf
import tensorflow.keras as keras
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

print("Current Working Directory " , os.getcwd())
os.chdir("C:\\Users\\Frank\\dmml2_deep_learning\\scripts")

train_datagen = ImageDataGenerator(rescale = 1/255)
test_datagen = ImageDataGenerator(rescale = 1/255)

# Batch Gradient Descent. Batch Size = Size of Training Set
# Stochastic Gradient Descent. Batch Size = 1
# Mini-Batch Gradient Descent. 1 < Batch Size < Size of Training Set

batch_size = 4 #Should be 128

# get training images
train_gen = train_datagen.flow_from_directory(
    r'.\cleaned_data\train',
    target_size = (224,224), # target_size = (32, 32),
    batch_size = batch_size,
    class_mode='binary'
    #classes = ['normal','viral']
)

# get testing images
test_gen = test_datagen.flow_from_directory(
    r'.\cleaned_data\test',
    target_size = (224,224), #FK: Target size is the height and width of the images...
    batch_size  = batch_size,
    class_mode='binary'
    #classes = ['normal','viral']
)

#Model:
model = Sequential()
model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))


model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
#model.add(Dense(units=2, activation="softmax"))
model.add(Dense(units=1, activation="softmax"))
#FK: Try... model.add(Dense(10, activation="softmax"))

from tensorflow.keras.optimizers import Adam
opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

#FK:
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#End

#https://stackoverflow.com/questions/50304156/tensorflow-allocation-memory-allocation-of-38535168-exceeds-10-of-system-memor

model.summary()

#from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
#checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq=1)
#early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
#hist = model.fit_generator(steps_per_epoch=100,generator=train_gen, validation_data= test_gen, validation_steps=10,epochs=100,callbacks=[checkpoint,early])

hist = model.fit(train_gen,
        steps_per_epoch=10,
        epochs=10,
        #validation_steps = 948/batch_size
        validation_steps = 10,
        validation_data=test_gen
    )


model.save('saved_models/frank_vgg16_adam_opt') 
model.save_weights('saved_models/frank_vgg16_adam_opt/weights') 

# load new unseen validate dataset
validation_datagen = ImageDataGenerator(rescale = 1 / 255)

val_generator = validation_datagen.flow_from_directory(
    r'.\cleaned_data\validate',
    target_size = (224, 224),
    batch_size = 4, #Should be 18??
    class_mode = 'binary'
)

eval_result = model.evaluate_generator(val_generator)
print('Loss rate for validation: ', eval_result[0])
print('Accuracy rate for validation: ', eval_result[1])





#Load the saved model
new_model = tf.keras.models.load_model('saved_models/frank_vgg16_adam_opt')
new_weights = new_model.load_weights('saved_models/frank_vgg16_adam_opt/weights')

new_model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

# Check its architecture
new_model.summary()

eval_result = new_model.evaluate_generator(val_generator) #eval_result = model.evaluate_generator(val_generator, 300)
print('Loss rate for validation: ', eval_result[0])
print('Accuracy rate for validation: ', eval_result[1])

