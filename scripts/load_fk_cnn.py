# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 22:59:19 2020

@author: Frank
"""

##########################Load and validate the CNN model (224,224) 10 layer

##Repeat with the cleaned test/training dataset
import tensorflow as tf
import random as rn
import os
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import math
import numpy as np
import pandas as pd

print(keras.__version__)

from numpy.random import seed
seed(42)# keras seed fixing
import tensorflow as tf
tf.random.set_seed(42)# tensorflow seed fixing



#########################################10 layered#################################


#Load the saved model
new_model = tf.keras.models.load_model('saved_models/frank_cnn_10layer_224_224')
new_weights = new_model.load_weights('saved_models/frank_cnn_10layer_224_224/weights')

new_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Check its architecture
new_model.summary()


#Validation Dataset...
# load new unseen validate dataset
validation_datagen = ImageDataGenerator(rescale = 1 / 255)

val_generator = validation_datagen.flow_from_directory(
    r'.\cleaned_data\validate',
    target_size = (224, 224),
    batch_size = 128, #Should be 18??
    class_mode = 'binary' #,
    #shuffle = False # see: https://github.com/keras-team/keras/issues/4875
)


eval_result = new_model.evaluate_generator(val_generator) #eval_result = model.evaluate_generator(val_generator, 300)
print('Loss rate for validation: ', eval_result[0])
print('Accuracy rate for validation: ', eval_result[1])

accuracy = []
predcitions = []

# get predictions:
predictions = new_model.predict(val_generator, verbose=1)
predictions_array = np.array(predictions)
print(predictions_array.shape)
predicted_classes = np.argmax(predictions_array, axis=1)

# save results:
accuracy.append(['10_layer_CNN', eval_result[1]])
predcitions.append([new_model, predicted_classes])


###################################5 layered####################################

#Load the saved model
new_model = tf.keras.models.load_model('saved_models/frank_cnn_5layer_224_224')
new_weights = new_model.load_weights('saved_models/frank_cnn_5layer_224_224/weights')

new_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Check its architecture
new_model.summary()


#Validation Dataset...
# load new unseen validate dataset
validation_datagen = ImageDataGenerator(rescale = 1 / 255)

val_generator = validation_datagen.flow_from_directory(
    r'.\cleaned_data\validate',
    target_size = (224, 224),
    batch_size = 128, #Should be 18??
    class_mode = 'binary' #,
    #shuffle = False # see: https://github.com/keras-team/keras/issues/4875
)


eval_result = new_model.evaluate_generator(val_generator) #eval_result = model.evaluate_generator(val_generator, 300)
print('Loss rate for validation: ', eval_result[0])
print('Accuracy rate for validation: ', eval_result[1])

# get predictions:
predictions = new_model.predict(val_generator, verbose=1)
predictions_array = np.array(predictions)
print(predictions_array.shape)
predicted_classes = np.argmax(predictions_array, axis=1)

# save results:
accuracy.append(['5_layer_CNN', eval_result[1]])
predcitions.append([new_model, predicted_classes])


############################################################
# Export CSV:
############################################################

df_results = pd.DataFrame()

df_results['ACCURACY'] = [accuracy[0][1], accuracy[1][1]]
df_results['MODEL'] = [accuracy[0][0], accuracy[1][0]]
df_results1 = df_results.sort_values('ACCURACY')
df_results1 = df_results1.reset_index()
df_results1 = df_results1[['ACCURACY', 'MODEL']]

df_results1.to_csv(r".\fk_CNN_results.csv", index=False)
















# # get predictions:
# predictions = new_model.predict(val_generator, verbose=1)

# predictions_array = np.array(predictions)
# print(predictions_array.shape)
# predicted_classes = np.argmax(predictions_array, axis=1)
