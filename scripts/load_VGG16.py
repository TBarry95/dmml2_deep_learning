############################################################
# DES: Load in the trained GoogLeNet models and run validation dataset
############################################################

############################################################
# Libraries:
############################################################

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import math
import numpy as np
import pandas as pd

try:
    import scripts.set_working_dir as set_wd
except:
    import set_working_dir as set_wd

#########################################################
# Set Working Directory:
# - Ensure RELATIVE working directory (so it can be replicated by any user)
# - Ensure users can read data using either Windows or UNIX folders
# - Working directory should be '.\scripts' for windows or './scripts' for UNIX
#########################################################

working_dir = set_wd.set_correct_working_dir()

############################################################
# Get trained GoogLeNet models:
############################################################

# loss function: 'binary_crossentropy'
# - SGD = 0.1
goog_bc_1 = tf.keras.models.load_model(r'saved_models\frank_vgg16_binary_crossentropy0.1')

# - SGD = 0.01
goog_bc_2 = tf.keras.models.load_model(r'saved_models\frank_vgg16_binary_crossentropy0.01')

# - SGD = 0.001
goog_bc_3 = tf.keras.models.load_model(r'saved_models\frank_vgg16_binary_crossentropy0.001')

# loss function: 'mean_squared_error'
# - SGD = 0.1
goog_mse_1 = tf.keras.models.load_model(r'saved_models\frank_vgg16_mean_squared_error0.1')

# - SGD = 0.01
goog_mse_2 = tf.keras.models.load_model(r'saved_models\frank_vgg16_mean_squared_error0.01')

# - SGD = 0.001
goog_mse_3 = tf.keras.models.load_model(r'saved_models\frank_vgg16_mean_squared_error0.001')

# loss function: 'mean_squared_logarithmic_error'
# - SGD = 0.1
goog_msle_1 = tf.keras.models.load_model(r'saved_models\frank_vgg16_mean_squared_logarithmic_error0.1')

# - SGD = 0.01
goog_msle_2 = tf.keras.models.load_model(r'saved_models\frank_vgg16_mean_squared_logarithmic_error0.01')

# - SGD = 0.001
goog_msle_3 = tf.keras.models.load_model(r'saved_models\frank_vgg16_mean_squared_logarithmic_error0.001')

models = [[goog_bc_1, "VGG16_binary_crossentropy0.1"],
          [goog_bc_2, "VGG16_binary_crossentropy0.01"],
          [goog_bc_3, "VGG16_binary_crossentropy0.001"],
          [goog_mse_1, "VGG16_mean_squared_error0.1"],
          [goog_mse_2, "VGG16_mean_squared_error0.01"],
          [goog_mse_3, "VGG16_mean_squared_error0.001"],
          [goog_msle_1, "VGG16_mean_squared_logarithmic_error0.1"],
          [goog_msle_2, "VGG16_mean_squared_logarithmic_error0.01"],
          [goog_msle_3, "VGG16_mean_squared_logarithmic_error0.001"]]

# Print summaries:
# for i in models:
#     print(i[1])
#     print(i[0].summary())

############################################################
# Validate Models: get final results
############################################################

def get_validation_results(model_list, batch_size, target_size):

    accuracy = []
    predcitions = []

    for i in model_list:

        # load new unseen dataset
        validation_datagen = ImageDataGenerator(rescale=1 / 255)
        val_generator = validation_datagen.flow_from_directory(
            r'.\cleaned_data\validate',
            target_size= target_size,
            batch_size = batch_size,
            class_mode='binary'
        )

        # accuracy summary
        eval_result = i[0].evaluate_generator(val_generator)
        print('Loss rate for validation: ', eval_result[0])
        print('Accuracy rate for validation: ', eval_result[1])

        # get predictions:
        predictions = i[0].predict(val_generator, verbose=1)
        predictions_array = np.array(predictions)
        print(predictions_array.shape)
        predicted_classes = np.argmax(predictions_array, axis=1)

        # save results:
        accuracy.append([i[1], eval_result[1]])
        predcitions.append([i[1], predicted_classes])

    return [accuracy, predcitions]

goog_output = get_validation_results(model_list=models, batch_size=4, target_size=(224,224))
goog_accuracy = goog_output[0]
goog_predictions = goog_output[1]


############################################################
# Export CSV:
############################################################

df_results = pd.DataFrame()
all_results = []
all_models = []

for i in goog_accuracy:
    all_results.append(i[1])
    all_models.append(i[0])

df_results['ACCURACY'] = all_results
df_results['MODEL'] = all_models
df_results1 = df_results.sort_values('ACCURACY')
df_results1 = df_results1.reset_index()
df_results1 = df_results1[['ACCURACY', 'MODEL']]

df_results1.to_csv(r".\VGG16_results.csv", index=False)

