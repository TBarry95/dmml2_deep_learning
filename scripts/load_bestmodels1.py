############################################################
# DES: Load in best models and get validaiton results
############################################################

import os
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import math
import numpy as np

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
# Get trained  models:
############################################################

model1 = tf.keras.models.load_model(r'best_models\cnn_5lyr_rmsprpmean_squared_error0.001')
model2 = tf.keras.models.load_model(r'saved_models\cnn_5lyr_rmsprpmean_squared_logarithmic_error0.001')

models = [[model1, "cnn_5lyr_rmsprpmean_squared_error0.001"], [model2, "cnn_5lyr_rmsprpmean_squared_logarithmic_error0.001"]]

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

#########################################
# Model 1: cnn_5lyr_rmsprpmean_squared_error0.001
#########################################

output1 = get_validation_results(model_list=models, batch_size=128, target_size=(300,300))
accuracy1 = output1[0]
predictions1 = output1[1]

############################################################
# Export CSV:
############################################################

df_results = pd.DataFrame()
all_results = []
all_models = []

# Lenet
for i in accuracy1:
    all_results.append(i[1])
    all_models.append(i[0])

df_results['ACCURACY'] = all_results
df_results['MODEL'] = all_models
df_results1 = df_results.sort_values('ACCURACY')
df_results1 = df_results1.reset_index()
df_results1 = df_results1[['ACCURACY', 'MODEL']]

df_results1.to_csv(r".\augmented_results.csv", index=False)


