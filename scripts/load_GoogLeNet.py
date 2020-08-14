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
goog_bc_1 = tf.keras.models.load_model(r'saved_models\GoogLeNet_binary_crossentropy0.1')

# - SGD = 0.01
goog_bc_2 = tf.keras.models.load_model(r'saved_models\GoogLeNet_binary_crossentropy0.01')

# - SGD = 0.001
goog_bc_3 = tf.keras.models.load_model(r'saved_models\GoogLeNet_binary_crossentropy0.001')

# loss function: 'mean_squared_error'
# - SGD = 0.1
goog_mse_1 = tf.keras.models.load_model(r'saved_models\GoogLeNet_mean_squared_error0.1')

# - SGD = 0.01
goog_mse_2 = tf.keras.models.load_model(r'saved_models\GoogLeNet_mean_squared_error0.01')

# - SGD = 0.001
goog_mse_3 = tf.keras.models.load_model(r'saved_models\GoogLeNet_mean_squared_error0.001')

# loss function: 'mean_squared_logarithmic_error'
# - SGD = 0.1
goog_msle_1 = tf.keras.models.load_model(r'saved_models\GoogLeNet_mean_squared_logarithmic_error0.1')

# - SGD = 0.01
goog_msle_2 = tf.keras.models.load_model(r'saved_models\GoogLeNet_mean_squared_logarithmic_error0.01')

# - SGD = 0.001
goog_msle_3 = tf.keras.models.load_model(r'saved_models\GoogLeNet_mean_squared_logarithmic_error0.001')

models = [[goog_bc_1, "GoogLeNet_binary_crossentropy0.1"],
          [goog_bc_2, "GoogLeNet_binary_crossentropy0.01"],
          [goog_bc_3, "GoogLeNet_binary_crossentropy0.001"],
          [goog_mse_1, "GoogLeNet_mean_squared_error0.1"],
          [goog_mse_2, "GoogLeNet_mean_squared_error0.01"],
          [goog_mse_3, "GoogLeNet_mean_squared_error0.001"],
          [goog_msle_1, "GoogLeNet_mean_squared_logarithmic_error0.1"],
          [goog_msle_2, "GoogLeNet_mean_squared_logarithmic_error0.01"],
          [goog_msle_3, "GoogLeNet_mean_squared_logarithmic_error0.001"]]

# Print summaries:
# for i in models:
#     print(i[1])
#     print(i[0].summary())

############################################################
# Validate Models: get final results
############################################################

#models_test = [models[0]]

accuracy_results = []
all_predictions = []

for i in models:

    batch_size = 128

    # load new unseen dataset
    validation_datagen = ImageDataGenerator(rescale = 1 / 255)

    val_generator = validation_datagen.flow_from_directory(
        r'.\cleaned_data\validate',
        target_size = (224, 224),
        batch_size = batch_size,
        class_mode = 'binary'
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
    accuracy_results.append([i[1], eval_result[1]])
    all_predictions.append([i[1], predicted_classes])

print(accuracy_results)
print(all_predictions)
