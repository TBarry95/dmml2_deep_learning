############################################################
# DES: Load in the trained LeNet models and run validation dataset
############################################################

############################################################
# Libraries:
############################################################

import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import math

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
# Get trained models:
############################################################

# loss function: 'binary_crossentropy'
# - SGD = 0.1
leNet_bc_1 = tf.keras.models.load_model(r'saved_models\LeNet_binary_crossentropy0.1')

# - SGD = 0.01
leNet_bc_2 = tf.keras.models.load_model(r'saved_models\LeNet_binary_crossentropy0.01')

# - SGD = 0.001
leNet_bc_3 = tf.keras.models.load_model(r'saved_models\LeNet_binary_crossentropy0.001')

# loss function: 'mean_squared_error'
# - SGD = 0.1
leNet_mse_1 = tf.keras.models.load_model(r'saved_models\LeNet_mean_squared_error0.1')

# - SGD = 0.01
leNet_mse_2 = tf.keras.models.load_model(r'saved_models\LeNet_mean_squared_error0.01')

# - SGD = 0.001
leNet_mse_3 = tf.keras.models.load_model(r'saved_models\LeNet_mean_squared_error0.001')

# loss function: 'mean_squared_logarithmic_error'
# - SGD = 0.1
leNet_msle_1 = tf.keras.models.load_model(r'saved_models\LeNet_mean_squared_logarithmic_error0.1')

# - SGD = 0.01
leNet_msle_2 = tf.keras.models.load_model(r'saved_models\LeNet_mean_squared_logarithmic_error0.01')

# - SGD = 0.001
leNet_msle_3 = tf.keras.models.load_model(r'saved_models\LeNet_mean_squared_logarithmic_error0.001')

models = [[leNet_bc_1, "LeNet_binary_crossentropy0.1"], [leNet_bc_2, "LeNet_binary_crossentropy0.01"], [leNet_bc_3, "LeNet_binary_crossentropy0.001"],
          [leNet_mse_1, "LeNet_mean_squared_error0.1"], [leNet_mse_2, "LeNet_mean_squared_error0.01"], [leNet_mse_3, "LeNet_mean_squared_error0.001"],
          [leNet_msle_1, "LeNet_mean_squared_logarithmic_error0.1"], [leNet_msle_2, "LeNet_mean_squared_logarithmic_error0.01"],
          [leNet_msle_3, "LeNet_mean_squared_logarithmic_error0.001"]]

# Print summaries:
# for i in models:
#     print(i[1])
#     print(i[0].summary())

############################################################
# Validate Models: get final results
############################################################

results = []

for i in models:

    batch_size = 128

    # load new unseen dataset
    validation_datagen = ImageDataGenerator(rescale = 1 / 255)

    val_generator = validation_datagen.flow_from_directory(
        r'.\cleaned_data\validate',
        target_size = (32, 32),
        batch_size = batch_size,
        class_mode = 'binary'
    )

    eval_result = i[0].evaluate_generator(val_generator, 624)
    print('Loss rate for validation: ', eval_result[0])
    print('Accuracy rate for validation: ', eval_result[1])

    # save results:
    results.append([i[1], eval_result[1]])

print(results)
