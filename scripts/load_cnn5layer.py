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
# Get trained models:
############################################################

# loss function: 'binary_crossentropy'
# - SGD = 0.1
cnn5_bc_1 = tf.keras.models.load_model(r'saved_models\cnn_5layer_binary_crossentropy0.1')

# - SGD = 0.01
cnn5_bc_2 = tf.keras.models.load_model(r'saved_models\cnn_5layer_binary_crossentropy0.01')

# - SGD = 0.001
cnn5_bc_3 = tf.keras.models.load_model(r'saved_models\cnn_5layer_binary_crossentropy0.001')

# loss function: 'mean_squared_error'
# - SGD = 0.1
cnn5_mse_1 = tf.keras.models.load_model(r'saved_models\cnn_5layer_mean_squared_error0.1')

# - SGD = 0.01
cnn5_mse_2 = tf.keras.models.load_model(r'saved_models\cnn_5layer_mean_squared_error0.01')

# - SGD = 0.001
cnn5_mse_3 = tf.keras.models.load_model(r'saved_models\cnn_5layer_mean_squared_error0.001')

# loss function: 'mean_squared_logarithmic_error'
# - SGD = 0.1
cnn5_msle_1 = tf.keras.models.load_model(r'saved_models\cnn_5layer_mean_squared_logarithmic_error0.1')

# - SGD = 0.01
cnn5_msle_2 = tf.keras.models.load_model(r'saved_models\cnn_5layer_mean_squared_logarithmic_error0.01')

# - SGD = 0.001
cnn5_msle_3 = tf.keras.models.load_model(r'saved_models\cnn_5layer_mean_squared_logarithmic_error0.001')

models = [[cnn5_bc_1, "cnn_5layer_binary_crossentropy0.1"], [cnn5_bc_2, "cnn_5layer_binary_crossentropy0.01"], [cnn5_bc_3, "cnn_5layer_binary_crossentropy0.001"],
          [cnn5_mse_1, "cnn_5layer_mean_squared_error0.1"], [cnn5_mse_2, "cnn_5layer_mean_squared_error0.01"], [cnn5_mse_3, "cnn_5layer_mean_squared_error0.001"],
          [cnn5_msle_1, "cnn_5layer_mean_squared_logarithmic_error0.1"], [cnn5_msle_2, "cnn_5layer_squared_logarithmic_error0.01"],
          [cnn5_msle_3, "cnn_5layer_mean_squared_logarithmic_error0.001"]]

############################################################
# Validate Model: get final results
############################################################

accuracy_results = []
all_predictions = []

for i in models:

    batch_size = 128

    # load new unseen dataset
    validation_datagen = ImageDataGenerator(rescale = 1 / 255)

    val_generator = validation_datagen.flow_from_directory(
        r'.\cleaned_data\validate',
        target_size=(300, 300),
        batch_size=batch_size,
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
    accuracy_results.append([i[1], eval_result[1]])
    all_predictions.append([i[1], predicted_classes])

