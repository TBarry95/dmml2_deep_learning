#################################################################
# DES: Extract all saved models and run results.
#      Plot accuracy metric across all models to find best performers.
# BY: Tiernan Barry
#################################################################

############################################################
# Libraries:
############################################################

import tensorflow as tf
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import math
import numpy as np
try:
    import scripts.set_working_dir as set_wd
except:
    import set_working_dir as set_wd

#################################################################
# Set Working Directory:
# - Ensure RELATIVE working directory (so it can be replicated by any user)
# - Ensure users can read data using either Windows or UNIX folders
# - Working directory should be '.\scripts' for windows or './scripts' for UNIX
#################################################################

working_dir = set_wd.set_correct_working_dir()

#################################################################
# Import all models:
# - Loss functions: 'binary_crossentropy', 'mean_squared_error', 'mean_squared_logarithmic_error'
# - Learn rates: 0.1,  0.01,  0.001
#################################################################

#########################################
# LeNet models:
# - SGD optimisation
#########################################

leNet_bc_1 = tf.keras.models.load_model(r'saved_models\LeNet_binary_crossentropy0.1')
leNet_bc_2 = tf.keras.models.load_model(r'saved_models\LeNet_binary_crossentropy0.01')
leNet_bc_3 = tf.keras.models.load_model(r'saved_models\LeNet_binary_crossentropy0.001')
leNet_mse_1 = tf.keras.models.load_model(r'saved_models\LeNet_mean_squared_error0.1')
leNet_mse_2 = tf.keras.models.load_model(r'saved_models\LeNet_mean_squared_error0.01')
leNet_mse_3 = tf.keras.models.load_model(r'saved_models\LeNet_mean_squared_error0.001')
leNet_msle_1 = tf.keras.models.load_model(r'saved_models\LeNet_mean_squared_logarithmic_error0.1')
leNet_msle_2 = tf.keras.models.load_model(r'saved_models\LeNet_mean_squared_logarithmic_error0.01')
leNet_msle_3 = tf.keras.models.load_model(r'saved_models\LeNet_mean_squared_logarithmic_error0.001')

lenet_models = [[leNet_bc_1, "LeNet_binary_crossentropy0.1"], [leNet_bc_2, "LeNet_binary_crossentropy0.01"], [leNet_bc_3, "LeNet_binary_crossentropy0.001"],
          [leNet_mse_1, "LeNet_mean_squared_error0.1"], [leNet_mse_2, "LeNet_mean_squared_error0.01"], [leNet_mse_3, "LeNet_mean_squared_error0.001"],
          [leNet_msle_1, "LeNet_mean_squared_logarithmic_error0.1"], [leNet_msle_2, "LeNet_mean_squared_logarithmic_error0.01"],
          [leNet_msle_3, "LeNet_mean_squared_logarithmic_error0.001"]]

#########################################
# AlexNet:
#########################################


#########################################
# GoogleNet:
#########################################


#########################################
# Others:
# 1. CNN 5 layers: SGD optimised
# 2. CNN 5 layers RMSPROP optimised
# 3. Franks CNN
#########################################

###############################
# 1. CNN 5 layers: SGD optimised
###############################

cnn5_bc_1 = tf.keras.models.load_model(r'saved_models\cnn_5layer_binary_crossentropy0.1')
cnn5_bc_2 = tf.keras.models.load_model(r'saved_models\cnn_5layer_binary_crossentropy0.01')
cnn5_bc_3 = tf.keras.models.load_model(r'saved_models\cnn_5layer_binary_crossentropy0.001')
cnn5_mse_1 = tf.keras.models.load_model(r'saved_models\cnn_5layer_mean_squared_error0.1')
cnn5_mse_2 = tf.keras.models.load_model(r'saved_models\cnn_5layer_mean_squared_error0.01')
cnn5_mse_3 = tf.keras.models.load_model(r'saved_models\cnn_5layer_mean_squared_error0.001')
cnn5_msle_1 = tf.keras.models.load_model(r'saved_models\cnn_5layer_mean_squared_logarithmic_error0.1')
cnn5_msle_2 = tf.keras.models.load_model(r'saved_models\cnn_5layer_mean_squared_logarithmic_error0.01')
cnn5_msle_3 = tf.keras.models.load_model(r'saved_models\cnn_5layer_mean_squared_logarithmic_error0.001')

#################################
# 2. CNN 5 layers: RMSPROP optimised
#################################

cnn5_bc_1_rm = tf.keras.models.load_model(r'saved_models\cnn_5lyr_rmsprpbinary_crossentropy0.1')
cnn5_bc_2_rm = tf.keras.models.load_model(r'saved_models\cnn_5lyr_rmsprpbinary_crossentropy0.01')
cnn5_bc_3_rm = tf.keras.models.load_model(r'saved_models\cnn_5lyr_rmsprpbinary_crossentropy0.001')
cnn5_mse_1_rm = tf.keras.models.load_model(r'saved_models\cnn_5lyr_rmsprpmean_squared_error0.1')
cnn5_mse_2_rm = tf.keras.models.load_model(r'saved_models\cnn_5lyr_rmsprpmean_squared_error0.01')
cnn5_mse_3_rm = tf.keras.models.load_model(r'saved_models\cnn_5lyr_rmsprpmean_squared_error0.001')
cnn5_msle_1_rm = tf.keras.models.load_model(r'saved_models\cnn_5lyr_rmsprpmean_squared_logarithmic_error0.1')
cnn5_msle_2_rm = tf.keras.models.load_model(r'saved_models\cnn_5lyr_rmsprpmean_squared_logarithmic_error0.01')
cnn5_msle_3_rm = tf.keras.models.load_model(r'saved_models\cnn_5lyr_rmsprpmean_squared_logarithmic_error0.001')

cnn5_models = [[cnn5_bc_1, "cnn_5layer_binary_crossentropy0.1"], [cnn5_bc_2, "cnn_5layer_binary_crossentropy0.01"], [cnn5_bc_3, "cnn_5layer_binary_crossentropy0.001"],
          [cnn5_mse_1, "cnn_5layer_mean_squared_error0.1"], [cnn5_mse_2, "cnn_5layer_mean_squared_error0.01"], [cnn5_mse_3, "cnn_5layer_mean_squared_error0.001"],
          [cnn5_msle_1, "cnn_5layer_mean_squared_logarithmic_error0.1"], [cnn5_msle_2, "cnn_5layer_squared_logarithmic_error0.01"], [cnn5_msle_3, "cnn_5layer_mean_squared_logarithmic_error0.001"], [cnn5_bc_1_rm, "cnn_5lyr_rmsprpbinary_crossentropy0.1"],
          [cnn5_bc_2_rm, "cnn_5lyr_rmsprpbinary_crossentropy0.01"], [cnn5_bc_3_rm, "cnn_5lyr_rmsprpbinary_crossentropy0.001" ], [cnn5_mse_1_rm, "cnn_5lyr_rmsprpmean_squared_error0.1"], [cnn5_mse_2_rm, "cnn_5lyr_rmsprpmean_squared_error0.01"], [cnn5_mse_3_rm, "cnn_5lyr_rmsprpmean_squared_error0.001"],
          [cnn5_msle_1_rm, "cnn_5lyr_rmsprpmean_squared_logarithmic_error0.1"], [cnn5_msle_2_rm, "cnn_5lyr_rmsprpmean_squared_logarithmic_error0.01"],
               [cnn5_msle_3_rm, "cnn_5lyr_rmsprpmean_squared_logarithmic_error0.001"] ]

#################################################################
# Get results:
#################################################################

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
# LeNet:
#########################################

lenet_output = get_validation_results(model_list=lenet_models, batch_size=128, target_size=(32,32))
lenet_accuracy = lenet_output[0]
lenet_predictions = lenet_output[1]

#########################################
# AlexNet:
#########################################


#########################################
# CNN 5 layers: SGD optimised:
#########################################

cnn5_output_sgd = get_validation_results(model_list=cnn5_models, batch_size=128, target_size=(300,300))
cnn5_sgd_accuracy = cnn5_output_sgd[0]
cnn5_sgd_predictions = cnn5_output_sgd[1]

#################################################################
# Combine all results:
#################################################################

df_results = pd.DataFrame()
all_results = []
all_models = []

# Lenet
for i in lenet_accuracy:
    all_results.append(i[1])
    all_models.append(i[0])

# CNN 5 layers
for i in cnn5_sgd_accuracy:
    all_results.append(i[1])
    all_models.append(i[0])

df_results['ACCURACY'] = all_results
df_results['MODEL'] = all_models
df_results1 = df_results.sort_values('ACCURACY')
df_results1 = df_results1.reset_index()
df_results1 = df_results1[['ACCURACY', 'MODEL']]

#################################################################
# Plot results:
#################################################################

df_results1.plot(kind='line')

#########################################
# Combine all results:
#########################################