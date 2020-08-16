#########################################################
# DES: Run best perfroming CNNs (4/6 of them - 2 were too large to upload to Github).
#      Results still added from CSV file
# BY: Tiernan Barry
#########################################################

import scripts.set_working_dir as set_wd
import pandas as pd
from tabulate import tabulate
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

#########################################################
# Set Working Directory:
# - Ensure RELATIVE working directory (so it can be replicated by any user)
# - Ensure users can read data using either Windows or UNIX folders
# - Working directory should be '.\scripts' for windows or './scripts' for UNIX
#########################################################

working_dir = set_wd.set_correct_working_dir()

#########################################################
# Get augmented models:
# - Can only run 3/5 of the best models, 2 models not uploaded to git as too much memory.
#########################################################

model1 = tf.keras.models.load_model(r'best_models\cnn_5lyr_rmsprpmean_squared_error0.001')
model2 = tf.keras.models.load_model(r'best_models\cnn_5lyr_rmsprpmean_squared_logarithmic_error0.001')
model3 = tf.keras.models.load_model(r'best_models\Aug_LeNet_mean_squared_error_0.01')
model4 = tf.keras.models.load_model(r'best_models\Aug_LeNet_binary_crossentropy_0.001')

model_list1 = [[model1, "cnn_5lyr_rmsprpmean_squared_error0.001"], [model2, "cnn_5lyr_rmsprpmean_squared_logarithmic_error0.001"]]
model_list2 = [[model3, "Aug_LeNet_mean_squared_error_0.01"], [model4, "Aug_LeNet_binary_crossentropy_0.001"]]

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

output1 = get_validation_results(model_list= model_list1, batch_size=128, target_size=(300,300))
output2 = get_validation_results(model_list= model_list2, batch_size=128, target_size=(32,32))
output = output1.append(output2)
accuracy1 = output1[0]
accuracy2 = output2[0]

# Get remaining 2 models:
remaining_models = pd.read_csv(r".\augmented_CNN_results.csv")

############################################################
# Graph data
############################################################

df_results = pd.DataFrame()
all_results = []
all_models = []

# Lenet
for i in accuracy1:
    all_results.append(i[1])
    all_models.append(i[0])

for i in accuracy2:
    all_results.append(i[1])
    all_models.append(i[0])

df_results['ACCURACY'] = all_results
df_results['MODEL'] = all_models
df_results1 = df_results.sort_values('ACCURACY')
df_results1 = df_results1.reset_index()
df_results1 = df_results1[['ACCURACY', 'MODEL']]

# bind:
all_results = pd.concat([df_results1, remaining_models])
all_results = all_results.sort_values('ACCURACY')
all_results = all_results.reset_index()
all_results = all_results[['ACCURACY', 'MODEL']]

plot = all_results.plot(kind='bar', title="Retrained models on Augmented Data")
plot.set_xlabel("CNN Model")
plot.set_ylabel("Classification Accuracy")
