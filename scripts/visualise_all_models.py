#########################################################
# DES: Run multiple CNNs and find optimal results:
# BY: Tiernan Barry
#########################################################

import os
import scripts.set_working_dir as set_wd
import pandas as pd
from tabulate import tabulate

#########################################################
# Set Working Directory:
# - Ensure RELATIVE working directory (so it can be replicated by any user)
# - Ensure users can read data using either Windows or UNIX folders
# - Working directory should be '.\scripts' for windows or './scripts' for UNIX
#########################################################

working_dir = set_wd.set_correct_working_dir()

#########################################################
# Get data:
# - Read in all csv exports from each models
#########################################################

cnn5_results = pd.read_csv(r".\cnn5_results.csv")
lenet_results = pd.read_csv(r".\leNet_results.csv") #
alexnet_results = pd.read_csv(r".\Alex_Net_Results.csv")
goognet_results = pd.read_csv(r".\googleNet_results.csv")
squeeze_results = pd.read_csv(r".\squeezeNet_results.csv")
vgg_results = pd.read_csv(r".\VGG16_results.csv")
fkcnn_results = pd.read_csv(r".\fk_CNN_results.csv")

# combine:
all_results = pd.concat([cnn5_results, lenet_results, alexnet_results, goognet_results, squeeze_results, vgg_results, fkcnn_results])
all_results = all_results[['ACCURACY', 'MODEL']]
all_results = all_results.sort_values('ACCURACY')
all_results1 = all_results.reset_index()
all_results1 = all_results1[['ACCURACY', 'MODEL']]

#########################################################
# Plot models:
#########################################################

plot = all_results1.plot(kind='line', title="Accuracy of all CNN models")
plot.set_xlabel("CNN Model Reference")
plot.set_ylabel("CNN Model Accuracy")

#########################################################
# Get best models:
#########################################################

best_model = all_results1[all_results1['ACCURACY'] == max(all_results1['ACCURACY'])]
best_5_models = all_results1.tail(5)
print(tabulate(best_5_models, headers=best_5_models.columns))

# get top 5 results:
all_scores = set(all_results1['ACCURACY'])
all_scores = list(all_scores)
top_5 = all_scores[len(all_scores)-5:len(all_scores)]

# export csv of best models:
top5_models = all_results1[all_results1['ACCURACY'].isin(top_5)]
top5_models.to_csv(r"./top5_models.csv", index=False)
print("Total number of models analysed: ", len(all_results1))
