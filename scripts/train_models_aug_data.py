#########################################################
# DES: Run top models on augmented data
# BY: Tiernan Barry
#########################################################

import scripts.set_working_dir as set_wd
import pandas as pd
import scripts.cnn_function as cnn_fns
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import math
#########################################################
# Set Working Directory:
# - Ensure RELATIVE working directory (so it can be replicated by any user)
# - Ensure users can read data using either Windows or UNIX folders
# - Working directory should be '.\scripts' for windows or './scripts' for UNIX
#########################################################

working_dir = set_wd.set_correct_working_dir()

#########################################################
# Get best models:
#########################################################

best_models = pd.read_csv(r"top5_models.csv")
best2_models = best_models.tail(2)

#########################################################
# Load models:
#########################################################

model_1 =
