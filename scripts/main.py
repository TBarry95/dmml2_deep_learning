#########################################################
# DES: Main file for automating execution of project.
#      Runs the following jobs:
#########################################################

import sys

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

print("#########################################################")
print("# Ensuring correct working directory:")
print("#########################################################")

working_dir = set_wd.set_correct_working_dir()

#########################################################
# clean_data.py: import data and clean image files into new folder (cleaned_data)
#########################################################

print("#########################################################")
print("# Cleaning data:")
print("#########################################################")

try:
    import clean_data
except:
    import scripts.clean_data

#########################################################
# Modelling: Run various CNN models and print results
# - LeNet CNN
# - AlexNet CNN
# -
#########################################################

print("#########################################################")
print("# Neural Network Modelling:")
print("#########################################################")

#########################################
# LeNet CNN:
# 9 models are ran by combining the following 3x3 paramters:
# - Loss functions: 'binary_crossentropy', 'mean_squared_error', 'mean_squared_logarithmic_error'
# - Gradient Descent Learning Rate: 0.1, 0.001, 0.001
#########################################

print("# LeNet CNN Validation Results:")

try:
    import load_LeNet
except:
    import scripts.load_LeNet

#########################################
# AlexNet CNN:
#
#########################################


