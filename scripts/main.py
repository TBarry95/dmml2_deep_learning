#########################################################
# DES: Main file for automating execution of project.
#      Runs the following jobs:
#      - Clean data (clean_data.py)
#      - Validate models:
#         - LeNet CNN
#         - AlexNet CNN
#         - VGGNet16 CNN
#         - GoogleNet CNN
#         - Alternative CNNs:
#           - 5 layers
#       - Plots initial results of best models
#      **Exception handling designed to run either from console or from correct working directory.
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
# Modelling: Run CNN models and print results
# - LeNet CNN
# - AlexNet CNN
# - VGGNet16
# - GoogleNet
# for each algorithim, 9 models are ran by combining the following 3x3 paramters:
# - Loss functions: 'binary_crossentropy', 'mean_squared_error', 'mean_squared_logarithmic_error'
# - Gradient Descent Learning Rate: 0.1, 0.001, 0.001
#########################################################

print("#########################################################")
print("# Run Neural Networks Models on Validation data:")
print("#########################################################")

#########################################
# LeNet CNN:
#########################################

print("# Running LeNet CNN Validation Results:")
try:
    import load_LeNet
except:
    import scripts.load_LeNet

#########################################
# googleNet CNN:
#########################################

print("# Running googleNet CNN Validation Results:")
try:
    import load_GoogLeNet
except:
    import scripts.load_GoogLeNet

#########################################
# SqueezeNet CNN:
#########################################

print("# Running SqueezeNet CNN Validation Results:")
try:
    import load_SqueezeNet
except:
    import scripts.load_SqueezeNet

#########################################
# AlexNet CNN:
#
#########################################



#########################################
# CNN 5 layer:
#########################################

print("# Running CNN 5 layer Validation Results:")
try:
    import load_cnn5layer
except:
    import scripts.load_cnn5layer

