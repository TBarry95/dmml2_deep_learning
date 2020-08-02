#########################################################
# DES: Main file for automating execution of project.
#      Runs the following jobs:
#      - import_data.py:
#      - cnn_comparison.py:
#########################################################

import os
import scripts.set_working_dir as set_wd

#########################################################
# Set Working Directory:
# - Ensure RELATIVE working directory (so it can be replicated by any user)
# - Ensure users can read data using either Windows or UNIX folders
# - Working directory should be '.\scripts' for windows or './scripts' for UNIX
#########################################################

working_dir = set_wd.set_correct_working_dir()

#########################################################
# import_data.py: Run first job
#########################################################

import scripts.import_data

#########################################################
# cnn_comparison.py:
#########################################################


