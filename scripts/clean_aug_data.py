#########################################################
# DES: Import raw dataset from '.\scripts\input_files' and export cleaned target dataset into 'cleaned_data' folder.
#      Removing all bacterial pneumonia images for the following reasons:
#      - Balance the dataset 50:50 between pneumonia / normal
#      - COVID is viral
# BY: Tiernan Barry
#########################################################

import glob
import re
import shutil
from sklearn.model_selection import train_test_split

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

#########################################################
# Import data and add to new folder:
#########################################################

#######################################
# Extract pictures into lists:
#######################################

if working_dir[len(working_dir)-8:len(working_dir)] == '/scripts':
    # UNIX: Get list of all files in each directory:
    all_files_norm = glob.glob(r"./augmented_data/normal/*g")
    all_files_pneu = glob.glob(r"./augmented_data/viral/*g")
else:
    # Windows: Get list of all files in each directory:
    all_files_norm = glob.glob(r".\augmented_data\normal\*g")
    all_files_pneu = glob.glob(r".\augmented_data\viral\*g")

print("Number of NORMAL images: ", len(set(all_files_norm)))
print("Number of PNEUMONIA images: ", len(set(all_files_pneu)))

#######################################
# Export new folder: Cleaned data:
# Test, Train, Validate: Each with Normal and Viral datasets loaded.
#######################################

normal_train, normal_test = train_test_split(all_files_norm, test_size=0.2, random_state=0)
viral_train, viral_test = train_test_split(all_files_pneu, test_size=0.2, random_state=0)

for i in normal_train:
    shutil.copy(i, r'.\augmented_data_clean\train\normal')

for i in normal_test:
    shutil.copy(i, r'.\augmented_data_clean\test\normal')

for i in viral_train:
    shutil.copy(i, r'.\augmented_data_clean\train\viral')

for i in viral_test:
    shutil.copy(i, r'.\augmented_data_clean\test\viral')