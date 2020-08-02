############################################################
# DES: Ensure that all scripts are ran from the correct location:
#      Correct location = '.\scripts'
# BY: Tiernan Barry
############################################################

############################################################
# Libraries:
############################################################

import os

#########################################################
# Set Working Directory:
# - Ensure RELATIVE working directory (so it can be replicated by any user)
# - Ensure users can read data using either Windows or UNIX folders
# - Working directory should be '.\scripts' for windows or './scripts' for UNIX
#########################################################

def set_correct_working_dir():

    current_dir = os.getcwd()
    print("Current directory: ", current_dir)

    if current_dir[len(current_dir)-7:len(current_dir)] != 'scripts':
        try:
            os.chdir(r".\scripts")
            print("Changing working directory to: ", os.getcwd())
            print("New working directory: ", os.getcwd())
        except:
            print(r"Can't find .\scripts folder, will try '/scripts' instead (Windows v UNIX) ")

            try:
                os.chdir(r"./scripts")
                print("Changing working directory to: ", os.getcwd())
                print("New working directory: ", os.getcwd())
            except:
                print(r"Still can't find correct directory, continuing script anyway")
    else:
        print("Working directory already correct: ", os.getcwd())

    working_dir = os.getcwd()

    return working_dir