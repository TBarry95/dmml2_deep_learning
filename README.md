# A Deployment of Deep Learning for Detecting COVID-19 in Chest X-Rays.

By Tiernan Barry, Frank Kelly, Eoghan O’Connor, Christopher Leckey 

## Summary of project:
This project utilises freely available chest X-Ray images for the detection of Viral Pneumonia - a sympton of COVID-19 - by deploying Deep Learning algorithims. 

## To reproduce this repository using Pycharm:
- Close current Pycharm project if open. 
- Checkout from Version Control, and enter: https://github.com/TBarry95/dmml2_deep_learning

## To reproduce this repository using terminal:
- git clone https://github.com/TBarry95/dmml2_deep_learning

## Key files for reproducing analysis: 
- All models are trained and immediately exported into the folder 'saved_models'.
  - For example, all LeNet CNN's are trained and exported in train_LeNet.py.
  - The same structure applies to all CNN's examined (train_GoogleNet.py etc). 
- All models are then loaded in order to run the validation dataset. 
  - For example, to get the accuracy of each LeNet CNN, run load_LeNet.py (same for other CNNs).
- To streamline this process and run all models validation, run: main_validate.py
- To run all "best models" (which were retrained on augmented data) on the validation dataset, run: main_augmented.py
