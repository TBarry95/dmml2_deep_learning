# DES: test NN example on dummy dataset
# BY: Tiernan Barry

import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.datasets import load_digits
import pylab as pl
import random
from sklearn import ensemble


#########################################################
# Get data:
#########################################################

digits = load_digits()

pl.gray()
pl.matshow(digits.images[0])
pl.show()

images_and_labels = list(zip(digits.images, digits.target))

plt.figure(figsize=(5,5))

for index, (image, label) in enumerate(images_and_labels[:15]):
    plt.subplot(3, 5, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('%i' % label)

#Define variables
n_samples = len(digits.images)
x = digits.images.reshape((n_samples, -1))
y = digits.target

#Create random indices
sample_index = random.sample(range(0,len(x)),int(len(x)/5)) #20-80
valid_index=[i for i in range(len(x)) if i not in sample_index]

#Sample and validation images
sample_images=[x[i] for i in sample_index]
valid_images=[x[i] for i in valid_index]

#Sample and validation targets
sample_target=[y[i] for i in sample_index]
valid_target=[y[i] for i in valid_index]

# Using the NN:
classifier_nn = MLPClassifier(activation = 'relu', # default
                              solver = 'lbfgs', # default = adam
                              learning_rate = 'constant', # default
                              shuffle = True,
                              random_state = 1,
                              verbose = False
                              )

#Fit model with sample data
classifier_nn.fit(sample_images, sample_target)

#Attempt to predict validation data
score=classifier_nn.score(valid_images, valid_target)
print( 'NN score:\n')
print( 'Score\t'+str(score))

# Chest x ray dataset:

