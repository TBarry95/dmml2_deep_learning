############################################################
# DES:
############################################################

############################################################
# Libraries:
############################################################

# from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing import image
import os
from sklearn.model_selection import train_test_split
from PIL import Image
from IPython.display import display


#########################################################
# Set Working Directory:
# - Ensure RELATIVE working directory (so it can be replicated by any user)
# - Ensure users can read data using either Windows or UNIX folders
# - Working directory should be '.\scripts' for windows or './scripts' for UNIX
#########################################################

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

############################################################
# Model building:
############################################################

model = tf.keras.models.Sequential([

    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    # The second convolution
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # The third convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),  # 512 neuron hidden layer
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for ('normal') clas and 1 for ('pneumonia') class
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics = ['accuracy'])

############################################################
# Model Training:
############################################################

train_datagen = ImageDataGenerator(rescale = 1/255)
test_datagen = ImageDataGenerator(rescale = 1/255)

train_gen = train_datagen.flow_from_directory(
    r'.\cleaned_data\train',
    target_size = (300,300),
    batch_size = 128,
    class_mode = 'binary'
)

test_gen = train_datagen.flow_from_directory(
    r'.\cleaned_data\test',
    target_size = (300,300),
    batch_size = 128,
    class_mode = 'binary'
)

# training the model
history = model.fit(
    train_gen,
    steps_per_epoch = 10,
    epochs = 10,
    validation_data = test_gen
)

############################################################
# Model Evaluation:
############################################################



############################################################
# Final Model Prediction:
############################################################






