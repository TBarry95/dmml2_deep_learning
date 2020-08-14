############################################################
# Importing Libraries:
############################################################
import math
import tensorflow as tf
from itertools import product
import numpy as np
import tensorflow.keras
from keras.preprocessing.image import ImageDataGenerator
import set_working_dir as set_wd
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D

np.random.seed(1000)

working_dir = set_wd.set_correct_working_dir()

# Create function to construct Alex_Net
def create_AlexNet(number_of_classes, loss_func, learning_rate):
    # Create AlexNet structure
    alex_model = tf.keras.models.Sequential([
            # Creating the 1st Convolutional Layer
            tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4, activation='relu', input_shape=(224,224,3)),
            # Max Pooling
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
            # Creating the 2nd Convolutional Layer
            tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same', activation='relu'),
            # Max Pooling
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
            # Creating the 3rd Convolutional Layer
            tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                                   activation='relu'),
            # Creating the 4th Convolutional Layer
            tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                                   activation='relu'),
            # Creating the 5th Convolutional Layer
            tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same',
                                   activation='relu'),
            # Max Pooling
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
            # Passing it to a Fully Connected layer
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096,input_shape=(224*224*3,), activation='relu'),
            # Droput used to reduce overfitting
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(4096, activation='relu'),
            # Droput used to reduce overfitting
            tf.keras.layers.Dropout(0.5),
            # Define the output layer, setting the number of classes
            tf.keras.layers.Dense(number_of_classes)
        ])

    # Setting the gradient descent learning rate
    var_SGD = tf.keras.optimizers.SGD(learning_rate=learning_rate,
                                      momentum=0.0,
                                      nesterov=False,
                                      name='SGD')
    batch_size = 128

    # Compile the model
    alex_model.compile(loss=loss_func, optimizer=var_SGD, metrics=["accuracy"])

    # Extract dataset from folder:
    train_datagen = ImageDataGenerator(rescale=1/255)
    test_datagen = ImageDataGenerator(rescale=1 / 255)

    # Fetch training images
    train_gen = train_datagen.flow_from_directory(
        r'.\cleaned_data\train',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary'
    )

    # Fetch test images
    test_gen = test_datagen.flow_from_directory(
        r'.\cleaned_data\test',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary'
    )

    # Define parameters for model fitting
    training_size = 2148
    testing_size = 2686 - training_size
    epochs = 5

    fn_steps_per_epoch = lambda x: int(math.ceil(1. * x / batch_size))
    steps_per_epoch = fn_steps_per_epoch(training_size)
    test_steps = fn_steps_per_epoch(testing_size)

    # Fit Alex_Net model
    alex_model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=test_gen,
        validation_steps=test_steps
    )

    # Save model to folder
    model_folder = r".\saved_models\AlexNet_" + str(loss_func) + str(learning_rate)
    alex_model.save(model_folder)
