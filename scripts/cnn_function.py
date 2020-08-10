############################################################
# DES: Define a 5 layer CNN which can be adjusted based on the following paramters:
#      - Type of loss function
#      - Type of optimisation
#      - Activation function (default = relu)
############################################################

############################################################
# Libraries:
############################################################

import os
import scripts.set_working_dir as set_wd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import math

#########################################################
# Set Working Directory:
# - Ensure RELATIVE working directory (so it can be replicated by any user)
# - Ensure users can read data using either Windows or UNIX folders
# - Working directory should be '.\scripts' for windows or './scripts' for UNIX
#########################################################

working_dir = set_wd.set_correct_working_dir()

############################################################
# Build and train model:
# Define function for building a 5 layer CNN for our target dataset with the following inputs:
# - Type of loss function
# - Type of optimisation
# - Activation function (default = relu)
# ** areas to work on: Defining number of layers, filters, max pooling etc
############################################################

def cnn_5_layers(loss, optimizer, activation = 'relu'):
    """
    :param loss: Choose which methodology for calculating the loss function of the NN.
                 Exhaustive list is provided below from Keras documentation: https://keras.io/api/losses/

    :param optimizer: Choose which methodology for optimising the NN.
                      Exhaustive list is provided below from Keras documentation: https://keras.io/api/optimizers/
                      Can either be applied by string with default param applied, or by instantiating custom.

    :param activation: Choose which activation method used for NN model.
                      Exhaustive list is provided below from Keras documentation: https://keras.io/api/layers/activations/

    :return: The trained NN model
    """

    ##################################
    # Define model:
    ##################################

    cnn_model = tf.keras.models.Sequential([

        # Parameters to consider:
        # - Number of layers to NN
        # - Number of filters per NN
        # - Dimensions of filter (kernel)
        # - Max pooling: Reduces the dimesnionality of images by reducing pixels from output of previous layer
        #                Pooling layers are used to reduce the dimensions of the feature maps.
        #                Thus, it reduces the number of parameters to learn and the amount of computation performed in the network.

        # 1st layer (verbose)
        tf.keras.layers.Conv2D(filters = 16,
                               kernel_size = (3, 3),
                               activation = activation,
                               input_shape = (300, 300, 3) # x*x pixels, 3 bytes of colour
                               ),
        tf.keras.layers.MaxPooling2D(2, 2), # each layer will result in half the width x half height
                                            # exports new shape = 150x150
        # 2nd layer:
        tf.keras.layers.Conv2D(32,  (3, 3), activation = activation),
        tf.keras.layers.MaxPooling2D(2, 2), # exports new shape = 75 x 75

        # 3rd layer:
        tf.keras.layers.Conv2D(64, (3, 3), activation = activation),
        tf.keras.layers.MaxPooling2D(2, 2) , # exports new shape = 32 x 32

        # 4th layer:
        tf.keras.layers.Conv2D(64, (3, 3), activation = activation),
        tf.keras.layers.MaxPooling2D(2, 2),

        # 5th layer:
        tf.keras.layers.Conv2D(64, (3, 3), activation = activation),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(512, activation = 'relu'),  # 512 neuron hidden layer

        # Only 1 output neuron = 'normal' and 1 'pneumonia'
        tf.keras.layers.Dense(1, activation='sigmoid') 
    ])

    model_summary = cnn_model.summary()
    print(model_summary)

    sgd = tf.keras.optimizers.SGD(learning_rate = optimizer, momentum=0.0, nesterov=False, name='SGD')

    cnn_model.compile(loss = loss,
                      optimizer = sgd,
                      metrics = ['accuracy'])

    ############################################################
    # Train and Test Model:
    ############################################################

    batch_size = 128
    training_size = 2213
    testing_size = 2801 - training_size
    epochs = 5

    fn_steps_per_epoch = lambda x: int(math.ceil(1. * x / batch_size))
    steps_per_epoch = fn_steps_per_epoch(training_size)
    test_steps = fn_steps_per_epoch(testing_size)

    # Extract dataset from folder:
    train_datagen = ImageDataGenerator(rescale=1 / 255)
    test_datagen = ImageDataGenerator(rescale=1 / 255)

    # get training images
    train_gen = train_datagen.flow_from_directory(
        r'.\cleaned_data\train',
        target_size=(300, 300),
        batch_size=batch_size,
        class_mode='binary'
    )

    # get testing images
    test_gen = test_datagen.flow_from_directory(
        r'.\cleaned_data\test',
        target_size=(300, 300),
        batch_size=batch_size,
        class_mode='binary'
    )

    # train model
    history = cnn_model.fit(
        train_gen,
        steps_per_epoch = steps_per_epoch,
        epochs = epochs,
        validation_data = test_gen,
        validation_steps = test_steps
    )

    model_name_loc = r".\saved_models\cnn_5layer_" + str(loss) + str(optimizer)
    cnn_model.save(model_name_loc)

    return r".\saved_models\cnn_5layer_" + str(loss) + str(optimizer)

