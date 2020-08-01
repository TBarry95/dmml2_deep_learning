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
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop

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

    :param activation:

    :return: The NN model
    """
    ##################################
    # Available loss metrics:
    ##################################
    # Probabilistic losses
    # - BinaryCrossentropy class
    # - CategoricalCrossentropy class
    # - SparseCategoricalCrossentropy class
    # - Poisson class
    # - binary_crossentropy function
    # - categorical_crossentropy function
    # - sparse_categorical_crossentropy function
    # - poisson function
    # - KLDivergence class
    # - kl_divergence function
    # Regression losses
    # - MeanSquaredError class
    # - MeanAbsoluteError class
    # - MeanAbsolutePercentageError class
    # - MeanSquaredLogarithmicError class
    # - CosineSimilarity class
    # - mean_squared_error function
    # - mean_absolute_error function
    # - mean_absolute_percentage_error function
    # - mean_squared_logarithmic_error function
    # - cosine_similarity function
    # - Huber class
    # - huber function
    # - LogCosh class
    # - log_cosh function
    # Hinge losses for "maximum-margin" classification
    # - Hinge class
    # - SquaredHinge class
    # - CategoricalHinge class
    # - hinge function
    # - squared_hinge function
    # - categorical_hinge function

    ##################################
    # Available optimizers:
    ##################################
    # - SGD
    # - RMSprop
    # - Adam
    # - Adadelta
    # - Adagrad
    # - Adamax
    # - Nadam
    # - Ftrl

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

        # 6th layer:
        #tf.keras.layers.Conv2D(64, (3, 3), activation = activation),
        #tf.keras.layers.MaxPooling2D(2, 2),

        # 7th layer:
        #tf.keras.layers.Conv2D(64, (3, 3), activation = activation),
        #tf.keras.layers.MaxPooling2D(2, 2),

        # 8th layer:
        #tf.keras.layers.Conv2D(64, (3, 3), activation = activation),
        #tf.keras.layers.MaxPooling2D(2, 2),

        # 9th layer:
        #tf.keras.layers.Conv2D(64, (3, 3), activation = activation),
        #tf.keras.layers.MaxPooling2D(2, 2),

        # 10th layer:
        #tf.keras.layers.Conv2D(64, (3, 3), activation = activation),
        #tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(512, activation = 'relu'),  # 512 neuron hidden layer

        # Only 1 output neuron = 'normal' and 1 'pneumonia'
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model_summary = cnn_model.summary()
    print(model_summary)

    cnn_model.compile(loss = loss,
                      optimizer = optimizer,
                      metrics = ['accuracy'])

    ############################################################
    # Train and Test Model:
    ############################################################

    # Extract dataset from folder:
    train_datagen = ImageDataGenerator(rescale = 1/255)
    test_datagen = ImageDataGenerator(rescale = 1/255)

    # get training images
    train_gen = train_datagen.flow_from_directory(
        r'.\cleaned_data\train',
        target_size=(300, 300),
        batch_size=128,
        class_mode='binary'
    )

    # get testing images
    test_gen = test_datagen.flow_from_directory(
        r'.\cleaned_data\test',
        target_size=(300, 300),
        batch_size=128,
        class_mode='binary'
    )

    # train model
    history = cnn_model.fit(
        train_gen,
        steps_per_epoch=10,
        epochs=5,
        validation_data=test_gen
    )

    ############################################################
    # Validate Model: get final results
    ############################################################

    # load new unseen dataset
    validation_datagen = ImageDataGenerator(rescale = 1 / 255)

    val_generator = validation_datagen.flow_from_directory(
        r'.\cleaned_data\validate',
        target_size=(300, 300),
        batch_size=128,
        class_mode='binary'
    )

    eval_result = cnn_model.evaluate_generator(val_generator, 624)
    print('loss rate at evaluation data :', eval_result[0])
    print('accuracy rate at evaluation data :', eval_result[1])

    return eval_result

