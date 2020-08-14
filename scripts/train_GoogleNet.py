############################################################
# Libraries:
############################################################
from IPython.utils import capture
from keras.models import Model
from keras.layers import Input, Dense, Conv2D
from keras.layers import Flatten, MaxPool2D, AvgPool2D
from keras.layers import Concatenate, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras.backend as k
import math
import tensorflow as tf
from itertools import product
import scripts.set_working_dir as set_wd

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

#########################################################
# Set Working Directory:
# - Ensure RELATIVE working directory (so it can be replicated by any user)
# - Ensure users can read data using either Windows or UNIX folders
# - Working directory should be '.\scripts' for windows or './scripts' for UNIX
#########################################################

working_dir = set_wd.set_correct_working_dir()


############################################################
# Build model:
############################################################

def googlenet(input_shape, n_classes):
    def inception_block(x, f):
        t1 = Conv2D(f[0], 1, activation='relu')(x)

        t2 = Conv2D(f[1], 1, activation='relu')(x)
        t2 = Conv2D(f[2], 3, padding='same', activation='relu')(t2)

        t3 = Conv2D(f[3], 1, activation='relu')(x)
        t3 = Conv2D(f[4], 5, padding='same', activation='relu')(t3)

        t4 = MaxPool2D(3, 1, padding='same')(x)
        t4 = Conv2D(f[5], 1, activation='relu')(t4)

        output = Concatenate()([t1, t2, t3, t4])
        return output

    input = Input(input_shape)

    x = Conv2D(64, 7, strides=2, padding='same', activation='relu')(input)
    x = MaxPool2D(3, strides=2, padding='same')(x)

    x = Conv2D(64, 1, activation='relu')(x)
    x = Conv2D(192, 3, padding='same', activation='relu')(x)
    x = MaxPool2D(3, strides=2)(x)

    x = inception_block(x, [64, 96, 128, 16, 32, 32])
    x = inception_block(x, [128, 128, 192, 32, 96, 64])
    x = MaxPool2D(3, strides=2, padding='same')(x)

    x = inception_block(x, [192, 96, 208, 16, 48, 64])
    x = inception_block(x, [160, 112, 224, 24, 64, 64])
    x = inception_block(x, [128, 128, 256, 24, 64, 64])
    x = inception_block(x, [112, 144, 288, 32, 64, 64])
    x = inception_block(x, [256, 160, 320, 32, 128, 128])
    x = MaxPool2D(3, strides=2, padding='same')(x)

    x = inception_block(x, [256, 160, 320, 32, 128, 128])
    x = inception_block(x, [384, 192, 384, 48, 128, 128])

    x = AvgPool2D(7, strides=1)(x)
    x = Dropout(0.4)(x)

    x = Flatten()(x)
    output = Dense(n_classes, activation='softmax')(x)

    model = Model(input, output)
    return model


############################################################
# Model Summary:
############################################################

INPUT_SHAPE = 224, 224, 3
N_CLASSES = 2

k.clear_session()
model = googlenet(INPUT_SHAPE, N_CLASSES)
model.summary()


#########################################
# Define combinations of parameters:
#########################################

# Loss functions
loss_fns = ['binary_crossentropy', 'mean_squared_error', 'mean_squared_logarithmic_error']

# Optimisation for SGD learning rate:
opts = [0.001]

# combinations:
combos = list(product(loss_fns, opts))

#SVG(model_to_dot(model).create(prog='dot', format='svg'))


############################################################
# Train model:
############################################################

for i in combos:
    ############################################################
    # Define Constants:
    ############################################################
    batch_size = 128
    training_size = 2148
    testing_size = 2686 - training_size
    epochs = 5

    fn_steps_per_epoch = lambda x: int(math.ceil(1. * x / batch_size))
    steps_per_epoch = fn_steps_per_epoch(training_size)
    test_steps = fn_steps_per_epoch(testing_size)

    ############################################################
    # Extract Data:
    ############################################################
    train_datagen = ImageDataGenerator(rescale=1 / 255)
    test_datagen = ImageDataGenerator(rescale=1 / 255)

    # get training images
    train_gen = train_datagen.flow_from_directory(
        r'.\cleaned_data\train',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary'
    )

    # get testing images
    test_gen = test_datagen.flow_from_directory(
        r'.\cleaned_data\test',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary'
    )

    sgd = tf.keras.optimizers.SGD(learning_rate=i[1], momentum=0.0)

    model.compile(
        loss=i[0],
        optimizer=sgd,
        metrics=['accuracy']
    )

    history = model.fit(train_gen,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        validation_data=test_gen,
                        validation_steps=test_steps
                        )

    model_name_loc = r".\saved_models\GoogLeNet_" + str(i[0]) + str(i[1])

    model.save(model_name_loc)
