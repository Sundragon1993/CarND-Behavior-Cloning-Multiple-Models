import os
import cv2
import math
import numpy as np
import pandas as pd
import csv
import time
import argparse
import json
from keras.optimizers import Adam
from keras.callbacks import Callback
import matplotlib.image as mpimg
import preprocess_input
from tensorflow.python.keras.metrics import Metric

# Angle offset for the left and right cameras. It's and estimation of the
# additional steering angle (normalized to [-1,1]) that we would have to steer
# if the center camera was in the position of the left or right one
ANGLE_OFFSET = 0.25

# Angle offsets applied to center, left and right image
ANGLE_OFFSETS = [0.0, ANGLE_OFFSET, -ANGLE_OFFSET]

# Batch size
BATCH_SIZE = 64
input_shape = preprocess_input.FINAL_IMG_SHAPE


def random_horizontal_flip(x, y):
    flip = np.random.randint(2)

    if flip:
        x = cv2.flip(x, 1)
        y = -y

    return x, y


def random_translation(img, steering):
    # Maximum shift of the image, in pixels
    trans_range = 50  # Pixels

    # Compute translation and corresponding steering angle
    tr_x = np.random.uniform(-trans_range, trans_range)
    steering = steering + (tr_x / trans_range) * ANGLE_OFFSET

    # Warp image using the computed translation
    rows = img.shape[0]
    cols = img.shape[1]

    M = np.float32([[1, 0, tr_x], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (cols, rows))

    return img, steering


def data_augmentation(x, y):
    # Random horizontal shift
    x, y = random_translation(x, y)

    # Random flip
    x, y = random_horizontal_flip(x, y)

    return x, y


def train_generator(X, y, batch_size):
    """ Provides a batch of images from a log file. The main advantage
        of using a generator is that we do not need to read the whole log file,
        only one batch at a time, so it will fit in RAM.
        This function also generates extended data on the fly. """
    # Supply training images indefinitely
    while 1:
        # Declare output data
        x_out = []
        y_out = []

        # Fill batch
        for i in range(0, batch_size):
            # Get random index to an element in the dataset.
            idx = np.random.randint(len(y))

            # Randomly select which of the 3 images (center, left, right) to use
            idx_img = np.random.randint(len(ANGLE_OFFSETS))

            # Read image and steering angle (with added offset)
            x_i = mpimg.imread('./data/' + X[idx][idx_img].strip())
            y_i = y[idx] + ANGLE_OFFSETS[idx_img]

            # Preprocess image
            x_i = preprocess_input.main(x_i)

            # Augment data
            x_i, y_i = data_augmentation(x_i, y_i)

            # Add to batch
            x_out.append(x_i)
            y_out.append(y_i)

        yield (np.array(x_out), np.array(y_out))


def val_generator(X, y):
    """ Provides images for validation. This generator is different
        from the previous one in that it does **not** perform data augmentation:
        it just reads images from disk, preprocess them and yields them """
    # Validation generator
    while 1:
        for i in range(len(y)):
            # Read image and steering angle
            x_out = mpimg.imread('./data/' + X[i][0].strip())
            y_out = np.array([[y[i]]])

            # Preprocess image
            x_out = preprocess_input.main(x_out)
            x_out = x_out[None, :, :, :]

            # Return the data
            yield x_out, y_out


def make_multiple(x, number):
    """ Increases x to be the smallest multiple of number """
    return int(math.ceil(float(x) / float(number)) * number)


def normalize(X):
    """ Normalizes the input between -0.5 and 0.5 """
    return X / 255. - 0.5


from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation, Cropping2D, Merge
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
# from keras.layers.merge import concatenate
from keras.preprocessing.image import load_img, img_to_array
from keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
from config import TrainConfig


def create_comma_model_relu():
    model = Sequential()

    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    print('Model is created and compiled..')
    return model


def create_comma_model_lrelu():
    model = Sequential()

    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", input_shape=input_shape))
    model.add(LeakyReLU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(LeakyReLU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    # model.add(Dropout(.5))
    model.add(LeakyReLU())
    model.add(Dense(512))
    # model.add(Dropout(.5))
    model.add(LeakyReLU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    print('Model is created and compiled..')
    return model


def create_comma_model_prelu():
    model = Sequential()

    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", input_shape=input_shape))
    model.add(PReLU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(PReLU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    # model.add(Dropout(.5))
    model.add(PReLU())
    model.add(Dense(512))
    # model.add(Dropout(.5))
    model.add(PReLU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    print('Model is created and compiled..')
    return model


def create_comma_model2():
    # additional dense layer

    model = Sequential()

    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    print('Model is created and compiled..')
    return model


def create_comma_model3():
    # additional conv layer
    model = Sequential()

    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode="same"))
    model.add(Flatten())
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    print('Model is created and compiled..')
    return model


def create_comma_model4():
    # 2 additional conv layers
    model = Sequential()

    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode="same"))
    model.add(Flatten())
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    print('Model is created and compiled..')
    return model


def create_comma_model5():
    # more filters in first 2 conv layers
    model = Sequential()

    model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode="same", input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    print('Model is created and compiled..')
    return model


def create_comma_model6():
    # remove one conv layer
    model = Sequential()

    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    print('Model is created and compiled..')
    return model


def create_comma_model_bn():
    model = Sequential()

    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Flatten())

    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    print('Model is created and compiled..')
    return model


def create_nvidia_model1():
    model = Sequential()

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same", input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    print('Model is created and compiled..')
    return model


def create_nvidia_model2():
    model = Sequential()

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same", input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    print('Model is created and compiled..')
    return model


def create_nvidia_model3():
    model = Sequential()

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same", input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.summary()
    model.compile(optimizer="adam", loss="mse")

    print('Model is created and compiled..')
    return model


def create_comma_model_large():
    # Parameters
    model = Sequential()

    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", input_shape=input_shape))
    # model.add(ELU())
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    # model.add(ELU())
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    # model.add(Dropout(.5))
    # model.add(ELU())
    model.add(Activation('relu'))
    model.add(Dense(1024))
    # model.add(Dropout(.5))
    # model.add(ELU())
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    print('Model is created and compiled..')
    return model


def create_comma_model_large_dropout():
    # Parameters
    weight_init = 'glorot_uniform'
    padding = 'valid'
    model = Sequential()

    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", input_shape=input_shape))
    # model.add(ELU())
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    # model.add(ELU())
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    # model.add(Dropout(.5))
    # model.add(ELU())
    model.add(Activation('relu'))
    model.add(Dense(1024))
    model.add(Dropout(.5))
    # model.add(ELU())
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.summary()
    model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])

    print('Model is created and compiled..')
    return model


def create_rambo_model():
    # First branch
    branch1 = Sequential()
    branch1.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", input_shape=input_shape))
    branch1.add(Activation('relu'))
    branch1.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    branch1.add(Activation('relu'))
    branch1.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    branch1.add(Flatten())
    branch1.add(Activation('relu'))
    branch1.add(Dense(512))
    branch1.add(Activation('relu'))
    branch1.add(Dense(1, input_dim=512))

    # Second branch
    branch2 = Sequential()
    branch2.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same", input_shape=input_shape))
    branch2.add(Activation('relu'))
    branch2.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
    branch2.add(Activation('relu'))
    branch2.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="same"))
    branch2.add(Activation('relu'))
    branch2.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    branch2.add(Activation('relu'))
    branch2.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    branch2.add(Flatten())
    branch2.add(Activation('relu'))
    branch2.add(Dense(100))
    branch2.add(Activation('relu'))
    branch2.add(Dense(50))
    branch2.add(Activation('relu'))
    branch2.add(Dense(10))
    branch2.add(Activation('relu'))
    branch2.add(Dense(1, input_dim=10))

    # Third branch
    branch3 = Sequential()
    branch3.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same", input_shape=input_shape))
    branch3.add(Activation('relu'))
    branch3.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
    branch3.add(Activation('relu'))
    branch3.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="same"))
    branch3.add(Activation('relu'))
    branch3.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    branch3.add(Activation('relu'))
    branch3.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    branch3.add(Activation('relu'))
    branch3.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    branch3.add(Flatten())
    branch3.add(Activation('relu'))
    branch3.add(Dense(100))
    branch3.add(Activation('relu'))
    branch3.add(Dense(50))
    branch3.add(Activation('relu'))
    branch3.add(Dense(10))
    branch3.add(Activation('relu'))
    branch3.add(Dense(1, input_dim=10))

    # merged = concatenate([branch1.output, branch2.output, branch3.output], axis=-1)
    # dense_out = Dense(1, activation='relu')(merged)
    # model = Model(inputs=[branch1.input, branch2.input, branch3.input], outputs=dense_out)
    # Final merge
    model = Sequential()
    model.add(Merge([branch1, branch2, branch3], mode='concat'))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    print('Model is created and compiled..')
    return model


def createPreProcessingLayers():
    """
    Creates a model with the initial pre-processing layers.
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((50, 20), (0, 0))))
    return model


def define_model():
    """ Defines the network architecture, following Nvidia's example on:
        http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf """

    # Parameters
    input_shape = preprocess_input.FINAL_IMG_SHAPE

    weight_init = 'glorot_uniform'
    padding = 'valid'
    dropout_prob = 0.25

    # Define model
    model = Sequential()

    model.add(Lambda(normalize, input_shape=input_shape, output_shape=input_shape))

    model.add(Convolution2D(24, 5, 5,
                            border_mode=padding,
                            init=weight_init, subsample=(2, 2)))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5,
                            border_mode=padding,
                            init=weight_init, subsample=(2, 2)))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5,
                            border_mode=padding,
                            init=weight_init, subsample=(2, 2)))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3,
                            border_mode=padding,
                            init=weight_init, subsample=(1, 1)))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3,
                            border_mode=padding,
                            init=weight_init, subsample=(1, 1)))

    model.add(Flatten())
    model.add(Dropout(dropout_prob))
    model.add(ELU())

    model.add(Dense(100, init=weight_init))
    model.add(Dropout(dropout_prob))
    model.add(ELU())

    model.add(Dense(50, init=weight_init))
    model.add(Dropout(dropout_prob))
    model.add(ELU())

    model.add(Dense(10, init=weight_init))
    model.add(Dropout(dropout_prob))
    model.add(ELU())

    model.add(Dense(1, init=weight_init, name='output'))

    model.summary()

    # Compile it
    model.compile(loss='mse', optimizer=Adam(lr=0.001))

    return model


def save_model(out_dir, model, name_model):
    """ Saves model (json) and weights (h5) to disk """
    print('Saving model in %s...' % out_dir)

    # Create directory if needed
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Save model
    # model_json = model.to_json()
    # with open(os.path.join(out_dir, 'model.json'), 'w+') as f:
    #     json.dump(model_json, f)

    # Save weights
    # model.save_weights(os.path.join(out_dir, 'model.h5'))
    model.save(os.path.join(out_dir, name_model))


class EpochSaverCallback(Callback):
    def __init__(self, out_dir):
        self.out_dir = out_dir

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        current_loss = logs.get('val_loss')

        if not self.losses or current_loss < np.amin(self.losses):
            out_dir = os.path.join(self.out_dir, 'e' + str(epoch + 1))
            save_model(out_dir, self.model)

        self.losses.append(current_loss)


def train_model(model, save_dir, n_epochs, X, y):
    """ Trains model """
    print('Training model...')

    batch_size = BATCH_SIZE

    n_train_samples = 0.005 * make_multiple(len(y), batch_size)
    n_val_samples = len(y)

    gen_train = train_generator(X, y, batch_size)
    gen_val = val_generator(X, y)

    checkpoint_callback = EpochSaverCallback(save_dir)

    history = model.fit_generator(generator=gen_train,
                                  samples_per_epoch=n_train_samples,
                                  validation_data=gen_val,
                                  nb_val_samples=n_val_samples,
                                  nb_epoch=n_epochs,
                                  callbacks=[checkpoint_callback],
                                  verbose=1)
    preprocess_input.visualize(history)


def get_training_data_without_generator(log_file_path):
    lines = []
    with open(log_file_path) as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        for line in reader:
            lines.append(line)

    images = []
    measurements = []
    for line in lines:
        # source_path = line[0]
        # # print(source_path)
        # filename = source_path.split('/')[-1]
        # current_path = './data/IMG/' + filename
        # image = cv2.imread(current_path)
        image_center = cv2.imread(
            "./data/IMG/" + line[0].split('/')[-1])  # left image
        image_left = cv2.imread(
            "./data/IMG/" + line[1].split('/')[-1])  # center image
        image_right = cv2.imread(
            "./data/IMG/" + line[2].split('/')[-1])  # right image
        steering_center = float(line[3])  # steer
        # x_i, y_i = data_augmentation(x_i, y_i)
        # Preprocess image
        # image_center = preprocess_input.main(image_center)
        # image_left = preprocess_input.main(image_left)
        # image_right = preprocess_input.main(image_right)

        correction = 0.2  # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        # x_center, y_center = data_augmentation(image_center, steering_center)
        # x_left, y_left = data_augmentation(image_left, steering_left)
        # x_right, y_right = data_augmentation(image_right, steering_right)
        #
        # images.extend((x_center, x_left, x_right))
        # measurements.extend((y_center, y_left, y_right))
        images.extend((image_center, image_left, image_right))
        measurements.extend((steering_center, steering_left, steering_right))

    X_train = np.array(images)
    y_train = np.array(measurements)
    return X_train, y_train


def train_model_without_generator(model, save_dir, n_epochs, X, y):
    history = model.fit([X, X, X], y, validation_split=0.2, shuffle=True, nb_epoch=n_epochs, batch_size=BATCH_SIZE,
                        verbose=1)
    preprocess_input.visualize(history)


def get_training_data(log_file_path):
    """ Reads the CSV file and splits it into training and validation sets """
    # Read CSV file with pandas
    data = pd.read_csv(log_file_path)
    # data['center'] = ('./data/' + data['center']).strip()
    # data['left'] = ('./data/' + data['left']).strip()
    # data['right'] = ('./data/' + data['right']).strip()

    # Get image paths and steering angles
    X = np.column_stack((np.copy(data['center']), np.copy(data['left']), np.copy(data['right'])))
    y = np.copy(data['steering'])

    return X, y


def build_model(log_file_path, n_epochs, save_dir):
    """ Builds and trains the network given the input data in train_dir """

    # Get training and validation data
    # X, y = get_training_data(log_file_path)
    X, y = get_training_data_without_generator(log_file_path)

    # Build and train the network
    # model = define_model()
    # model = create_comma_model_large_dropout()
    # model = create_comma_model_large()
    model = create_rambo_model()
    # train_model(model, save_dir, n_epochs, X, y)
    train_model_without_generator(model, save_dir, n_epochs, X, y)

    return model


def parse_input():
    """ Sets up the required input arguments and parses them """
    parser = argparse.ArgumentParser()

    parser.add_argument('log_file', help='CSV file of log data')
    parser.add_argument('-e, --n_epochs', dest='n_epochs',
                        help='number of training epochs', metavar='',
                        type=int, default=5)
    parser.add_argument('-o, --out_dir', dest='out_dir', metavar='',
                        default=time.strftime("%Y%m%d_%H%M%S"),
                        help='directory where the model is stored')
    parser.add_argument('-n, --name_model', dest='name_model', metavar='',
                        help='name of the trained model', type=str, default='model.h5')

    return parser.parse_args()


def main():
    """ Main function """
    # Get input
    args = parse_input()

    # Build a model
    model = build_model(args.log_file, args.n_epochs, args.out_dir)

    # Save model
    save_model(args.out_dir, model, args.name_model)

    print('Finished!')


from keras.models import load_model
import h5py
from keras import __version__ as keras_version


def eval_model():
    # args = parse_input()
    # check that model Keras version is same as local Keras version
    f = h5py.File('./Rambo/nvidia_original.h5', mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str('2.2.4').encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

    model = load_model('./Rambo/nvidia_original.h5')
    print(model.summary())
    X, y = get_training_data_without_generator('./data/driving_log.csv')
    X_test = X[20000:]
    Y_test = y[20000:]
    print('\n# Evaluate on test data')
    print(model.metrics_names)
    loss = model.evaluate(X_test, Y_test, batch_size=64)
    print('\nTesting loss: {}, acc: {}\n'.format('', loss))


# eval_model()

if __name__ == '__main__':
    main()  # %%
