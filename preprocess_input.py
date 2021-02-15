import cv2
import numpy as np
import matplotlib.pyplot as plt

# FINAL_IMG_SHAPE = (66, 200, 3)
FINAL_IMG_SHAPE = (160, 320, 3)
TEST_IMG_SHAPE = (160, 320, 3)


def resize(x):
    # height = x.shape[0]
    # width = x.shape[1]
    width, height = x.size
    factor = float(FINAL_IMG_SHAPE[1]) / float(width)

    resized_size = (int(width * factor), int(height * factor))
    image_array = np.asarray(x)
    x = cv2.resize(image_array, resized_size)
    crop_height = resized_size[1] - FINAL_IMG_SHAPE[0]

    return x[crop_height:, :, :]


def rgb_to_yuv(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2YUV)


def main(img):
    """ Preprocesses input data
        img is an image of shape (height, width, depth) """
    img = resize(img)
    # img = rgb_to_yuv(img)

    return img


def visualize(model):
    # plot history
    print(model.history.keys())
    print('Loss')
    print(model.history['loss'])
    print('Validation Loss')
    print(model.history['val_loss'])
    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
