import csv
import cv2
import numpy as np
from zipfile import ZipFile
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D as Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D, Activation, ELU
from keras.optimizers import Adam

# with ZipFile('data.zip', 'r') as zipObj:
#    # Extract all the contents of zip file in current directory
#    zipObj.extractall()

# Batch size
BATCH_SIZE = 64

lines = []
with open('./data/driving_log.csv') as csvfile:
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
    image_center = cv2.imread("./data/IMG/" + line[0].split('/')[-1])  # left image
    image_left = cv2.imread("./data/IMG/" + line[1].split('/')[-1])  # center image
    image_right = cv2.imread("./data/IMG/" + line[2].split('/')[-1])  # right image
    steering_center = float(line[3])  # steer
    correction = 0.2  # this is a parameter to tune
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    images.extend((image_center, image_left, image_right))
    measurements.extend((steering_center, steering_left, steering_right))

X_train = np.array(images)
y_train = np.array(measurements)

# model = Sequential()
# model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
# model.add(Cropping2D(cropping=((70, 25), (0, 0))))  # (top,bottom),(left,right)
# model.add(Convolution2D(6, (5, 5), activation='relu'))
# model.add(MaxPooling2D(2, 2))
# model.add(Convolution2D(6, (5, 5), activation='relu'))
# model.add(MaxPooling2D(2, 2))
# model.add(Flatten())
# model.add(Dense(128))
# model.add(Dense(84))
# model.add(Dense(1))
#
# model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10)

""" Defines the network architecture, following Nvidia's example on:
       http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf """

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((50, 20), (0, 0))))  # (top,bottom),(left,right)
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(48, (5, 5), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.summary()

# Compile it
model.compile(loss='mse', optimizer=Adam(lr=0.001))
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10)

model.save('modellocal1.h5')
