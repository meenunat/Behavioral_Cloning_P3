import numpy as np
import pandas as pd
import math
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf

# Initial Setup for Keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.activations import relu
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

# Load all driving log data (3 images, angle, throttle, break, speed)
driving_data = pd.read_csv('./input/driving_log.csv', sep=',', header = None)

# Make lists of only the center image names (to pull image data later) and steering angles
car_images = []
steering_angles = []
# this is a parameter to tune
correction = 0.2 

for x in range(0, len(driving_data)):
    # read in image path for center, left and right images
    center_image_path = driving_data.get_value(x, 0)
    left_image_path = driving_data.get_value(x, 1)
    right_image_path = driving_data.get_value(x, 2)
    # create adjusted steering measurements for the side camera images from centre camera image
    steering_center = float(driving_data.get_value(x, 3))
    steering_left = steering_center + correction
    steering_right = steering_center - correction

    # read in images from center, left and right cameras
    img_center = mpimg.imread(center_image_path)
    img_left = np.asarray(mpimg.imread(left_image_path))
    img_right = np.asarray(mpimg.imread(right_image_path))

    # add images and angles to data set
    car_images.extend([img_center, img_left, img_right])
    steering_angles.extend([steering_center, steering_left, steering_right])
    #car_images.append(img_center)
    #steering_angles.append(steering_center)
    
print('All images loaded.')

#Create Dataset
X_data = np.array(car_images)
y_data = np.array(steering_angles)

# Shuffling first, then splitting into training and validation sets
X_train, y_train = shuffle(X_data, y_data)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=23)

# Model
print('Neural network initializing...')
batch_size = 128
epochs = 10

model = Sequential()
# Preprocess incoming data - Normalize the image 
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(160, 320, 3)))
#crop the image along spatial dimensions
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
#LeNet Architecture with additional fully connected layer and a dropout layer
#First Convolution Layer
model.add(Convolution2D(6, 5, 5, activation ='relu'))
#Pooling Layer
model.add(MaxPooling2D((2, 2)))
#second Convolution Layer
model.add(Convolution2D(16, 5, 5, activation ='relu'))
#Pooling Layer
model.add(MaxPooling2D((2, 2)))
#Dropout Layer
model.add(Dropout(0.5))
model.add(Flatten())
#Fully connected layers with 120; 84; 10; 1 neurons
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(10))
model.add(Dense(1))
#Optimizer - Adam and the loss is measured to reduce the mean square error
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs, verbose=2, validation_data=(X_val, y_val))

# Save model architecture and weights
model.save('model.h5')

# Show summary of model
model.summary()