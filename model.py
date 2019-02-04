#%%
import csv
import cv2
import numpy as np
import datetime
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Training data path
dataPath = './training_data/02/'
input_width = 200
input_height = 66

#load csv file
def loadDrivingLog(path):
    lines = []
    with open(path + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines

# Print iterations progress
import sys
def print_progress(iteration, total):
    """
    Call in a loop to create terminal progress bar
    
    Parameters
    ----------
        
    iteration : 
                Current iteration (Int)
    total     : 
                Total iterations (Int)
    """
    str_format = "{0:.0f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(100 * iteration / float(total)))
    bar = '█' * filled_length + '-' * (100 - filled_length)

    sys.stdout.write('\r |%s| %s%%' % (bar, percents)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def preprocessImg(img):
    # filter out unnecessary scene and the car front part
    img = img[70:-20, :, :]

    # Resize to fit the Nvidia input image size
    img = cv2.resize(img, (input_width, input_height), cv2.INTER_AREA)

    # Change color space to YUV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    return img

#generator to yeild processed images for training as well as validation data set
def img_generator(data, batchSize = 32):
    while 1:
        data = shuffle(data)
        for i in range(0, len(data), int(batchSize/4)):
            X = []
            y = []
            details = data[i: i+int(batchSize/4)]
            for line in details:
                delimiter = '/'
                if '/' not in line[0]:
                    delimiter = '\\'

                image = cv2.imread(dataPath + 'IMG/'+ line[0].split(delimiter)[-1])
                image = preprocessImg(image)
                steering_angle = float(line[3])
                
                #appending original image
                X.append(image)
                y.append(steering_angle)
                
                #appending flipped image
                X.append(np.fliplr(image))
                y.append(-steering_angle)
                
                # appending left camera image and steering angle with offset
                l_img = cv2.imread(dataPath + 'IMG/'+ line[1].split(delimiter)[-1])
                l_img = preprocessImg(l_img)
                X.append(l_img)
                y.append(steering_angle+0.45)
                
                # appending right camera image and steering angle with offset
                r_img = cv2.imread(dataPath + 'IMG/'+ line[2].split(delimiter)[-1])
                r_img = preprocessImg(r_img)
                X.append(r_img)
                y.append(steering_angle-0.45)
            
            # converting to numpy array
            X = np.array(X)
            y = np.array(y)
            yield shuffle(X, y)

# Load all images directly
def loadImages(lines):
    X = []
    y = []
    for i in range(len(lines)):
        line = lines[i]
        delimiter = '/'
        if '/' not in line[0]:
            delimiter = '\\'
            
        image = cv2.imread(dataPath + 'IMG/'+ line[0].split(delimiter)[-1])
        image = preprocessImg(image)
        steering_angle = float(line[3])

        #appending original image
        X.append(image)
        y.append(steering_angle)

        #appending flipped image
        X.append(np.fliplr(image))
        y.append(-steering_angle)

        # appending left camera image and steering angle with offset
        l_img = cv2.imread(dataPath + 'IMG/'+ line[1].split(delimiter)[-1])
        l_img = preprocessImg(l_img)
        X.append(l_img)
        y.append(steering_angle+0.45)

        # appending right camera image and steering angle with offset
        r_img = cv2.imread(dataPath + 'IMG/'+ line[2].split(delimiter)[-1])
        r_img = preprocessImg(r_img)
        X.append(r_img)
        y.append(steering_angle-0.45)
        print_progress(i+1, len(lines))

    return np.array(X), np.array(y)

lines = loadDrivingLog(dataPath)
training, valid = train_test_split(lines, test_size = 0.2)
# X_train, y_train = loadImages(lines)

from keras.models import Sequential
from keras.layers import Convolution2D, Dropout, MaxPooling2D, Flatten, Activation, Dense, Cropping2D, Lambda
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Model Architecture:
# Based on Nvidia End-to-End Learning for self-driving Cars
# https://devblogs.nvidia.com/deep-learning-self-driving-cars/
model = Sequential()
model.add(Lambda(lambda x: x /255.0 - 0.5, input_shape=(input_height,input_width,3)))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='elu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='elu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='elu'))
model.add(Convolution2D(64, 3, 3, activation='elu'))
model.add(Convolution2D(64, 3, 3, activation='elu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Compiling and running model
model.compile(optimizer='adam', loss='mse')

# Setup early stopping if there's no improvement of loss and store the best result.
callbacks = [
    EarlyStopping(patience=2, monitor='loss', min_delta=0, mode='min'),
    ModelCheckpoint('model_best.h5', monitor='loss', save_best_only=True, verbose=1)
]

# history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, batch_size=32, epochs=50, verbose=1, callbacks=callbacks)
history_object = model.fit_generator(img_generator(training), samples_per_epoch=len(training)*4, nb_epoch = 5, validation_data=img_generator(valid), nb_val_samples=len(valid), verbose=1, callbacks=callbacks)

#saving the model
model.save('model.h5')

#%%
# from keras.models import load_model
# from keras.models import Sequential
# import matplotlib.pyplot as plt
# history_object = load_model('model_final.h5')

# print(history_object.summary())
# print(history_object.history.keys())

# ### plot the training and validation loss for each epoch
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()
