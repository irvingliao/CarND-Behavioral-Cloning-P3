#%%
import csv
import cv2
import numpy as np
import datetime
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

dataPath = './training_data/03/'

#load csv file
def loadDrivingLog(path):
    lines = []
    with open(path + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines

#generator to yeild processed images for training as well as validation data set
def img_generator(data, batchSize = 32):
    while 1:
        data = shuffle(data)
        for i in range(0, len(data), int(batchSize/4)):
            X_batch = []
            y_batch = []
            details = data[i: i+int(batchSize/4)]
            for line in details:
                image = cv2.imread(dataPath + 'IMG/'+ line[0].split('/')[-1])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                steering_angle = float(line[3])
                #appending original image
                X_batch.append(image)
                y_batch.append(steering_angle)
                #appending flipped image
                X_batch.append(np.fliplr(image))
                y_batch.append(-steering_angle)
                # appending left camera image and steering angle with offset
                l_img = cv2.imread(dataPath + 'IMG/'+ line[1].split('/')[-1])
                l_img = cv2.cvtColor(l_img, cv2.COLOR_BGR2RGB)
                X_batch.append(l_img)
                y_batch.append(steering_angle+0.4)
                # appending right camera image and steering angle with offset
                r_img = cv2.imread(dataPath + 'IMG/'+ line[2].split('/')[-1])
                r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
                X_batch.append(r_img)
                y_batch.append(steering_angle-0.3)
            # converting to numpy array
            X_batch = np.array(X_batch)
            y_batch = np.array(y_batch)
            yield shuffle(X_batch, y_batch)

def loadImages(lines):
    X = []
    y = []
    for line in lines:
        image = cv2.imread(dataPath + 'IMG/'+ line[0].split('/')[-1])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        steering_angle = float(line[3])
        #appending original image
        X.append(image)
        y.append(steering_angle)
        #appending flipped image
        X.append(np.fliplr(image))
        y.append(-steering_angle)
        # appending left camera image and steering angle with offset
        l_img = cv2.imread(dataPath + 'IMG/'+ line[1].split('/')[-1])
        l_img = cv2.cvtColor(l_img, cv2.COLOR_BGR2RGB)
        X.append(l_img)
        y.append(steering_angle+0.4)
        # appending right camera image and steering angle with offset
        r_img = cv2.imread(dataPath + 'IMG/'+ line[2].split('/')[-1])
        r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
        X.append(r_img)
        y.append(steering_angle-0.3)

    return np.array(X), np.array(y)

lines = loadDrivingLog(dataPath)
# training, valid = train_test_split(lines, test_size = 0.2)
X_train, y_train = loadImages(lines)

from keras.models import Sequential
from keras.layers import Convolution2D, Dropout, MaxPooling2D, Flatten, Activation, Dense, Cropping2D, Lambda

#creating model to be trained
model = Sequential()
model.add(Lambda(lambda x: x /255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,20),(0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
# model.add(Dropout(0.75))
# model.add(MaxPooling2D())
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
# model.add(Dropout(0.5))
# model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#compiling and running the model
model.compile(optimizer='adam', loss='mse')
# history_object = model.fit_generator(img_generator(training), samples_per_epoch=len(training)*4, nb_epoch = 2, validation_data=img_generator(valid), nb_val_samples=len(valid), verbose=1)
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, batch_size=32, epochs=10, verbose=1)

print(history_object.history.keys())

#saving the model
model.save('model.h5')
