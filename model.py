import tensorflow as tf
import numpy as np
import cv2
import csv
import json
import os.path
import pickle
import h5py
import os
import argparse
import json
import pandas as pandas
import math

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Lambda, ELU, Dropout, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.utils import np_utils
#from keras.utils import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

#fix from Thomas Antony - MUch thanks!!!
import tensorflow as tf
tf.python.control_flow_ops = tf

#all the things ------------------------------------------
img_cols = 200
img_rows = 66
batch_size = 64
nb_epochs = 200

# read the image from the given path and convert BGR to RGB
def readImage(image_path):
    image_cv2 = cv2.imread(image_path)
    image_color = cv2.cvtColor(image_cv2,cv2.COLOR_BGR2RGB)
    return image_color

#shuffling is done in generator and this is no longer used
def shuffle(images, labelx):
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    images = images[idx]
    labels = labels[idx]
    return (x, y)

#normalize the images - note that this is done through the lambda in the model and not being used
def normalize(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [-0.5, 0.5]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    a = 0.5
    b = -0.5
    greyscale_min = 0
    greyscale_max = 255
    return a + ( ( (image_data - greyscale_min)*(b - a) )/( greyscale_max - greyscale_min ) )

#convert to YUV now does as a convolution layer at beginning of model
def convertToYUV(image):
    converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    return converted_image

def flipImage(img):
    flipped_image = img.copy()
    flipped_image = cv2.flip(img, 1)
    # num_rows, num_cols = img.shape[:2]
    # rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 30, 1)
    # img_rotation = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
    return flipped_image

#vivek yadav's brightness changer for augmentation
def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

#vivek yadav's translation for augmentation
def trans_image(image,steer,trans_range):
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 40*np.random.uniform()-40/2
    #tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(img_cols,img_rows))

    return image_tr,steer_ang

def cropImage(image):
     image_clipped = image[math.floor(image.shape[0]/5):image.shape[0]-25, 0:image.shape[1]]
     return image_clipped

def resizeImage(image):
    image_resized =  cv2.resize(image, (img_cols, img_rows),interpolation=cv2.INTER_AREA)
    return image_resized

def processImage(image, angle):
    #image = readImage(image)
    image = cropImage(image)
    image = resizeImage(image)
    image = augment_brightness_camera_images(image)
    image, angle = trans_image(image, angle, 100)
    #image = convertToYUV(image)
    random_int = np.random.randint(100)
    if random_int > 50: # flip the image half the time
        image = flipImage(image)
        angle = -angle
    return image, angle

#generator
def generate(images, labels, batch_size):
    batch_images = np.zeros((batch_size, img_rows, img_cols, 3))
    batch_labels = np.zeros(batch_size)
    total = 0
    while 1:
        for i in range(batch_size):
            if total >= len(images):
                total = 0
            random_int = np.random.randint(len(images))
            image = images[random_int]
            label = labels[random_int]
            batch_images[i], batch_labels[i]= processImage(image, label)
            total = total + 1
        yield batch_images, batch_labels

# Implementation ------------------------------------------------------
if(os.path.exists('train.p')):
    training_data = pickle.load(open('train.p','rb'))
    test_data = pickle.load(open('test.p','rb'))
    X_train, y_train = training_data['X_train'], training_data['y_train']
    X_test, y_test = test_data['X_test'], test_data['y_test']
else:
    #load the data
    data = pandas.read_csv('driving_log.csv', header = None)
    data.columns = ["center_images","left_images","right_images","steering","brake","throttle","speed"]
    angles = data['steering']
    center_images = data['center_images']
    left_images = data['left_images']
    right_images = data['right_images']

    ## Prepare the training and test data -----------
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    #center images
    for i,image in enumerate(center_images):
        if not i == 0: #first record was None probably because of the header row in CSV
            image_center = readImage(center_images[i])
            image_left = readImage(left_images[i])
            image_right = readImage(right_images[i])
            angle_center = float(angles[i])
            angle_left = angle_center + .25 #((0.1*angle_center)+.1)
            angle_right = angle_center - .25 #((0.1*angle_center)+.1)
            if(i%19==0):
                X_test.append(image_center)
                y_test.append(angle_center)
                X_test.append(image_left)
                y_test.append(angle_left)
                X_test.append(image_right)
                y_test.append(angle_right)
            if angle_center == 0:
                random_int = np.random.randint(100)
                if random_int > 50:
                    X_train.append(image_center)
                    y_train.append(angle_center)
            else:
                X_train.append(image_center)
                y_train.append(angle_center)
            X_train.append(image_left)
            y_train.append(angle_left)
            X_train.append(image_right)
            y_train.append(angle_right)

    # #left images
    # for j,image2 in enumerate(left_images):
    #     if not (j == 0 ): #first record was None probably because of the header row in CSV
    #         image_read = readImage(image2)
    #         angle = float(angles[j]) + .25
    #         if(j%9==0):
    #             X_test.append(image_read)
    #             y_test.append(angle)
    #         X_train.append(image_read)
    #         y_train.append(angle)
    #
    # #right images
    # for k,image3 in enumerate(right_images):
    #     if not (k == 0): #first record was None probably because of the header row in CSV
    #         image_read = readImage(image3)
    #         angle = float(angles[k]) - .25
    #         if(k%9==0):
    #                 X_test.append(image_read)
    #                 y_test.append(angle)
    #         X_train.append(image_read)
    #         y_train.append(angle)


    #convert list to np.array
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    #reshape the arrays
    X_train = X_train.reshape(len(X_train),160,320,3)
    X_test = X_test.reshape(len(X_test),160,320,3)
    train_labels = np.zeros(len(y_train))
    test_labels = np.zeros(len(y_test))
    for i,label in enumerate(y_train):
        train_labels[i] = label

    for i,label in enumerate(y_test):
        test_labels[i] = label

    y_train = train_labels
    y_test = test_labels

## -------------------------------
#model - Based on Nvidia model plus a conv layer to change the color space
img_shape = (img_rows,img_cols,3)
model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1., #normalization
            input_shape=img_shape))
#change color space
model.add(Convolution2D(3,1,1, subsample=(1,1), border_mode="valid", init="he_normal"))
# Nvidia model
model.add(Convolution2D(24,5,5, subsample=(2,2), border_mode="valid", init="he_normal"))
model.add(ELU())
model.add(Convolution2D(36,5,5, subsample=(2,2), border_mode="valid", init="he_normal"))
model.add(ELU())
model.add(Convolution2D(48,5,5, subsample=(2,2), border_mode="valid", init="he_normal"))
model.add(ELU())
model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode="valid", init="he_normal"))
model.add(ELU())
model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode="valid", init="he_normal"))
model.add(Flatten())
model.add(ELU())
model.add(Dense(100, init="he_normal"))
model.add(ELU())
model.add(Dense(50, init="he_normal"))
model.add(ELU())
model.add(Dense(10, init="he_normal"))
model.add(ELU())
model.add(Dense(1))

model.summary()
adam = Adam(lr=1e-4)
model.compile(loss='mse',
              optimizer=adam,
              metrics=[])

'''
saves the model weights after each epoch if the validation loss decreased
'''
checkpointer = ModelCheckpoint(filepath="./model{epoch:02d}.h5", verbose=1, save_best_only=False)
#history = model.fit(X_train, y_train, nb_epoch=nb_epochs,validation_split=0.05, shuffle=True, callbacks=[checkpointer])
history = model.fit_generator(
    generate(X_train, y_train, batch_size),
    samples_per_epoch=batch_size * 300,
    nb_epoch=nb_epochs,
    verbose=1,
    validation_data=generate(X_test, y_test, 100),
    nb_val_samples=len(X_test),
    callbacks=[checkpointer]
  )

#Save final weights and models
print("Saving model weights and configuration file.")

model.save_weights("./model.h5", True)
with open('./model.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)
