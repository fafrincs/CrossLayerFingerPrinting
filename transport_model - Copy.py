'''Packages for data processing'''
import numpy as np
import pandas as pd



'''Packages for file processing'''
import os
import glob
import sys


'''Packages for ML'''
import tensorflow as tf
import keras as kr
'''Packages for DCNN'''
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from tensorflow.keras.layers import Conv2D, BatchNormalization, TimeDistributed, LSTM, Dropout, Input
from keras.preprocessing.image import ImageDataGenerator


import cv2

'''Paths to the input files'''
path1= r"C:\Users\saisu\OneDrive\Documents\Research\CNS_CPS_2023\poster\data\datasets\4sics\4SICS-GeekLounge-151020_transport.csv"
path2= r"C:\Users\saisu\OneDrive\Documents\Research\CNS_CPS_2023\poster\data\datasets\4sics\4SICS-GeekLounge-151021_transport.csv"
path3= r"C:\Users\saisu\OneDrive\Documents\Research\CNS_CPS_2023\poster\data\datasets\4sics\4SICS-GeekLounge-151022_transport.csv"


''' n_outputs for 151020 (num of devices) = 9
    n_outputs for 151021 (num of devices) = ??
    n_outputs for 151022 (num of devices) = ??
'''
n_outputs =  9



'''Preprocess the images'''
#1. Input IAT graphs for the DCNN
iat_path = r"C:\Users\saisu\OneDrive\Documents\Research\CNS_CPS_2023\poster\code\images\iat"
x_train=[]
for folder in os.listdir(iat_path):
    sub_path=iat_path+"/"+folder
    for img in os.listdir(sub_path):
        image_path=sub_path+"/"+img
        img_arr=cv2.imread(image_path)
        img_arr=cv2.resize(img_arr,(256,256))
        x_train.append(img_arr)


#Normalize the images
train_x=np.array(x_train)
train_x=train_x/255.0
print(len(train_x))
print(train_x.shape)
#train_x = train_x.reshape(len(train_x),256,256,3)
train_x = train_x.reshape(len(train_x),256, 256,3,1)
print(train_x.shape)


model = Sequential()
#See: https://datascience.stackexchange.com/questions/97276/cnnlstm-valueerror-input-0-of-layer-sequential-10-is-incompatible-with-the-lay
#https://gist.github.com/HTLife/ca0a7d48bd9a3192cf8d3c8b1347e8dd
model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu'), input_shape=(256, 256,3,1)))
model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(1, 1),padding="same")))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Conv2D(128, (4,4), activation='relu',padding="same")))
model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2),padding="same")))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Conv2D(256, (4,4), activation='relu',padding="same")))
model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2),padding="same")))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Flatten()))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=False, dropout=0.2))
#model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-9), metrics=['accuracy'])
print(model.summary())

datagen = ImageDataGenerator(rescale=1./255)
train_it = datagen.flow_from_directory(iat_path, class_mode='categorical', batch_size=32,target_size=(256, 256),shuffle=True)
train_y = train_it.classes

#train_y = train_y.reshape(1,len(train_y))
#model.fit(train_x, train_y, epochs=3)

STEP_SIZE_TRAIN=train_it.n//train_it.batch_size
model.fit_generator(generator=train_it,steps_per_epoch=STEP_SIZE_TRAIN,validation_data=train_it,validation_steps=STEP_SIZE_TRAIN,epochs=10)





