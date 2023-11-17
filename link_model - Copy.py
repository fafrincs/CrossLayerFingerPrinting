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
from keras.layers import Conv2D, Conv1D
from keras.layers import MaxPooling2D, MaxPool1D
from tensorflow.keras.layers import Flatten, InputLayer
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.layers import Conv2D, BatchNormalization, TimeDistributed, LSTM, Dropout, Input, TextVectorization, Embedding, GlobalAveragePooling1D
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import BinaryCrossentropy,SparseCategoricalCrossentropy


import cv2


''' n_outputs for 151020 (num of devices) = 9
    n_outputs for 151021 (num of devices) = ??
    n_outputs for 151022 (num of devices) = ??
'''
n_outputs =  9



'''Preprocess the images'''
#1. Input bin graphs for the DCNN
bit_path = r"C:\Users\saisu\OneDrive\Documents\Research\CNS_CPS_2023\poster\code\images\bitstream"
bin_path= r"C:\Users\saisu\OneDrive\Documents\Research\CNS_CPS_2023\poster\code\images\binary"


# model = Sequential()
# model.add(Input(shape=(None,None,1)))
# model.add(Conv1D(64, 3, activation='relu'))
# #model.add(MaxPool1D(2, strides=1,padding="same"))
# #model.add(BatchNormalization())
# model.add(Conv1D(128, 4, activation='relu',padding="same"))
# #model.add(MaxPool1D(2, strides=2,padding="same"))
# #model.add(BatchNormalization())
# model.add(Conv1D(256, 4, activation='relu',padding="same"))
# #model.add(MaxPool1D(2, strides=2,padding="same"))
# #model.add(BatchNormalization())
# model.add(GlobalAveragePooling1D())
# model.add(Dropout(0.2))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(n_outputs, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-9), metrics=['accuracy'])
# print(model.summary())


train_it = tf.keras.utils.text_dataset_from_directory(
    bin_path,
    labels='inferred',
    label_mode='categorical',
    class_names=None,
    batch_size=32,
    max_length=None,
    shuffle=True,
    seed=2,
    validation_split=0.3,
    subset='training',
    follow_links=False
)
for i, label in enumerate(train_it.class_names):
      print("Label", i, "corresponds to", label)

def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label

#train_y = train_y.reshape(1,len(train_y))
#model.fit(train_x, train_y, epochs=3)
num_samples=2399
batch_size=32
STEP_SIZE_TRAIN=num_samples//batch_size

max_features = 10000
sequence_length = 250
embedding_dim = 128

vectorize_layer = TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)
# Make a text-only dataset (without labels), then call adapt
train_text = train_it.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label

train_ds = train_it.map(vectorize_text)

model = tf.keras.Sequential([
  Embedding(max_features + 1, embedding_dim),
  Dropout(0.2),
  GlobalAveragePooling1D(),
  Dropout(0.2),
  Dense(n_outputs)])

model.compile(
    #loss=SparseCategoricalCrossentropy(from_logits=True),
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy'])

history = model.fit(
    train_ds,
    validation_data=train_ds,
    epochs=5)


# epochs = 10
# history = model.fit(
#     train_it,
#     validation_data=train_it,
#     epochs=epochs)


#model.fit_generator(generator=train_it,steps_per_epoch=STEP_SIZE_TRAIN,validation_data=train_it,validation_steps=STEP_SIZE_TRAIN,epochs=10)









