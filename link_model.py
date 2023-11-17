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

'''Hyper params'''
num_samples=2399
batch_size=64#64 yeilds best results
epochs=15

train_it = tf.keras.utils.text_dataset_from_directory(
    bin_path,
    labels='inferred',
    label_mode='categorical',
    class_names=None,
    batch_size=batch_size,
    max_length=None,
    shuffle=True,
    seed=2,
    validation_split=0.5,
    subset='training',
    follow_links=False
)
val_it = tf.keras.utils.text_dataset_from_directory(
    bin_path,
    labels='inferred',
    label_mode='categorical',
    class_names=None,
    batch_size=batch_size,
    max_length=None,
    shuffle=True,
    seed=2,
    validation_split=0.5,
    subset='validation',
    follow_links=False
)
test_it = tf.keras.utils.text_dataset_from_directory(
    bin_path,
    labels='inferred',
    label_mode='categorical',
    class_names=None,
    batch_size=batch_size,
    max_length=None,
    shuffle=True,
    seed=2,
    validation_split=None,
    subset=None,
    follow_links=False
)
for i, label in enumerate(train_it.class_names):
      print("Label", i, "corresponds to", label)

def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label


max_features = 10000#2399
sequence_length = 250
embedding_dim = 128

vectorize_layer = TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)
# Make a text-only dataset (without labels), then call adapt
train_text = train_it.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

val_text = train_it.map(lambda x, y: x)
vectorize_layer.adapt(val_text)

test_text = test_it.map(lambda x, y: x)
vectorize_layer.adapt(test_text)


def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label

train_ds = train_it.map(vectorize_text)
val_ds = val_it.map(vectorize_text)
test_ds = test_it.map(vectorize_text)


# model = tf.keras.Sequential([
#   Embedding(max_features + 1, embedding_dim),
#   Dropout(0.1),
#   GlobalAveragePooling1D(),
#   BatchNormalization(),#essential
#   Dropout(0.1),
#   Dense(128,activation='relu'),
#   Dense(n_outputs,activation='softmax')])


model = tf.keras.Sequential([
  Embedding(max_features + 1, embedding_dim),
  Dropout(0.1),
  GlobalAveragePooling1D(),
#   Conv1D(32, 8, activation="relu"),
#   MaxPool1D(2),
  BatchNormalization(),#essential
  Dropout(0.1),
  Dense(128,activation='relu'),
  Dense(128,activation='relu'),
  Dense(n_outputs,activation='softmax')])

model.compile(
    #loss=SparseCategoricalCrossentropy(from_logits=True),
    loss='categorical_crossentropy', 
    optimizer=tf.keras.optimizers.experimental.RMSprop(learning_rate=1e-9),#'rmsprop',#tf.keras.optimizers.Adam(learning_rate=1e-9), 
    #steps_per_execution=num_samples//batch_size,
    metrics=['accuracy'])

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)

loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

