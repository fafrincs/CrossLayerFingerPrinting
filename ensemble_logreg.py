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


#model.fit_generator(generator=train_it,steps_per_epoch=STEP_SIZE_TRAIN,validation_data=train_it,validation_steps=STEP_SIZE_TRAIN,epochs=10)

#https://machinelearningmastery.com/implementing-stacking-scratch-python/







