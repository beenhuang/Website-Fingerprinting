#!/usr/bin/env python3

"""
<file>    awf_cnn.py
<brief>   AWF_CNN model
"""

from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Flatten, Dense, Dropout
import tensorflow as tf

def awf_cnn_model(input_shape, classes):
    model = Sequential()

    model.add(Dropout(input_shape=input_shape, rate=0.1))
    
    model.add(Conv1D(filters=32, kernel_size=5, padding='valid', activation="relu", strides=1))
    model.add(MaxPooling1D(pool_size=4, padding='valid'))
    
    model.add(Conv1D(filters=32, kernel_size=5, padding='valid', activation="relu", strides=1))
    model.add(MaxPooling1D(pool_size=4, padding='valid'))

    model.add(Flatten())
    model.add(Dense(units=classes, activation='softmax'))

    return model

if __name__ == "__main__":
    model = awf_cnn_model((2000, 1), 100)
    out = model(tf.ones([1, 2000, 1]))
    model.summary()
    #print(out.shape)


