#!/usr/bin/env python3

"""
<file>    dfnet.py
<brief>   DF model is based on https://github.com/deep-fingerprinting/df/
"""

from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.layers import ELU
from keras.initializers import glorot_uniform

class DFNet:
    @staticmethod
    def build(input_shape, classes):
        # parameters from Block_1 to Block_4
        filter_num = ['None',32,64,128,256]
        kernel_size = ['None',8,8,8,8]
        conv_stride_size = ['None',1,1,1,1]
        pool_stride_size = ['None',4,4,4,4]
        pool_size = ['None',8,8,8,8]

        model = Sequential()

        # Block_1
        model.add(Conv1D(filters=filter_num[1], kernel_size=kernel_size[1], input_shape=input_shape,
                         strides=conv_stride_size[1], padding='same',
                         name='block1_conv1'))
        model.add(BatchNormalization(axis=-1))
        model.add(ELU(alpha=1.0, name='block1_adv_act1'))
        model.add(Dropout(0.1, name='block1_dropout'))

        # Block_2
        model.add(Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                         strides=conv_stride_size[2], padding='same',
                         name='block2_conv1'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='block2_act1'))
        model.add(Dropout(0.1, name='block2_dropout'))


        # Dense
        model.add(Flatten(name='flatten'))
        model.add(Dense(512, kernel_initializer=glorot_uniform(seed=0), name='fc1'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='fc1_act'))
        model.add(Dropout(0.7, name='fc1_dropout'))
        model.add(Dense(512, kernel_initializer=glorot_uniform(seed=0), name='fc2'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='fc2_act'))
        model.add(Dropout(0.5, name='fc2_dropout'))

        # Prediction
        model.add(Dense(classes, kernel_initializer=glorot_uniform(seed=0), name='fc3'))
        model.add(Activation('softmax', name="softmax"))
        
        return model

if __name__ == "__main__":
    model = DFNet.build((5000,1), 100)
    model.summary()

