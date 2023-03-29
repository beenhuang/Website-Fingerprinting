#!/usr/bin/env python3

"""
<file>    dfnet.py
<brief>   DF model is based on https://github.com/deep-fingerprinting/df/
"""
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.initializers import glorot_uniform
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
from keras.optimizers import Adamax


# define the ConvNet
class ConvNet:
    @staticmethod
    def build(classes,
              input_shape,
              activation_function=("elu", "relu", "relu", "relu", "relu", "relu"),
              dropout=(0.1, 0.1, 0.1, 0.1, 0.5, 0.7),
              filter_num=(32, 64, 128, 256),
              kernel_size=8,
              conv_stride_size=1,
              pool_stride_size=4,
              pool_size=8,
              fc_layer_size=(512, 512)):

        # confirm that parameter vectors are acceptable lengths
        assert len(filter_num) + len(fc_layer_size) <= len(activation_function)
        assert len(filter_num) + len(fc_layer_size) <= len(dropout)

        # Sequential Keras model template
        model = Sequential()

        # add convolutional layer blocks
        for block_no in range(0, len(filter_num)):
            if block_no == 0:
                model.add(Conv1D(filters=filter_num[block_no],
                                 kernel_size=kernel_size,
                                 input_shape=input_shape,
                                 strides=conv_stride_size,
                                 padding='same',
                                 name='block{}_conv1'.format(block_no)))
            else:
                model.add(Conv1D(filters=filter_num[block_no],
                                 kernel_size=kernel_size,
                                 strides=conv_stride_size,
                                 padding='same',
                                 name='block{}_conv1'.format(block_no)))

            model.add(BatchNormalization())

            model.add(Activation(activation_function[block_no], name='block{}_act1'.format(block_no)))

            model.add(Conv1D(filters=filter_num[block_no],
                             kernel_size=kernel_size,
                             strides=conv_stride_size,
                             padding='same',
                             name='block{}_conv2'.format(block_no)))

            model.add(BatchNormalization())

            model.add(Activation(activation_function[block_no], name='block{}_act2'.format(block_no)))

            model.add(MaxPooling1D(pool_size=pool_size,
                                   strides=pool_stride_size,
                                   padding='same',
                                   name='block{}_pool'.format(block_no)))

            model.add(Dropout(dropout[block_no], name='block{}_dropout'.format(block_no)))

        # flatten output before fc layers
        model.add(Flatten(name='flatten'))

        # add fully-connected layers
        for layer_no in range(0, len(fc_layer_size)):
            model.add(Dense(fc_layer_size[layer_no],
                            kernel_initializer=glorot_uniform(seed=0),
                            name='fc{}'.format(layer_no)))

            model.add(BatchNormalization())
            model.add(Activation(activation_function[len(filter_num)+layer_no],
                                 name='fc{}_act'.format(layer_no)))

            model.add(Dropout(dropout[len(filter_num)+layer_no],
                              name='fc{}_drop'.format(layer_no)))

        # add final classification layer
        model.add(Dense(classes, kernel_initializer=glorot_uniform(seed=0), name='fc_final'))
        model.add(Activation('softmax', name="softmax"))

        # compile model with Adamax optimizer
        optimizer = Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        return model


if __name__ == "__main__":
    model = ConvNet.build(100, (5000,1))
    model.summary()
