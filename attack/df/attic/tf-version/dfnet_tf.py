# DF model used for non-defended dataset
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.layers import ELU
from keras.initializers import glorot_uniform
import tensorflow as tf

class DFNet:
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()

        # convolutional_block_1
        model.add(Conv1D(filters=32, kernel_size=8, input_shape=input_shape, strides=1, padding='same', name='block1_conv1'))
        model.add(BatchNormalization(axis=-1))
        model.add(ELU(alpha=1.0, name='block1_adv_act1'))
        model.add(Conv1D(filters=32, kernel_size=8, strides=1, padding='same', name='block1_conv2'))
        model.add(BatchNormalization(axis=-1))
        model.add(ELU(alpha=1.0, name='block1_adv_act2'))
        model.add(MaxPooling1D(pool_size=8, strides=4, padding='same', name='block1_pool'))
        model.add(Dropout(0.1, name='block1_dropout'))
        
        # convolutional_block_2
        model.add(Conv1D(filters=64, kernel_size=8, strides=1, padding='same', name='block2_conv1'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='block2_act1'))
        model.add(Conv1D(filters=64, kernel_size=8, strides=1, padding='same', name='block2_conv2'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='block2_act2'))
        model.add(MaxPooling1D(pool_size=8, strides=4, padding='same', name='block2_pool'))
        model.add(Dropout(0.1, name='block2_dropout'))
        

        # convolutional_block_3
        model.add(Conv1D(filters=128, kernel_size=8,strides=1, padding='same', name='block3_conv1'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='block3_act1'))
        model.add(Conv1D(filters=128, kernel_size=8, strides=1, padding='same', name='block3_conv2'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='block3_act2'))
        model.add(MaxPooling1D(pool_size=8, strides=4, padding='same', name='block3_pool'))
        model.add(Dropout(0.1, name='block3_dropout'))

        # convolutional_block_4
        model.add(Conv1D(filters=256, kernel_size=8, strides=1, padding='same', name='block4_conv1'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='block4_act1'))
        model.add(Conv1D(filters=256, kernel_size=8, strides=1, padding='same', name='block4_conv2'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='block4_act2'))
        model.add(MaxPooling1D(pool_size=8, strides=4, padding='same', name='block4_pool'))
        model.add(Dropout(0.1, name='block4_dropout')) 

        # Fully_Connected_Block
        model.add(Flatten(name='flatten'))
        model.add(Dense(512, kernel_initializer=glorot_uniform(seed=0), name='fc1'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='fc1_act'))
        model.add(Dropout(0.7, name='fc1_dropout'))
        model.add(Dense(512, kernel_initializer=glorot_uniform(seed=0), name='fc2'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='fc2_act'))
        model.add(Dropout(0.5, name='fc2_dropout'))
       
        # Prediction_Block
        model.add(Dense(classes, kernel_initializer=glorot_uniform(seed=0), name='fc3'))
        model.add(Activation('softmax', name="softmax"))

        return model
if __name__ == "__main__":
    model = DFNet.build((5000,1), 100)
    out = model(tf.ones([2, 5000, 1]))
    
    print(model.summary())
    print(out.shape)
