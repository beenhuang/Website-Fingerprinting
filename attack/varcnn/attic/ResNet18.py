from tensorflow import keras
from tensorflow.keras import layers

from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Dense, Activation, ZeroPadding1D, \
    GlobalAveragePooling1D, Add, Concatenate, Dropout
from keras.layers import BatchNormalization

# stage_name=2,3,4,5;  block_name=a,b,c
def ConvBlock(input_tensor, num_output, stride, dilation_rate, stage_name, block_name):
    x = Conv1D(num_output, 3, strides=stride, padding='causal', dilation_rate=dilation_rate[0], use_bias=False, name='res'+stage_name+block_name+'_branch2a')(input_tensor)
    x = BatchNormalization(epsilon=1e-5, name='bn'+stage_name+block_name+'_branch2a')(x)
    x = Activation('relu', name='res'+stage_name+block_name+'_branch2a_relu')(x)

    x = Conv1D(num_output, 3, padding='causal', dilation_rate=dilation_rate[1], use_bias=False, name='res'+stage_name+block_name+'_branch2b')(x)
    x = BatchNormalization(epsilon=1e-5, name='bn'+stage_name+block_name+'_branch2b')(x)
    
    shortcut = Conv1D(num_output, 1, strides=stride, padding='same', use_bias=False, name='res'+stage_name+block_name+'_branch1')(input_tensor)
    shortcut = BatchNormalization(epsilon=1e-5, name='bn'+stage_name+block_name+'_branch1')(shortcut)

    x = Add(name='res'+stage_name+block_name)([x, shortcut])
    x = Activation('relu', name='res'+stage_name+block_name+'_relu')(x)

    return x

def IdentityBlock(input_tensor, num_output, stride, dilation_rate, stage_name, block_name):
    x = Conv1D(num_output, 3, strides=stride, padding='causal', dilation_rate=dilation_rate[0], use_bias=False, name='res'+stage_name+block_name+'_branch2a')(input_tensor)
    x = BatchNormalization(epsilon=1e-5, name='bn'+stage_name+block_name+'_branch2a')(x)
    x = Activation('relu', name='res'+stage_name+block_name+'_branch2a_relu')(x)

    x = Conv1D(num_output, 3, padding='causal', dilation_rate=dilation_rate[1], use_bias=False, name='res'+stage_name+block_name+'_branch2b')(x)
    x = BatchNormalization(epsilon=1e-5, name='bn'+stage_name+block_name+'_branch2b')(x)

    shortcut = input_tensor

    x = Add(name='res'+stage_name+block_name)([x, shortcut])
    x = Activation('relu', name='res'+stage_name+block_name+'_relu')(x)

    return x

def ResNet18(input_shape, class_num):
    input = keras.Input(shape=input_shape, name='input')

    # conv1
    x = Conv1D(64, 7, strides=2, padding='same', use_bias=False, name='conv1')(input)  
    x = BatchNormalization(epsilon=1e-5, name='bn_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling1D(3, strides=2, padding='same', name='pool1')(x)   

    # conv2_x
    x = ConvBlock(input_tensor=x, num_output=64, stride=1, dilation_rate=(1,2), stage_name='2', block_name='a')
    x = IdentityBlock(input_tensor=x, num_output=64, stride=1, dilation_rate=(4,8), stage_name='2', block_name='b')

    # conv3_x
    x = ConvBlock(input_tensor=x, num_output=128, stride=2, dilation_rate=(1,2), stage_name='3', block_name='a')
    x = IdentityBlock(input_tensor=x, num_output=128, stride=1, dilation_rate=(4,8), stage_name='3', block_name='b')

    # conv4_x
    x = ConvBlock(input_tensor=x, num_output=256, stride=2, dilation_rate=(1,2), stage_name='4', block_name='a')
    x = IdentityBlock(input_tensor=x, num_output=256, stride=1, dilation_rate=(4,8), stage_name='4', block_name='b')

    # conv5_x
    x = ConvBlock(input_tensor=x, num_output=512, stride=2, dilation_rate=(1,2), stage_name='5', block_name='a')
    x = IdentityBlock(input_tensor=x, num_output=512, stride=1, dilation_rate=(4,8), stage_name='5', block_name='b')

    # average pool, 1000-d fc, softmax
    x = GlobalAveragePooling1D(name='pool5')(x)

    model = keras.Model(input, x, name='resnet18')

    return model

if __name__ == '__main__':
    model = ResNet18((5000, 1), 100)
    model.summary()
