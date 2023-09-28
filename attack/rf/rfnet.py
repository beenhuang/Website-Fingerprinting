#!/usr/bin/env python3

"""
<file>    rf.py
<brief>   RF model using the PyTorch api
"""

import math
import torch
import torch.nn as nn
from torchsummary import summary
       
class RFNet(nn.Module):

    def __init__(self, classes):
        super().__init__()
        self.conv2d_block1 = self.__conv2d_block(1, 32, (1,3))
        self.conv2d_block2 = self.__conv2d_block(32, 64, (2,2))
        self.conv1d_block1 = self.__conv1d_block(32, 128)
        self.conv1d_block2 = self.__conv1d_block(128, 256)
        self.glob_avg_pooling_block = self.__glob_avg_pooling_block(classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.conv2d_block1(x)
        x = self.conv2d_block2(x)
        x = x.view(x.size(0), 32, -1)
        x = self.conv1d_block1(x)
        x = self.conv1d_block2(x)
        x = self.glob_avg_pooling_block(x)
        x = x.view(x.size(0), -1)

        return x        
    
    # 2D convolutional block
    def __conv2d_block(self, channels_in, channel_out, kernel_size):
        return nn.Sequential(nn.Conv2d(channels_in, channel_out, kernel_size=(3, 6), stride=1, padding=(1, 1)),
                             nn.BatchNorm2d(channel_out, eps=1e-05, momentum=0.1, affine=True),
                             nn.ReLU(),
            
                             nn.Conv2d(channel_out, channel_out, kernel_size=(3, 6), stride=1, padding=(1, 1)),
                             nn.BatchNorm2d(channel_out, eps=1e-05, momentum=0.1, affine=True),
                             nn.ReLU(),

                             nn.MaxPool2d(kernel_size=kernel_size),
                             nn.Dropout(0.1))
    
    # 1D convolutional block
    def __conv1d_block(self, channels_in, channel_out):
        return nn.Sequential(nn.Conv1d(channels_in, channel_out, kernel_size=3, stride=1, padding=1),
                             nn.BatchNorm1d(channel_out, eps=1e-05, momentum=0.1, affine=True),
                             nn.ReLU(),

                             nn.Conv1d(channel_out, channel_out, kernel_size=3, stride=1, padding=1),
                             nn.BatchNorm1d(channel_out, eps=1e-05, momentum=0.1, affine=True),
                             nn.ReLU(),

                             nn.MaxPool1d(3),
                             nn.Dropout(0.3))

    # global average pooling block
    def __glob_avg_pooling_block(self, classes):
        return nn.Sequential(nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
                             nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True),
                             nn.ReLU(),

                             nn.Conv1d(512, classes, kernel_size=3, stride=1, padding=1),
                             nn.BatchNorm1d(classes, eps=1e-05, momentum=0.1, affine=True),
                             nn.ReLU(),
                             nn.AdaptiveAvgPool1d(1))  

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

if __name__ == '__main__':
    model = RFNet(100)
    output = model(torch.rand(1, 1, 2, 5000))    
     
    print(summary(model, (1, 2, 5000)))  
    print(output.shape) 
