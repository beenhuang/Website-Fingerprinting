#!/usr/bin/env python3

"""
<file>    dfnet.py
<brief>   DF model with PyTorch
"""

import torch
import torch.nn as nn
from torchsummary import summary
       
class DFNet(nn.Module):

    def __init__(self, classes):
        super(DFNet, self).__init__()
        self.block1 = self.__conv_block(1, 32, nn.ELU())
        self.block2 = self.__conv_block(32, 64, nn.ReLU())
        self.block3 = self.__conv_block(64, 128, nn.ReLU())
        self.block4 = self.__conv_block(128, 256, nn.ReLU())
        self.fc_block = self.__fc_block()
        self.pred_block = self.__pred_block(classes)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc_block(out)
        out = self.pred_block(out)

        return out        
    
    # convolutional block
    def __conv_block(self, channels_in, channels, activation):
        return nn.Sequential(nn.Conv1d(channels_in, channels, kernel_size=8, padding="same"),
                             nn.BatchNorm1d(channels),
                             activation,
            
                             nn.Conv1d(channels, channels, kernel_size=8, padding="same"),
                             nn.BatchNorm1d(channels),
                             activation,

                             nn.MaxPool1d(kernel_size=8, stride=4, padding=3),
                             nn.Dropout(p=0.1))

    # fully-connected block
    def __fc_block(self):
        return nn.Sequential(nn.Linear(256*20, 512),
                             nn.BatchNorm1d(512),
                             nn.ReLU(),
                             nn.Dropout(0.7),

                             nn.Linear(512, 512),
                             nn.BatchNorm1d(512),
                             nn.ReLU(),
                             nn.Dropout(0.5))  

    # prediction block
    def __pred_block(self, classes):
        return nn.Sequential(nn.Linear(512, classes),
                            #nn.Softmax(dim=1) # when using CrossEntropyLoss, already computed internally
                            )

if __name__ == '__main__':
    model = DFNet(100)
    output = model(torch.rand(2, 1, 5000))    
     
    print(summary(model, (1, 5000)))  
    print(output.shape) 
