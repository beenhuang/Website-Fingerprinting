#!/usr/bin/env python3

"""
<file>    dfnet.py
<brief>   deep fingerprinting model
"""

import torch
import torch.nn as nn
from torchsummaryX import summary
       
class DFNet(nn.Module):
    #
    def __init__(self, classes):
        super(DFNet, self).__init__()

        self.block1 = self.__make_block(1, 32, nn.ELU())
        self.block2 = self.__make_block(32, 64, nn.ReLU())
        self.block3 = self.__make_block(64, 128, nn.ReLU())
        self.block4 = self.__make_block(128, 256, nn.ReLU())
        self.fc_block = self.__make_fc()
        self.prediction = self.__make_prediction(classes)

    # 
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_block(x)
        x = self.prediction(x)

        return x        
    
    # block
    def __make_block(self, channels_in, channels, activation):
        return nn.Sequential(
            nn.Conv1d(in_channels=channels_in, out_channels=channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(num_features=channels),
            activation,

            nn.Conv1d(channels, channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(channels),
            activation,

            nn.MaxPool1d(kernel_size=7, stride=4, padding=3),
            nn.Dropout(p=0.1)
        )

    # fully-connected 
    def __make_fc(self):
        return nn.Sequential(
            nn.Linear(in_features=256*20, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.7),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )  

    # prediction
    def __make_prediction(self, classes):
        return nn.Sequential(
            nn.Linear(512, classes),
            # when using CrossEntropyLoss, already computed internally
            #nn.Softmax(dim=1)
        )

if __name__ == '__main__':
    df = DFNet(100)
    #output = df(torch.rand(32, 1, 5000))
    #print(output.size())
    summary(df, torch.zeros(32, 1, 5000))        
