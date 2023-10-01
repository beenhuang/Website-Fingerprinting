#!/usr/bin/env python3

"""
<file>    dfnet.py
<brief>   DF model with PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dialtion):
        super(ConvBlock, self).__init__() 
        self.conv_branch_2a=nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1, dilation=dialtion[0], bias=False)
        self.batchnorm_branch_2a=nn.BatchNorm1d(out_channels)
        self.conv_branch_2b=nn.Conv1d(out_channels, out_channels, 3, padding=2, dilation=dialtion[1], bias=False)
        self.batchnorm_branch_2b=nn.BatchNorm1d(out_channels)
        self.conv_branch_1=nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False)
        self.batchnorm_branch_1=nn.BatchNorm1d(out_channels)

    
    def forward(self, x):
        identity = x

        x = self.conv_branch_2a(x)
        x = self.batchnorm_branch_2a(x)
        x = F.relu(x)
        x = self.conv_branch_2b(x)
        x = self.batchnorm_branch_2b(x)

        identity = self.conv_branch_1(identity)
        identity = self.batchnorm_branch_1(identity)

        x += identity
        x = F.relu(x)

        return x

class IdentityBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dialtion):
        super(IdentityBlock, self).__init__() 
        self.conv_branch_2a=nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=4, dilation=dialtion[0], bias=False)
        self.batchnorm_branch_2a=nn.BatchNorm1d(out_channels)
        self.conv_branch_2b=nn.Conv1d(out_channels, out_channels, 3, padding=8, dilation=dialtion[1], bias=False)
        self.batchnorm_branch_2b=nn.BatchNorm1d(out_channels)
   
    
    def forward(self, x):
        identity = x

        x = self.conv_branch_2a(x)
        x = self.batchnorm_branch_2a(x)
        x = F.relu(x)
        x = self.conv_branch_2b(x)
        x = self.batchnorm_branch_2b(x)

        x += identity
        x = F.relu(x)

        return x
      
class ResNet18(nn.Module):
    
    def __init__(self, classes):
        super(ResNet18, self).__init__()
        self.conv1 = self.__conv1_block(1, 64)
        self.conv2_a = ConvBlock(64, 64, 1, (1,2))
        self.conv2_b = IdentityBlock(64, 64, 1, (4,8))
        self.conv3_a = ConvBlock(64, 128, 2, (1,2))
        self.conv3_b = IdentityBlock(128, 128, 1, (4,8))
        self.conv4_a = ConvBlock(128, 256, 2, (1,2))
        self.conv4_b = IdentityBlock(256, 256, 1, (4,8))
        self.conv5_a = ConvBlock(256, 512, 2, (1,2))
        self.conv5_b = IdentityBlock(512, 512, 1, (4,8))      

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_a(x)
        x = self.conv2_b(x)
        x = self.conv3_a(x)
        x = self.conv3_b(x)
        x = self.conv4_a(x)
        x = self.conv4_b(x)
        x = self.conv5_a(x)
        x = self.conv5_b(x)
        x = F.adaptive_avg_pool1d(x, 1)

        return x        

    # convolutional block
    def __conv1_block(self, channels_in, channels_out):
        return nn.Sequential(nn.Conv1d(channels_in, channels_out, kernel_size=7, stride=2, padding=3, bias=False),
                             nn.BatchNorm1d(channels_out),
                             nn.ReLU(),
                             nn.MaxPool1d(3, stride=2, padding=1))


class VarNet(nn.Module):
    
    def __init__(self, classes):
        super(VarNet, self).__init__()
        self.resnet_block = ResNet18(classes)
        self.meta_block = self.__metadata_block(7, 32)
        self.fc_block = self.__fc_block(512+32, classes)

    def forward(self, x):
        res_input = x[:, :, :5000]
        meta_input = x[:, :, 5000:].squeeze(dim=1) # remove channel: (N, C, L) -> (N, L)

        res_output = self.resnet_block(res_input)
        meta_output = self.meta_block(meta_input)
        
        res_meta = torch.cat((res_output.squeeze(dim=2), meta_output), dim=1)

        out = self.fc_block(res_meta)

        return out

    def __metadata_block(self, in_features, out_features):    
        return nn.Sequential(nn.Linear(in_features, out_features),
                             nn.BatchNorm1d(out_features),
                             nn.ReLU())

    def __fc_block(self, in_features, classes):
        return nn.Sequential(nn.Linear(in_features, 1024),
                             nn.BatchNorm1d(1024),
                             nn.ReLU(),
                             nn.Dropout(0.5),
                             nn.Linear(1024, classes))
                             #nn.Softmax(dim=1))     
    
if __name__ == '__main__':
    #model = ResNet18(100)
    #output = model(torch.rand(2, 1, 5000))    
    #print(summary(model, (1, 5000)))  
    #print(output.shape) 

    model = VarNet(100)
    output = model(torch.rand(2, 1, 5007))    
    print(summary(model, (1, 5007)))  
    print(output.shape) 
