#!/usr/bin/env python3

"""
<file>    xxnet.py
<brief>   XX model with PyTorch
"""

import torch
import torch.nn as nn
from torchsummary import summary
       
class XX_Net(nn.Module):

    def __init__(self, classes):
        super(XX_Net, self).__init__()
   
    def forward(self, x):

        return x        


if __name__ == '__main__':
    model = XX_Net(100)
    print(summary(model, (1, 5000))) 

    output = model(torch.rand(2, 1, 5000))    
    print(output.shape) 
