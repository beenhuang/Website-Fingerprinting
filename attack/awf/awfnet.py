#!/usr/bin/env python3

"""
<file>    awfnet.py
<brief>   AWF model with PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
       
class AWFNet(nn.Module):

    def __init__(self, classes):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding="valid", stride=1)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=5, padding="valid", stride=1)
        self.fc = nn.Linear(32*123, classes)

    def forward(self, x):
        x = F.dropout(x, p=0.1)

        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=4)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=4)

        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        #out = F.softmax(out)

        return x        
    
if __name__ == '__main__':
    model = AWFNet(100)
    output = model(torch.rand(2, 1, 2000))
    
    print(output.shape)        
    print(summary(model, (1, 2000)))

