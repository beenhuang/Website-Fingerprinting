#!/usr/bin/env python3

"""
<file>    rf.py
<brief>   RF model using the PyTorch api
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torchsummary import summary

class RFClassifier():
    def __init__(self, classes, device, m_file=None):
        self.device = device
        self.classes = classes
        self.model = RFNet(classes).to(device) if m_file == None else torch.load(m_file).to(device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=0.0005, weight_decay=0.001)
        self.n_epochs = 30

    def train(self, train_dl, valid_data=None):
        for epoch in range(self.n_epochs):
            self.__train_one_epoch(epoch+1, train_dl)

            if valid_data != None: # valiate
                cur_loss, cur_acc = self.__validate(epoch+1, valid_data)
            
    def __train_one_epoch(self, epoch_idx, train_dl):
        self.model.train() # update batch_normalization and enable dropout layer.
        running_loss = 0.0 # loss value

        for batch_X, batch_y in train_dl:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device) # dataset load to device

            self.optimizer.zero_grad() # Zero your gradients for every batch!
            pred = self.model(batch_X) # Make predictions for this batch
            loss = self.loss_fn(pred, batch_y) # Compute the loss and its gradients
            loss.backward()
            self.optimizer.step() # Adjust learning weights
            
            running_loss += loss.item() # accumulate loss value
        
        print(f"[Epoch_{epoch_idx}] Avg_loss:{running_loss/len(train_dl):.4f} (total_loss/num_batch):{running_loss:.4f}/{len(train_dl)}")  

    def __validate(self, epoch_idx, valid_data):
        X_valid, y_valid = valid_data[0], valid_data[1]
        self.model.eval() # Set the model to evaluation mode, disabling dropout and using population statistics for batch normalization.
            
        with torch.no_grad(): # Disable gradient computation and reduce memory consumption.
            X_valid, y_valid = X_valid.to(self.device), y_valid.to(self.device) # dataset load to device
            
            pred = self.model(X_valid) # Make predictions for this batch
            vloss = self.loss_fn(pred, y_valid) # Compute the loss and its gradients

            y_pred = [np.argmax(x) for x in F.softmax(pred, dim=1).data.cpu().tolist()]
            acc = sum([1 if x[0] == x[1] else 0 for x in zip(y_valid,y_pred)]) / float(len(y_pred))

        print(f"[Epoch_{epoch_idx}_valid] loss_total:{vloss:.4f}, Accuracy:{acc:.4f}")  
        return vloss, acc          

    def test(self, test_data):
        X_test, y_test = test_data[0], test_data[1]
        self.model.eval() # not update batch_normalization and disable dropout layer
        
        with torch.no_grad(): # Disable gradient computation and reduce memory consumption.
            X_test = X_test.to(self.device) # dataset load to device
            pred = self.model(X_test) # Make predictions for this batch

            y_pred = [np.argmax(x) for x in F.softmax(pred, dim=1).data.cpu().tolist()]
            pos_score=[x[1] for x in F.softmax(pred, dim=1).data.cpu().tolist()] if self.classes == 2 else None
 
        return y_test.tolist(), y_pred, pos_score
  
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
