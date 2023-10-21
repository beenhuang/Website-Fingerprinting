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
        
    def train(self, n_epochs, train_dataloader, valid_dataloader=None):
        for epoch in range(n_epochs):
            self.__train_one_epoch(epoch+1, train_dataloader)

            if valid_dataloader != None: # valiate
                val_loss = self.__validate(epoch+1, valid_dataloader)
            
    def __train_one_epoch(self, epoch_idx, train_dataloader):
        # update batch_normalization and enable dropout layer
        self.model.train()

        # loss value
        running_loss = 0.0

        for batch_X, batch_y in train_dataloader:
            # dataset load to device
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

            # train
            self.optimizer.zero_grad() # Zero your gradients for every batch!
            pred = self.model(batch_X) # Make predictions for this batch
            loss = self.loss_fn(pred, batch_y) # Compute the loss and its gradients
            loss.backward()
            self.optimizer.step() # Adjust learning weights

            # accumulate loss value
            running_loss += loss.item()
        
        # print loss average:
        print(f"[Epoch_{epoch_idx}] Avg_loss:{running_loss/len(train_dataloader):.4f} (total_loss/num_batch):{running_loss:.4f}/{len(train_dataloader)}")  

    def __validate(self, epoch_idx, valid_dataloader):
        # Set the model to evaluation mode, disabling dropout and using population statistics for batch normalization.
        self.model.eval()

        # validate loss value
        running_vloss = 0.0

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for batch_X, batch_y in valid_dataloader:
                # dataset load to device
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                # validate
                pred = self.model(batch_X) # Make predictions for this batch
                vloss = self.loss_fn(pred, batch_y) # Compute the loss and its gradients
                running_vloss += vloss

        avg_vloss = running_vloss/len(valid_dataloader)
        print(f"[Epoch_{epoch_idx}_validation] Avg_loss:{avg_vloss:.4f} (total_loss/num_batch):{running_vloss:.4f}/{len(valid_dataloader)}")  

        return avg_vloss

    def test(self, test_dataloader):
        # not update batch_normalization and disable dropout layer
        self.model.eval()

        y_true, y_pred, pos_score = [], [], []
        
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for batch_X, batch_y in test_dataloader:
                # dataset load to device
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                # Make predictions for this batch
                prediction = self.model(batch_X)

                y_true.extend(batch_y.data.cpu().tolist())
                y_pred.extend([np.argmax(x) for x in F.softmax(prediction, dim=1).data.cpu().tolist()])
                if self.classes == 2:
                    pos_score.extend([x[1] for x in F.softmax(prediction, dim=1).data.cpu().tolist()])
                
        return y_true, y_pred, pos_score
  
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
