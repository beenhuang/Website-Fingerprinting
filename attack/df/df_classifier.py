#!/usr/bin/env python3

"""
<file>    dfnet.py
<brief>   DF model with PyTorch
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adamax
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torchsummary import summary


class DFClassifier():
    def __init__(self, classes, device, m_file=None):
        self.device = device
        self.classes = classes
        self.model = DFNet(classes).to(device) if m_file == None else torch.load(m_file).to(device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = Adamax(params=self.model.parameters())
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=np.sqrt(0.1), patience=5, min_lr=1e-5)

    def train(self, n_epochs, train_dl, valid_data=None):
        for epoch in range(n_epochs):
            self.__train_one_epoch(epoch+1, train_dl)

            if valid_data != None: # valiate
                val_loss, acc = self.__validate(epoch+1, valid_data)
                #self.scheduler.step(val_loss)
            
    def __train_one_epoch(self, epoch_idx, train_dl):
        self.model.train() # update batch_normalization and enable dropout layer

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

        print(f"[Epoch_{epoch_idx}_valid] Total_loss:{vloss:.4f}, Acc:{acc}")  

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
