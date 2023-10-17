#!/usr/bin/env python3

"""
<file>    dfnet.py
<brief>   DF model with PyTorch
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adamax
import torch.nn.functional as F
from torchsummary import summary


class DFClassifier():
    def __init__(self, classes, device, m_file=None):
        self.device = device
        self.classes = classes
        self.model = DFNet(classes).to(device) if m_file == None else torch.load(m_file).to(device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = Adamax(params=self.model.parameters())
        
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
