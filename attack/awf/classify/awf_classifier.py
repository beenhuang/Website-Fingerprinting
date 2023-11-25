#!/usr/bin/env python3

"""
<file>    awfnet.py
<brief>   AWF model with PyTorch
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import RMSprop
import torch.nn.functional as F
from torchsummary import summary


class AWFClassifier():
    def __init__(self, classes, device, m_file=None):
        self.device = device
        self.classes = classes
        self.model = AWFNet(classes).to(device) if m_file == None else torch.load(m_file).to(device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = RMSprop(self.model.parameters(), lr=0.001)
        self.n_epochs = 8
        
    def train(self, train_dl, valid_data=None):
        for epoch in range(self.n_epochs):
            self.__train_one_epoch(epoch+1, train_dl)

            if valid_data != None: # valiate
                val_loss = self.__validate(epoch+1, valid_data)
            
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

        print(f"[Epoch_{epoch_idx}_valid] loss_total:{vloss:.4f}, Acc:{acc:.4f}")  
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

