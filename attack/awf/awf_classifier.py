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

