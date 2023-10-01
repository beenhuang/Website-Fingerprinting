#!/usr/bin/env python3

"""
<file>    classify.py
<brief>   classify website fingerprints.
"""

import numpy as np
from metrics import *


def training_loop(n_epochs, train_dataloader, model, loss_fn, optimizer, device, valid_dataloader=None, lr_reducer=None):
    
    for epoch in range(n_epochs):
        train_one_epoch(epoch+1, train_dataloader, model, loss_fn, optimizer, device)

        if valid_dataloader != None:
            val_loss = validate(epoch+1, valid_dataloader, model, loss_fn, device)
        if lr_reducer != None:
            lr_reducer.step(val_loss)

def train_one_epoch(epoch_idx, train_dataloader, model, loss_fn, optimizer, device):
    # update batch_normalization and enable dropout layer
    model.train()

    # loss value
    running_loss = 0.0

    for batch_X, batch_y in train_dataloader:
        # dataset load to device
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        pred = model(batch_X)

        # Compute the loss and its gradients
        loss = loss_fn(pred, batch_y)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # accumulate loss value
        running_loss += loss.item()
    
    # print loss average:
    print(f"[Epoch_{epoch_idx}] Avg_loss(total_loss/num_batch): {running_loss}/{len(train_dataloader)}={running_loss/len(train_dataloader)}")  

def validate(epoch_idx, valid_dataloader, model, loss_fn, device):
    # Set the model to evaluation mode, disabling dropout and using population statistics for batch normalization.
    model.eval()

    # validate loss value
    running_vloss = 0.0

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for batch_X, batch_y in valid_dataloader:
            # dataset load to device
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Make predictions for this batch
            pred = model(batch_X)

            # Compute the loss and its gradients
            vloss = loss_fn(pred, batch_y)
            running_vloss += vloss

    avg_vloss = running_vloss/len(valid_dataloader)
    # print loss average:
    print(f"[Epoch_{epoch_idx}] Avg_vloss(total_vloss/num_batch): {running_vloss}/{len(valid_dataloader)} = {avg_vloss}")  

    return avg_vloss

def test(test_dataloader, model, classes, device):
    # not update batch_normalization and disable dropout layer
    model.eval()

    # prediction, true label
    y_true, y_pred = [], []
    
    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for batch_X, batch_y in test_dataloader:
            # dataset load to device
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Make predictions for this batch
            prediction = model(batch_X)

            y_pred.extend([np.argmax(x) for x in F.softmax(prediction, dim=1).data.cpu().tolist()])
            y_true.extend(batch_y.data.cpu().tolist())

    lines = openworld_score(y_true, y_pred, max(y_true))

    return lines 


if __name__ == "__main__":




