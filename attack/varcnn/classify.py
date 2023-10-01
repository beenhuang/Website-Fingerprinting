#!/usr/bin/env python3

"""
<file>    classify.py
<brief>   classify website fingerprints.
"""

import os
import sys
import time
import pickle
import logging
import argparse
from os.path import join, basename, abspath, splitext, dirname, pardir, isdir, exists

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

from dataloader import train_test_dataloader
from varnet import VarNet
from metrics import *

def main(X, y):
    X_direct = [e[0] for e in X]
    X_time = [e[1] for e in X]

    # get train_dataloader & test_dataloader
    logger.info(f"split the direction data.")
    train_dataloader_direct, valid_dataloader_direct, test_dataloader_direct = train_valid_test_dataloader(X_direct, y)    
    logger.info(f"split the time data")
    train_dataloader_time, valid_dataloader_time, test_dataloader_time = train_valid_test_dataloader(X_time, y) 
    

    # select cpu/gpu mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    # number of outputs = number of classes
    CLASSES = 101
    # create VarNet model    
    model_direct = VarNet(CLASSES).to(device) 
    model_time = VarNet(CLASSES).to(device)    
    # loss function    
    loss_fn = nn.CrossEntropyLoss()
    # optimizer
    optimizer_direct = Adam(model_direct.parameters())
    optimizer_time = Adam(model_time.parameters())
    
    scheduler_direct = ReduceLROnPlateau(optimizer_direct, mode='min', factor=np.sqrt(0.1), patience=5, min_lr=1e-5)
    scheduler_time = ReduceLROnPlateau(optimizer_time, mode='min', factor=np.sqrt(0.1), patience=5, min_lr=1e-5)

    # training loop
    logger.info(f"run the training loop.")
    N_EPOCHS = 150 # Number of training epoch

    logger.info(f"training the direction model.")
    training_validate_loop(N_EPOCHS, train_dataloader_direct, valid_dataloader_direct, model_direct, loss_fn, optimizer_direct, scheduler_direct, device)
    logger.info(f"training the time model.")
    training_validate_loop(N_EPOCHS, train_dataloader_time, valid_dataloader_time, model_time, loss_fn, optimizer_time, scheduler_time, device)

    # test
    logger.info(f"run the test.")
    lines = test(test_dataloader, model_direct, model_time, device)

    return lines

def training_validate_loop(n_epochs, train_dataloader, valid_dataloader, model, loss_fn, optimizer, scheduler, device):
    
    for epoch in range(n_epochs):
        train_one_epoch(epoch+1, train_dataloader, model, loss_fn, optimizer, device)

        val_loss = validate(epoch+1, valid_dataloader, model, loss_fn, device)
        scheduler.step(val_loss)

def train_one_epoch(epoch_index, train_dataloader, model, loss_fn, optimizer, device):
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
    print(f"[Epoch_{epoch_index}] Avg_loss(total_loss/num_batch): {running_loss}/{len(train_dataloader)} = {running_loss/len(train_dataloader)}")  

def validate(epoch_index, valid_dataloader, model, loss_fn, device):
    # validate loss value
    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

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
    print(f"[Epoch_{epoch_index}] Avg_vloss(total_vloss/num_batch): {running_vloss}/{len(valid_dataloader)} = {avg_vloss}")  

    return avg_vloss

def test(test_dataloader, model_direct, model_time, device):
    # prediction & true label
    y_pred_sm, y_true = [], []
    
    # not update batch_normalization and disable dropout layer
    model.eval()
    # set gradient calculation to off
    with torch.no_grad():
        for batch_X, batch_y in test_dataloader:
            # data load to device
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
           
            # get the prediction of the model.
            pred_direct = model_direct(batch_X)
            pred_time = model_time(batch_X)

            pred_direct_sm = [x for x in F.softmax(pred_direct, dim=1).data.cpu().tolist()]
            pred_time_sm = [x for x in F.softmax(pred_time, dim=1).data.cpu().tolist()]
            batch_pred_sm = [(np.argmax(x), np.max(x)) for x in torch.div(torch.add(pred_direct_sm, pred_time_sm), 2)]
            y_pred_sm.extend(batch_pred_sm)
            y_true.extend(batch_y.data.cpu().tolist())

    score_total = []
    # threshold = [0.0, 0.1, 0.2, ... , 1.0]
    for threshold in np.arange(0, 1.01, 0.1):
        score_total.append(f"threshold: {threshold}\n")
        score_total.extend(ow_score_with_th(y_true, y_pred_sm, max(y_true), threshold))

    return score_total 

# logger and arguments
def logger_and_arguments():
    # get logger
    logging.basicConfig(format="[%(asctime)s]>>> %(message)s", level=logging.INFO, datefmt = "%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(splitext(basename(__file__))[0])
    
    # parse arugment
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in", required=True, help="load the feature.pkl")
    parser.add_argument("-o", "--out", required=True, help="save the result")
    args = vars(parser.parse_args())

    return logger, args

if __name__ == "__main__":
    CURRENT_TIME = time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())
    BASE_DIR = abspath(join(dirname(__file__)))
    FEATURE_DIR = join(BASE_DIR, "feature")
    RESULT_DIR = join(BASE_DIR, "result")
    if not exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    logger, args = logger_and_arguments()
    logger.info(f"Arguments: {args}")

    # load X, y
    with open(join(FEATURE_DIR, args["in"]), "rb") as f:
        X, y = pickle.load(f)
        logger.info(f"X:{len(X)}, y:{len(y)}")

    # main function    
    lines = main(X, y)

    # save the result
    with open(join(RESULT_DIR, CURRENT_TIME+"_"+args["out"]+".txt"), "a") as f:
        f.writelines(lines)

    logger.info(f"Complete!")




