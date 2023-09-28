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
from torch.optim import Adamax
import numpy as np

from dataloader import train_test_dataloader
from dfnet import DFNet
from metrics import *

def main(X, y):
    # get train_dataloader & test_dataloader
    train_dataloader, test_dataloader = train_test_dataloader(X, y)    
    # select cpu/gpu mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    # number of outputs = number of classes
    CLASSES = 101
    # create DFNet model    
    model = DFNet(CLASSES).to(device)     
    # loss function    
    loss_fn = nn.CrossEntropyLoss()
    # optimizer
    optimizer = Adamax(model.parameters())

    # training loop
    logger.info(f"run the training loop.")
    N_EPOCHS = 30 # Number of training epoch
    training_loop(N_EPOCHS, train_dataloader, model, loss_fn, optimizer, device)
    
    # test
    logger.info(f"run the test.")
    lines = test(test_dataloader, model, CLASSES, device)

    return lines

def training_loop(n_epochs, train_dataloader, model, loss_fn, optimizer, device):
    
    for epoch in range(n_epochs):
        # loss value
        running_loss = 0.0
        # update batch_normalization and enable dropout layer
        model.train()

        # loop
        for batch_X, batch_y in train_dataloader:
            # dataset load to device
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # 1. Compute prediction error
            pred = model(batch_X)
            loss = loss_fn(pred, batch_y)

            # 2. Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # accumulate loss value
            running_loss += loss.item()
        
        # print loss average:
        print(f"[Epoch_{epoch+1}] Avg_loss(total_loss/num_batch): {running_loss}/{len(train_dataloader)} = {running_loss/len(train_dataloader)}")  

def test(test_dataloader, model, device):
    # prediction & true label
    y_pred, y_true = [], []
    
    # not update batch_normalization and disable dropout layer
    model.eval()
    # set gradient calculation to off
    with torch.no_grad():
        for batch_X, batch_y in test_dataloader:
            # data load to device
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
           
            # get the prediction of the model.
            prediction = model(batch_X)

            y_pred.extend([np.argmax(x) for x in F.softmax(prediction, dim=1).data.cpu().tolist()])
            y_true.extend(batch_y.data.cpu().tolist())

    # get score
    lines = openworld_score(y_true, y_pred, max(y_true))

    return lines 

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




