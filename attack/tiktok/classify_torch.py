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
from sklearn.model_selection import train_test_split

from metrics import *

# logger and arguments
def logger_and_arguments():
    # get logger
    logging.basicConfig(format="[%(asctime)s]>>> %(message)s", level=logging.INFO, datefmt = "%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(splitext(basename(__file__))[0])
    
    # parse arugment
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--in", required=True, help="load the feature.pkl")
    parser.add_argument("-o", "--out", required=True, help="save the result")
    args = vars(parser.parse_args())

    return logger, args


import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adamax
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from dfnet_torch import DFNet

# Training the DF model
EPOCH = 30   # Number of training epoch
BATCH_SIZE = 128 # Batch size

# number of outputs = number of classes
CLASSES = 101
INPUT_SHAPE = (5000,1)

class DFDataset(Dataset):
    def __init__(self, X, y):
        self.datas=X
        self.labels=y
        
    def __getitem__(self, index):
        data = torch.from_numpy(self.datas[index])
        label = torch.tensor(self.labels[index]) 

        return data, label
    
    def __len__(self):
        return len(self.datas)

def spilt_dataset(X, y):
    X = np.array(X, dtype=np.float32)
    X = X[:,np.newaxis,:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=247, stratify=y)

    train_data = DFDataset(X_train, y_train)
    test_data = DFDataset(X_test, y_test)
    logger.info(f"split: traning size: {len(train_data)}, test size: {len(test_data)}")

    return train_data, test_data

def train_loop(dataloader, model, loss_function, optimizer, device):
    # loss value
    running_loss = 0.0
    # number of batches 
    num_batch = len(dataloader)

    # update batch_normalization and enable dropout layer
    model.train()
    
    # loop
    for X, y in dataloader:
        # dataset load to device
        X, y = X.to(device), y.to(device)
        #print(f"tain_loop: X: {X}")

        # 1. Compute prediction error
        pred = model(X)
        loss = loss_function(pred, y)

        # 2. Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accumulate loss value
        running_loss += loss.item()
    
    # print loss average:
    print(f"[training] Avg_loss(loss/num_batch): {running_loss}/{num_batch} = {running_loss/num_batch}")        

def test(dataloader, model, device, classes):
    # prediction, true label
    y_true, y_pred_softmax = [], []
    
    # not update batch_normalization and disable dropout layer
    model.eval()
    # set gradient calculation to off
    with torch.no_grad():
        # travel
        for X, y in dataloader:
            # dataset load to device
            X, y = X.to(device), y.to(device)
            # 1. Compute prediction:
            prediction = model(X)
            # extend softmax result to the prediction list:
            y_pred_softmax.extend(F.softmax(prediction, dim=1).data.cpu().numpy().tolist())
            
            # extend actual label to the label list:
            y_true.extend(y.data.cpu().numpy().tolist())

    y_pred = [np.argmax(x) for x in y_pred_softmax]
    print(f"test -> prediction: {len(y_pred)}, label: {len(y_true)}")

    lines = openworld_score(y_true, y_pred, max(y_true))
    return lines 

def main(X, y, result_path):
    # split dataset
    train_data, test_data = spilt_dataset(X, y)    

    # select cpu/gpu mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    # create DFNet model    
    df_net = DFNet(CLASSES).to(device)         
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adamax(params=df_net.parameters())

    #### TRAINING ####
    logger.info(f"training: the Tik-Tok model")
    # training/validation dataloader: 
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    # training loop
    for i in range(EPOCH):
        logger.info(f"Epoch {i+1}")
        train_loop(train_dataloader, df_net, loss_function, optimizer, device)

    #### TESTING ####
    logger.info(f"test: the Tik-Tok model")
    # test dataloader:
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)
    # run test
    lines = test(test_dataloader, df_net, device, CLASSES)

    # save the result
    with open(result_path, "a") as f:
        f.writelines(lines)
        logger.info(f"saved the result.")

    logger.info(f"Complete!")


if __name__ == "__main__":
    CURRENT_TIME = time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())
    BASE_DIR = abspath(join(dirname(__file__)))
    FEATURE_DIR = join(BASE_DIR, "feature")
    RESULT_DIR = join(BASE_DIR, "result")

    try:
        logger, args = logger_and_arguments()
        logger.info(f"{basename(__file__)} -> Arguments: {args}")

        # load X, y
        feature_path = join(FEATURE_DIR, args["in"])
        with open(feature_path, "rb") as f:
            X, y = pickle.load(f)
            logger.info(f"load dataset:{len(X)}, labels:{len(y)}")

        if not exists(RESULT_DIR):
            os.makedirs(RESULT_DIR)
        #result_path = join(RESULT_DIR, CURRENT_TIME+"_"+args["out"]+".txt")
        result_path = join(RESULT_DIR, args["out"]+".txt")
        
        main(X, y, result_path)

    except KeyboardInterrupt:
        sys.exit(-1)




