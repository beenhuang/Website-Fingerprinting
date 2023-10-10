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
from traintest import *

def main(X, y, classes, m_file, load_model):
    # get train_dataloader & test_dataloader
    train_dataloader, test_dataloader = train_test_dataloader(X, y)    

    # select cpu/gpu mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    if load_model: # load the trained model
        model = torch.load(m_file).to(device) 

    else: # train new model    
        # create DFNet model    
        model = DFNet(classes).to(device)     
        # loss function    
        loss_fn = nn.CrossEntropyLoss()
        # optimizer
        optimizer = Adamax(params=model.parameters())

        # training loop
        logger.info(f"run the training loop.")
        training_loop(30, train_dataloader, model, loss_fn, optimizer, device)

        torch.save(model, m_file)
        
    # test
    logger.info(f"run the test")
    label, pred, score = test(test_dataloader, model, classes, device)

    # metrics
    res = ow_score_twoclass(label, pred)
    pr_curve(label, score)
    roc_curve(label, score)

    return res, label, score

# logger and arguments
def logger_and_arguments():
    # get logger
    logging.basicConfig(format="[%(asctime)s]>>> %(message)s", level=logging.INFO, datefmt = "%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger()
    
    # parse arugment
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", required=True, help="load the feature.pkl")
    parser.add_argument("--out", required=True, help="save the result")
    parser.add_argument("--class", required=True, help="classes")
    parser.add_argument('--load_model', action='store_true', default=False, help="load the trained model")
    args = vars(parser.parse_args())

    return logger, args

if __name__ == "__main__":
    BASE_DIR = abspath(join(dirname(__file__)))
    cur_time = time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())
    logger, args = logger_and_arguments()
    logger.info(f"Arguments: {args}")

    res_dir = join(BASE_DIR, "result")
    if not exists(res_dir):
        os.makedirs(res_dir)
    m_dir = join(BASE_DIR, "model", args["in"])
    if not exists(m_dir):
        os.makedirs(m_dir)

    # load X, y
    with open(join(BASE_DIR, "feature", args["in"], "feature.pkl"), "rb") as f:
        X, y = pickle.load(f)
        logger.info(f"X:{len(X)}, y:{len(y)}")

        if int(args["class"]) == 2:
            y = [1 if x != 0 else x for x in y] # mon is 1, unmon is 0.
            logger.info(f"[Two Class] mon:{y.count(1)}, unmon:{y.count(0)}")
    
    # main function    
    res, label, score = main(X, y, int(args["class"]), join(m_dir,"model.pkl"), args["load_model"])

    # save the result
    with open(join(res_dir, cur_time+"_"+args["out"]+".txt"), "a") as f:
        f.writelines(res)

    with open(join(res_dir, cur_time+"_"+args["out"]+".pkl"), "wb") as f:
        pickle.dump((label, score), f)       

    logger.info(f"Complete!")

