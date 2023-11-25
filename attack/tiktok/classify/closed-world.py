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
from os.path import join, abspath, dirname, pardir, exists
from collections import Counter
import numpy as np
import torch

from dataloader import *
from metrics import *
from df_classifier import DFClassifier

def main(X, y, classes, m_file, load_model):
    train_dl, test_data = train_test_dataloader(X, y)  

    # select cpu/gpu mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    if load_model: # load the trained model
        clf = DFClassifier(classes, device, m_file) 
    else: # create&train new model 
        logger.info(f"run the training loop.")   
        clf = DFClassifier(classes, device)  # create DFNet mode   
        clf.train(30, train_dl)
        torch.save(clf.model, m_file)
        
    logger.info(f"run the test.")
    label, pred, score = clf.test(test_data)
    res = cw_score(label, pred)

    return res

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
    BASE_DIR = abspath(dirname(__file__))
    cur_time = time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())
    logger, args = logger_and_arguments()
    logger.info(f"Arguments: {args}")

    res_dir = join(BASE_DIR, "cw-res")
    if not exists(res_dir):
        os.makedirs(res_dir)
    m_dir = join(BASE_DIR, "model", args["in"])
    if not exists(m_dir):
        os.makedirs(m_dir)

    with open(join(BASE_DIR, "feature", args["in"], "feature.pkl"), "rb") as f:
        X, y = pickle.load(f)
        
        mon_inst = [(x[0], x[1]-1) for x in zip(X,y) if x[1] != 0]
        mon_X, mon_y = zip(*mon_inst)
        logger.info(f"X:{np.array(mon_X).shape}, y:{np.array(mon_y).shape}")
        logger.info(f"labels:{Counter(mon_y)}")
   
    res = main(mon_X, mon_y, int(args["class"]), join(m_dir,"model.pkl"), args["load_model"])

    with open(join(res_dir, args["out"]+".txt"), "a") as f: # save the result
        f.writelines(res)    

    logger.info(f"Complete!")

