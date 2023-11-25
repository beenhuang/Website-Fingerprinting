#!/usr/bin/env python3

"""
<file>    one-page.py
<brief>   classify website fingerprints.
"""

import os
import sys
import time
import random
import pickle
import logging
import argparse
from os.path import abspath, dirname, pardir, join, exists
from collections import Counter
import numpy as np

from dataloader import *
from metrics import *
from df_classifier import DFClassifier

def main(X, y):
    train_dl, test_data = train_test_dataloader(X, y)  

    # select cpu/gpu mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")
    
    clf = DFClassifier(2, device)  
    logger.info(f"run the training loop.") # train
    clf.train(30, train_dl)
        
    logger.info(f"run the test.") # test
    label, pred, score = clf.test(test_data)
    res = ow_score_twoclass(label, pred)

    return res

# logger and arguments
def logger_and_arguments():
    logging.basicConfig(format="[%(asctime)s]>>> %(message)s", level=logging.INFO, datefmt = "%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", required=True, help="feature directory")
    parser.add_argument("--out", required=True, help="result directory")
    args = vars(parser.parse_args())

    return logger, args

if __name__ == "__main__":
    BASE_DIR = abspath(join(dirname(__file__), pardir))
    cur_time = time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())
    logger, args = logger_and_arguments()
    logger.info(f"Arguments: {args}")

    res_dir = join(BASE_DIR, "onepage-res")
    if not exists(res_dir):
        os.makedirs(res_dir)

    # load X, y
    with open(join(BASE_DIR, "feature", args["in"], "feature.pkl"), "rb") as f:
        X, y = pickle.load(f)

    unmon_inst_total = [x for x in zip(X,y) if x[1] == 0]
    for label in range(1, max(y)+1): 
        mon_inst = [(x[0], 1) for x in zip(X,y) if x[1] == label]
        unmon_inst = random.choices(unmon_inst_total, k=len(mon_inst))
        op_X, op_y = zip(*(mon_inst+unmon_inst))
        logger.info(f"X:{np.array(op_X).shape}, y:{np.array(op_y).shape}")
        #logger.info(f"labels:{Counter(op_y)}")

        # main function    
        res = main(op_X, op_y)

        with open(join(res_dir, args["out"]+".txt"), "a") as f:
            f.write(f"monitored label:{label}\n")
            f.writelines(res)
        logger.info(f"label_{label}: completed!")    

    logger.info(f"Classification completed!")
