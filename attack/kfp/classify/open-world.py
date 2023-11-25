#!/usr/bin/env python3

"""
<file>    classify.py
<brief>   classify website fingerprints.
"""

import os
import sys
import time
import random
import pickle
import logging
import argparse
from os.path import join, abspath, dirname, pardir, exists
from collections import Counter
import numpy as np

from sklearn.model_selection import train_test_split
from metrics import *
from kfp_classifier import KFPClassifier

def main(X, y, classes, m_file, load_model): 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random.randint(0,10000), stratify=y)
    print(f"X_train:{X_train.shape}, X_test:{X_test.shape}")

    if load_model: # load the trained model.
        clf = KFPClassifier(m_file) 
    else: # create & train a new model.    
        logger.info(f"run the training.")
        clf = KFPClassifier()
        clf.train(X_train, y_train)
        with open(m_file, 'wb') as f:
            pickle.dump((clf.fp_generator, clf.knn_finder, clf.fp_label), f)

    logger.info(f"run the test.")
    pred, score = clf.test(X_test)

    # metrics
    if classes == 2:
        res = ow_score_twoclass(y_test, pred)
        #pr_curve(y_test, score)
        #roc_curve(y_test, score)
    else:
        res = ow_score_multiclass(y_test, pred)

    return res, y_test, score

# logger and arguments
def logger_and_arguments():  
    logging.basicConfig(format="[%(asctime)s]>>> %(message)s", level=logging.INFO, datefmt = "%Y-%m-%d %H:%M:%S") # get logger
    logger = logging.getLogger()
    
    parser = argparse.ArgumentParser() # parse arugment
    parser.add_argument("--in", required=True, help="load the feature.pkl")
    parser.add_argument("--out", required=True, help="save the result")
    parser.add_argument("--class", required=True, help="classes")
    parser.add_argument('--load_model', action='store_true', default=False, help="load the trained model")
    args = vars(parser.parse_args())

    return logger, args

if __name__ == "__main__":
    BASE_DIR = abspath(join(dirname(__file__), pardir))
    cur_time = time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())
    logger, args = logger_and_arguments()
    logger.info(f"Arguments:{args}")

    res_dir = join(BASE_DIR, "res")
    if not exists(res_dir):
        os.makedirs(res_dir)
    m_dir = join(BASE_DIR, "model", args["in"])
    if not exists(m_dir):
        os.makedirs(m_dir)

    with open(join(BASE_DIR, "feature", args["in"], "feature.pkl"), "rb") as f:
        X, y = pickle.load(f) # load X, y
        if int(args["class"]) == 2:
            y = [1 if e != 0 else e for e in y] # mon is 1, unmon is 0.
        logger.info(f"X:{np.array(X).shape}, y:{np.array(y).shape}")
        logger.info(f"labels:{Counter(y)}")

    # main function    
    res, label, score = main(X, y, int(args["class"]), join(m_dir,"model.pkl"), args["load_model"])

    # save the result
    with open(join(res_dir, args["out"]+".txt"), "a") as f:
        f.writelines(res)
    if int(args["class"]) == 2:
        with open(join(res_dir, cur_time+"_"+args["out"]+".pkl"), "wb") as f:
            pickle.dump((label, score), f)       

    logger.info(f"Classification completed!")
