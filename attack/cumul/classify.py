#!/usr/bin/env python3

"""
<file>    classify.py
<brief>   classify the dataset using svm_classifer.py model.
"""

import argparse
import os
import sys
import time
import pickle
import logging
import numpy as np
from os.path import join, basename, abspath, splitext, dirname, pardir, isdir, exists

from sklearn.model_selection import train_test_split

from svm_classifier import train_cumul, test_cumul, get_openworld_score

def get_logger():
    logging.basicConfig(format="[%(asctime)s] >> %(message)s", level=logging.INFO)
    logger = logging.getLogger(splitext(basename(__file__))[0])
    
    return logger

# parse arugment
def parse_arguments():
    # argument parser
    parser = argparse.ArgumentParser(description="CUMUL")

    # INPUT
    parser.add_argument("-i", "--in", required=True, help="load feature data")
    # OUTPUT
    parser.add_argument("-o", "--out", required=True, help="save the result")

    args = vars(parser.parse_args())

    return args


# [MAIN] function
def main(X, y, result_path):
    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=247, stratify=y)
    logger.info(f"X_train:{len(X_train)}, y_train:{len(y_train)}; X_test:{len(X_test)}, y_test:{len(y_test)}")
    #logger.debug(f"y_test: {y_test}")

    # 1. training
    model = train_cumul(X_train, y_train)
    logger.info(f"[TRAINED] the SVM Classifier.")

    # 2. test
    y_pred = test_cumul(model, X_test)
    logger.info(f"[GOT] predicted the labels from the test samples.")
    
    # get the open-world metrics score
    lines = get_openworld_score(y_test, y_pred, max(y))
    logger.info(f"[GOT] the result.")
    
    with open(result_path, "w") as f:
        f.writelines(lines)

    logger.info(f"Complete!")



if __name__ == "__main__":
    MODULE_NAME = basename(__file__)
    CURRENT_TIME = time.strftime("%Y.%m.%d-%H:%M:%S_", time.localtime())

    BASE_DIR = abspath(join(dirname(__file__)))
    FEATURE_DIR = join(BASE_DIR, "feature")
    RESULT_DIR = join(BASE_DIR, "result")

    try:
        logger = get_logger()
        args = parse_arguments()
        logger.info(f"{MODULE_NAME} -> Arguments: {args}")

        feature_path = join(FEATURE_DIR, args["in"])
        with open(feature_path, "rb") as f:
            X, y = pickle.load(f)
            logger.info(f"[LOADED] dataset:{len(X)}, labels:{len(y)}")

        if not exists(RESULT_DIR):
            os.makedirs(RESULT_DIR)

        result_path = join(RESULT_DIR, CURRENT_TIME+"_"+args["out"]+".txt")

        main(X, y, result_path)

        # [FIND BEST HYPERPARAMETERS]
        #lines = find_optimal_hyperparams(X, y)
        #print(lines)
        #sys.exit(0)

    except KeyboardInterrupt:
        sys.exit(-1)

