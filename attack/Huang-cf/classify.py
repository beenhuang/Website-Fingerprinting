#!/usr/bin/env python3

"""
<file>    classify.py
<brief>   classify the dataset using xx_classifer.py model.
"""

import argparse
import os
import sys
import time
import pickle
import logging
from os.path import join, basename, abspath, splitext, dirname, pardir, isdir, exists

from sklearn.model_selection import train_test_split

#from Classifier.rf_classifier import train_cf, test_cf
from Classifier.xg_classifier import train_cf, test_cf
#from Classifier.svm_classifier import train_cf, test_cf
#from Classifier.knn_classifier import train_cf, test_cf
#from Classifier.ada_classifier import train_cf, test_cf

from Classifier.metrics import closeworld_score, openworld_score


# create logger
def get_logger():
    logging.basicConfig(format="[%(asctime)s]>>> %(message)s", level=logging.INFO, datefmt = "%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(splitext(basename(__file__))[0])
    
    return logger

# parse arugment
def parse_arguments():
    # argument parser
    parser = argparse.ArgumentParser(description="cf")

    # INPUT
    parser.add_argument("-i", "--in", required=True, help="load feature.pkl file")
    # OUTPUT
    parser.add_argument("-o", "--out", required=True, help="save results")

    args = vars(parser.parse_args())

    return args


# MAIN function
def main(X, y, result_path):
    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=247, stratify=y)
    logger.info(f"X_train: {len(X_train)}, X_test: {len(X_test)}")
    #logger.debug(f"y_test: {y_test}")

    # 1. training
    model= train_cf(X_train, y_train)
    logger.info(f"[TRAINED] classifer of cf.")

    # 2. test
    y_pred = test_cf(model, X_test)
    logger.info(f"[GOT] predicted labels from test samples.")
    #logger.debug(f"y_pred: {y_pred}")
    
    # 3. get metrics value
    #lines = get_closeworld_score(y_test, y_pred)
    lines = openworld_score(y_test, y_pred, max(y))
  
    # save the results
    with open(result_path, "w") as f:
        f.writelines(lines)

    logger.info(f"Complete!")


if __name__ == "__main__":
    CURRENT_TIME = time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())

    BASE_DIR = abspath(join(dirname(__file__)))
    FEATURE_DIR = join(BASE_DIR, "feature")
    RESULT_DIR = join(BASE_DIR, "result")

    try:
        logger = get_logger()
        args = parse_arguments()
        logger.info(f"{basename(__file__)} -> Arguments: {args}")

        feature_path = join(FEATURE_DIR, args["in"])
        with open(feature_path, "rb") as f:
            X, y = pickle.load(f)
            logger.info(f"[LOADED] dataset:{len(X)}, labels:{len(y)}")

        if not exists(RESULT_DIR):
            os.makedirs(RESULT_DIR)
            
        result_path = join(RESULT_DIR, CURRENT_TIME+"_"+args["out"]+".txt")

        main(X, y, result_path)

    except KeyboardInterrupt:
        sys.exit(-1)


