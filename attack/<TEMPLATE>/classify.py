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

from metrics import openworld_score, closedworld_score

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

def main(X, y, result_path):
    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=247, stratify=y)

    # 1. training
    #
    logger.info(f"[TRAINED] fingerprint generator & fingerprints of train instances.")

    # 2. test
    #
    logger.info(f"[GOT] predicted labels of test samples.")
    
    # 3. get metrics value
    #lines = openworld_score(y_test, y_pred, max(y))
    logger.info(f"[CALCULATED] metrics.")
    
    # save the result
    with open(result_path, "w") as f:
        f.writelines(lines)
        logger.info(f"[SAVED] results.")

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
            logger.info(f"[LOADED] dataset:{len(X)}, labels:{len(y)}")

        if not exists(RESULT_DIR):
            os.makedirs(RESULT_DIR)
        result_path = join(RESULT_DIR, CURRENT_TIME+"_"+args["out"]+".txt")

        main(X, y, result_path)

    except KeyboardInterrupt:
        sys.exit(-1)




