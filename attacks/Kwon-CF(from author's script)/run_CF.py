#!/usr/bin/env python3

"""
<file>    CF.py
<brief>   Kwon's circuit fingerprinting attack
"""

import argparse
import os
import sys
import time
import logging
import numpy as np
from os.path import join, basename, abspath, splitext, dirname, pardir, isdir
import multiprocessing as mp

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from exfeature import extract_features
from c45 import C45


# CONSTANTS
MODULE_NAME = basename(__file__)
CURRENT_TIME = time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())

BASE_DIR = abspath(join(dirname(__file__)))
INPUT_DIR = join(BASE_DIR)
OUTPUT_DIR = join(BASE_DIR, "results")


# constants
DIRECTION_OUT = 1.0
DIRECTION_IN = -1.0

def get_logger():
    logging.basicConfig(format="[%(asctime)s]>>> %(message)s", level=logging.INFO, datefmt = "%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(splitext(basename(__file__))[0])
    
    return logger


# [FUNC] parse arugment
def parse_arguments():
    # argument parser
    parser = argparse.ArgumentParser(description="cf")

    # 1. INPUT: load ds-*.pkl dataset
    parser.add_argument("-i", "--in", required=True, metavar="<trace file directory>", help="load trace data")
    # 2. OUTPUT: save overhead in the overhead-*.txt file
    parser.add_argument("-o", "--out", required=True, metavar="<result-file>", help="save results in the text file.")

    args = vars(parser.parse_args())

    return args


def generate_feature_vectors(data_dir):
    files = [join(data_dir, file) for file in os.listdir(data_dir)]
    
    with mp.Pool(mp.cpu_count()) as pool:
        data = pool.map(extract_features, files)

    X, y = [], []
    for elem in data:  
        X.append(elem[0])
        y.append(elem[1])


    return X, y

# training
def train_cf(X_train, y_train):
    model = C45()
    model.fit(X_train, y_train)

    return model

# test
def test_cf(model, X_test):
    y_pred = model.predict(X_test)

    return y_pred


# open-world score
def get_openworld_score(y_true, y_pred, label_unmon):
    print(f"label_unmon: {label_unmon}")
    # TP-correct, TP-incorrect, FN  TN, FN
    tp_c, tp_i, fn, tn, fp = 0, 0, 0, 0, 0

    # traverse preditions
    for i in range(len(y_pred)):
        # [case_1]: positive sample, and predict positive and correct.
        if y_true[i] != label_unmon and y_pred[i] != label_unmon and y_pred[i] == y_true[i]:
            tp_c += 1
        # [case_2]: positive sample, predict positive but incorrect class.
        elif y_true[i] != label_unmon and y_pred[i] != label_unmon and y_pred[i] != y_true[i]:
            tp_i += 1
        # [case_3]: positive sample, predict negative.
        elif y_true[i] != label_unmon and y_pred[i] == label_unmon:
            fn += 1
        # [case_4]: negative sample, predict negative.    
        elif y_true[i] == label_unmon and y_pred[i] == y_true[i]:
            tn += 1
        # [case_5]: negative sample, predict positive    
        elif y_true[i] == label_unmon and y_pred[i] != y_true[i]:
            fp += 1   
        else:
            sys.exit(f"[ERROR]: {y_pred[i]}, {y_true[i]}")        

    # accuracy
    accuracy = (tp_c+tn) / float(tp_c+tp_i+fn+tn+fp)
    # precision      
    precision = tp_c / float(tp_c+tp_i+fp)
    # recall
    recall = tp_c / float(tp_c+tp_i+fn)
    # F-score
    f1 = 2*(precision*recall) / float(precision+recall)

    lines = []
    lines.append(f"[POS] TP-c: {tp_c}, TP-i(incorrect class): {tp_i}, FN: {fn}\n")
    lines.append(f"[NEG] TN: {tn}, FP: {fp}\n\n")
    lines.append(f"accuracy: {accuracy}\n")
    lines.append(f"precision: {precision}\n")
    lines.append(f"recall: {recall}\n")
    lines.append(f"F1: {f1}\n")
    return lines


# MAIN function
def main():
    logger = get_logger()
    logger.info(f"{MODULE_NAME}: start to run.")

    # parse arguments
    args = parse_arguments()
    logger.info(f"Arguments: {args}")

    # load dataset&labels
    data_dir = join(INPUT_DIR, args["in"])
    X, y = generate_feature_vectors(data_dir)
    logger.info(f"[LOADED] dataset&labels, length: {len(X)}, {len(y)}")

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=247, stratify=y)
    
    print(f"X_train: {len(X_train)}, X_test: {len(X_test)}")

    # training
    model= train_cf(X_train, y_train)
    logger.info(f"[TRAINED] C4.5 model of cf.")

    # test
    y_pred = test_cf(model, X_test)
    logger.info(f"[GOT] predicted labels of test samples.")
    
    # get metrics value
    lines = get_openworld_score(y_test, y_pred, max(y_test))
    logger.info(f"[CALCULATED] open-world scores.")
    
    with open(join(OUTPUT_DIR, f"{CURRENT_TIME}_{args['out']}.txt"), "w") as f:
        f.writelines(lines)
        logger.info(f"[SAVED] metrics scores in the {args['out']}.txt")

    logger.info(f"{MODULE_NAME}: complete successfully.\n")


if __name__ == "__main__":
    sys.exit(main())


