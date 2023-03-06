#!/usr/bin/env python3

"""
<file>    run_cf.py
<brief>   Huang's circuit fingerprinting attack
"""

import argparse
import os
import sys
import csv
import time
import pickle
import logging
import numpy as np
from os.path import join, basename, abspath, splitext, dirname, pardir, isdir, exists
import multiprocessing as mp

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

# set logging level
LOG_LEVEL = logging.INFO

# random state for splitting
RANDOM_STATE = 247

# direction value
DIRECTION_OUT = 1.0
DIRECTION_IN = -1.0

# Random Forest's hyperparameters:
TREES = 1000
CRITERION = "gini"


#
def generate_feature_vectors(data_dir):
    # gen files
    files = [join(data_dir, "general-trace", file) for file in os.listdir(join(data_dir, "general-trace"))]
    # hs files
    hs_files = [join(data_dir, "hs-trace", file) for file in os.listdir(join(data_dir, "hs-trace"))]
    files.extend(hs_files)

    X, y = [], []
    for file in files: # each file
        with open(file, "r") as f:
            reader = csv.reader(f, delimiter=",")

            for trace in reader:
                feature, label = extract_features(trace)

                X.append(feature)
                y.append(label)   

    #print(f"X: {X}")
    #print(f"y: {y}")
    
    return X, y


# training
def train_cf(X_train, y_train):
    #logger.info(f"X_train: {X_train}")
    model = RandomForestClassifier(n_estimators=TREES, criterion=CRITERION)
    model.fit(X_train, y_train)

    return model

# test
def test_cf(model, X_test):
    y_pred = model.predict(X_test)

    return y_pred

def get_closeworld_score(y_true, y_pred):
    # accuracy
    accuracy = accuracy_score(y_true, y_pred)
    # precision      
    precision = precision_score(y_true, y_pred, average="macro")
    # recall
    recall = recall_score(y_true, y_pred, average="macro")
    # F-score
    f1 = 2*(precision*recall) / float(precision+recall)

    lines = []
    lines.append(f"accuracy: {accuracy}\n")
    lines.append(f"precision: {precision}\n")
    lines.append(f"recall: {recall}\n")
    lines.append(f"F1: {f1}\n")

    return lines

# get open-world score
def get_openworld_score(y_true, y_pred, label_unmon):
    # TP-correct, TP-incorrect, FN  TN, FN
    tp_c, tp_i, fn, tn, fp = 0, 0, 0, 0, 0

    logger.info(f"label_unmon: {label_unmon}")

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
    # recall or TPR
    recall = tp_c / float(tp_c+tp_i+fn)
    # F1-score
    f1 = 2*(precision*recall) / float(precision+recall)
    # FPR
    fpr = fp / float(fp+tn)

    lines = []
    lines.append(f"[POS] TP-c: {tp_c},  TP-i(incorrect class): {tp_i},  FN: {fn}\n")
    lines.append(f"[NEG] TN: {tn},  FP: {fp}\n\n")
    lines.append(f"accuracy: {accuracy}\n")
    lines.append(f"precision: {precision}\n")
    lines.append(f"recall: {recall}\n")
    lines.append(f"F1: {f1}\n")
    lines.append(f"TPR: {recall}\n")
    lines.append(f"FPR: {fpr}\n\n\n")

    return lines


# create logger
def get_logger():
    logging.basicConfig(format="[%(asctime)s]>>> %(message)s", level=LOG_LEVEL, datefmt = "%Y-%m-%d %H:%M:%S")
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    logger.info(f"X_train: {len(X_train)}, X_test: {len(X_test)}")
    logger.debug(f"y_test: {y_test}")

    # 1. training
    model= train_cf(X_train, y_train)
    logger.info(f"[TRAINED] random_forest model of cf.")

    # 2. test
    y_pred = test_cf(model, X_test)
    logger.info(f"[GOT] predicted labels of test samples.")
    logger.debug(f"y_pred: {y_pred}")
    
    # 3. get metrics value
    #lines = get_closeworld_score(y_test, y_pred)
    lines = get_openworld_score(y_test, y_pred, max(y))

    
    # save the results
    with open(result_path, "w") as f:
        f.writelines(lines)

    logger.info(f"{basename(__file__)}: complete successfully.\n")


if __name__ == "__main__":
    CURRENT_TIME = time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())

    BASE_DIR = abspath(join(dirname(__file__)))
    INPUT_DIR = join(BASE_DIR, "feature")
    OUTPUT_DIR = join(BASE_DIR, "result")

    try:
        logger = get_logger()
        args = parse_arguments()
        logger.info(f"{basename(__file__)} -> Arguments: {args}")

        feature_path = join(INPUT_DIR, args["in"])
        with open(feature_path, "rb") as f:
            X, y = pickle.load(f)
            logger.info(f"[LOADED] dataset&labels, length: {len(X)}, {len(y)}")

        if not exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            
        result_path = join(OUTPUT_DIR, CURRENT_TIME+"_"+args["out"]+".txt")

        main(X, y, result_path)


    except KeyboardInterrupt:
        sys.exit(-1)


