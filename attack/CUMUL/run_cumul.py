#!/usr/bin/env python3

"""
<file>    run_cumul.py
<brief>   classify the cumul features and get the result.
"""

import argparse
import os
import sys
import time
import pickle
import logging
import numpy as np
from os.path import join, basename, abspath, splitext, dirname, pardir, isdir, exists
import multiprocessing as mp

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import make_scorer

# recall score used by find_optimal_hyperparams() and get_cross_val_score()
def openworld_recall_score(y_true, y_pred, label_unmon):
    # TP-correct, TP-incorrect, FN  TN, FP
    tp_c, tp_i, fn, tn, fp = 0, 0, 0, 0, 0
    
    #logger.debug(f"label_unmon: {label_unmon}")

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


    # recall
    recall = tp_c / float(tp_c+tp_i+fn)

    return recall

# in order to find the optimal hyperparameters
# c: 2**11 ~ 2**17
# gamma: 2**-3 ~ 2**3
def find_optimal_hyperparams(X, y):
    #
    model = Pipeline([("standardscaler", StandardScaler()), ("svc", SVC(kernel="rbf"))])
    #hyperparams = {"svc__C":[2048, 4096, 8192, 16384, 32768, 65536, 131072], "svc__gamma":[0.125, 0.25, 0.5, 0, 2, 4, 8]}
    hyperparams = {"svc__C":[256, 526, 1024, 2048], "svc__gamma":[0.0078125, 0.015625, 0.03125, 0.0625]}
    
    clf = GridSearchCV(model, hyperparams, n_jobs=-1, cv=10, scoring=make_scorer(openworld_recall_score, label_unmon=max(y)))
    clf.fit(X, y)
    
    # write to txt file
    lines = []
    lines.append(f"best_params_: {clf.best_params_} \n")
    lines.append(f"best_score_: {clf.best_score_} ")

    return lines


def train_cumul(X_train, y_train):
    model = Pipeline([("standardscaler", StandardScaler()), ("svc", SVC(kernel="rbf", C=2048, gamma=0.015625))])
    model.fit(X_train, y_train)

    return model


def test_cumul(model, X_test):
    return model.predict(X_test)


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
    lines.append(f"[POS] TP-c: {tp_c},  TP-i: {tp_i},  FN: {fn}\n")
    lines.append(f"[NEG] TN: {tn},  FP: {fp}\n\n")
    lines.append(f"accuracy: {accuracy}\n")
    lines.append(f"precision: {precision}\n")
    lines.append(f"recall: {recall}\n")
    lines.append(f"F1: {f1}\n")
    lines.append(f"TPR: {recall}\n")
    lines.append(f"FPR: {fpr}\n\n\n")

    return lines


# [MAIN] function
def main(X, y, result_path):
    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=247, stratify=y)
    logger.info(f"X_train: {len(X_train)}, X_test: {len(X_test)}")

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

