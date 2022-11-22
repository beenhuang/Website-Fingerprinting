#!/usr/bin/env python3

"""
<file>    cumul.py
<brief>   brief of thie file
"""

import argparse
import os
import sys
import time
import pickle
import logging
import numpy as np
from os.path import join, basename, abspath, splitext, dirname, pardir, isdir
import multiprocessing as mp

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import make_scorer

from exfeature import extract_features


# CONSTANTS
MODULE_NAME = basename(__file__)
CURRENT_TIME = time.strftime("%Y.%m.%d-%H:%M:%S_", time.localtime())

BASE_DIR = abspath(join(dirname(__file__), pardir, pardir))
INPUT_DIR = join(BASE_DIR, "simulation", "sim-traces")
OUTPUT_DIR = join(BASE_DIR, "results")


# 
NONPADDING_SENT = 1.0
NONPADDING_RECV = -1.0
PADDING_SENT = 2.0
PADDING_RECV = -2.0

def get_logger():
    logging.basicConfig(format="[%(asctime)s] >> %(message)s", level=logging.INFO)
    logger = logging.getLogger(splitext(basename(__file__))[0])
    
    return logger


# [FUNC] parse arugment
def parse_arguments():
    # argument parser
    parser = argparse.ArgumentParser(description="k-FP")

    # 1. INPUT: load ds-*.pkl dataset
    parser.add_argument("-i", "--in", required=True, metavar="<trace file directory>", help="load trace data")
    # 2. OUTPUT: save overhead in the overhead-*.txt file
    parser.add_argument("-o", "--out", required=True, metavar="<result-file>", help="save results in the text file.")

    args = vars(parser.parse_args())

    return args


def generate_feature_vectors(data_file):
    with open(data_file, "rb") as f:
        dataset, labels = pickle.load(f)  

    traces = []
    for trace in dataset:
        trace[trace[:,1] == PADDING_SENT, 1] = NONPADDING_SENT
        trace[trace[:,1] == PADDING_RECV, 1] = NONPADDING_RECV

        traces.append((trace, 100))
    
    with mp.Pool(mp.cpu_count()) as pool:
        features = pool.starmap(extract_features, traces)

    return features, labels


def train_cumul(X_train, y_train):
    #clf = Pipeline([('standardscaler', StandardScaler()), ('svc', SVC(kernel='rbf', C=c, gamma=int(g)))])
    model = Pipeline([('standardscaler', StandardScaler()), ('svc', SVC(kernel='rbf'))])
    model.fit(X_train, y_train)

    return model


def test_cumul(model, X_test):
    return model.predict(X_test)


# get open-world score
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
    lines.append(f"[POS] TP-c: {tp_c},  TP-i(incorrect class): {tp_i},  FN: {fn}\n")
    lines.append(f"[NEG] TN: {tn},  FP: {fp}\n\n")
    lines.append(f"accuracy: {accuracy}\n")
    lines.append(f"precision: {precision}\n")
    lines.append(f"recall: {recall}\n")
    lines.append(f"F1: {f1}\n")
    return lines


# in order to find the optimal hyperparameters
# c: 2**11 ~ 2**17
# gamma: 2**-3 ~ 2**3
def find_optimal_hyperparams(X, y):
    #
    model = Pipeline([("standardscaler", StandardScaler()), ("svc", SVC(kernel="rbf"))])
    hyperparams = {"svc__C":[2048, 4096, 8192, 16384, 32768, 65536, 131072], "svc__gamma":[0.125, 0.25, 0.5, 0, 2, 4, 8]}
    
    clf = GridSearchCV(model, hyperparams, n_jobs=-1, cv=2, scoring=make_scorer(openworld_recall_score, label_unmon=max(y)))
    clf.fit(X, y)
    
    # write to txt file
    lines = []
    lines.append(f"best_params_: {clf.best_params_} \n")
    lines.append(f"best_score_: {clf.best_score_} ")

    return lines


#
def get_cross_val_score(X, y, cv=10):
    #
    model = Pipeline([("standardscaler", StandardScaler()), ("svc", SVC(kernel="rbf"))])
    scores = cross_val_score(model, X, y, cv=cv, n_jobs=-1, scoring=make_scorer(openworld_recall_score, label_unmon=max(y)))
    
    lines = []
    lines.append(f"recall: {scores}\n")
    lines.append(f"recall_mean: {scores.mean()}\n")

    return lines


# recall score used by find_optimal_hyperparams() and get_cross_val_score()
def openworld_recall_score(y_true, y_pred, label_unmon):
    print(f"label_unmon: {label_unmon}")
    # TP-correct, TP-incorrect, FN  TN, FP
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


    # recall
    recall = tp_c / float(tp_c+tp_i+fn)

    return recall


# [MAIN] function
def main():
    # 
    logger = get_logger()
    logger.info(f"{MODULE_NAME}: start to run.")

    # parse arguments
    args = parse_arguments()
    logger.info(f"Arguments: {args}")

    # load dataset&labels
    data_file = join(INPUT_DIR, args["in"])
    X, y = generate_feature_vectors(data_file)
    logger.info(f"[EXTRACTED] fatures, length: {len(X)}")

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=247, stratify=y)
    
    # find best hyperparameters
    lines = find_optimal_hyperparams(X, y)
    print(lines)
    sys.exit(0)
    

    # training
    model = train_cumul(X_train, y_train)
    logger.info(f"[TRAINED] cumul model.")

    # test
    y_pred = test_cumul(model, X_test)
    logger.info(f"[GOT] predicted labels of test samples.")
    
    # get the open-world metrics score
    lines = get_openworld_score(y_test, y_pred, max(y_test))
    logger.info(f"[CALCULATED] metrics.")
    
    with open(join(OUTPUT_DIR, CURRENT_TIME+args["out"]+".txt"), "w") as f:
        f.writelines(lines)
        logger.info(f"[SAVED] results in the {args['out']}.")


    logger.info(f"{MODULE_NAME}: complete successfully.\n")


if __name__ == "__main__":
    sys.exit(main())
