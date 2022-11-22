#!/usr/bin/env python3

"""
<file>    k-FP.py
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from exfeature import extract_features


# CONSTANTS
MODULE_NAME = basename(__file__)
CURRENT_TIME = time.strftime("%Y.%m.%d-%H:%M:%S_", time.localtime())

BASE_DIR = abspath(join(dirname(__file__), pardir, pardir))
INPUT_DIR = join(BASE_DIR, "simulation", "sim-traces")
OUTPUT_DIR = join(BASE_DIR, "results")

# k-FP's hyperparameters: 
NUM_TREES = 1000
K = 6

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

        traces.append(trace)
    
    with mp.Pool(mp.cpu_count()) as pool:
        features = pool.map(extract_features, traces)

    return features, labels


def train_kfp(X_train, y_train):
    model = RandomForestClassifier(n_jobs=-1, n_estimators=NUM_TREES, oob_score=True)
    model.fit(X_train, y_train)

    fingerprints = model.apply(X_train)
    labeled_fps = [[fingerprints[i], y_train[i]] for i in range(len(fingerprints))]

    return model, labeled_fps


def test_kfp(model, labeled_fps, X_test):
    #
    pred_labels = []

    new_fps = model.apply(X_test)

    todo = [[labeled_fps, new_fp] for new_fp in new_fps]

    with mp.Pool(mp.cpu_count()) as pool:
        pred_labels = pool.starmap(knn_classifier, todo)

    return pred_labels

# KNN: k is 6 by default.
def knn_classifier(labeled_fps, new_fp, k=K):
    new_fp = np.array(new_fp, dtype=np.int32)
        
    hamming_dists=[]

    for elem in labeled_fps:
        labeled_fp = np.array(elem[0], dtype=np.int32)
        pred_label = elem[1]

        hamming_distance = np.sum(new_fp != labeled_fp) / float(labeled_fp.size)

        if hamming_distance == 1.0:
                 continue

        hamming_dists.append((hamming_distance, pred_label))


    by_distance = sorted(hamming_dists)
    k_nearest_labels = [p[1] for p in by_distance[:k]]
    majority_vote = max(set(k_nearest_labels), key=k_nearest_labels.count)


    return majority_vote


# [FUNC] get metrics values
def get_metrics(y_true, y_pred, label_unmon):
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


# [MAIN] function
def main():
    logger = get_logger()
    
    logger.info(f"{MODULE_NAME}: start to run.")

    # parse arguments
    args = parse_arguments()
    logger.info(f"Arguments: {args}")

    # load dataset&labels
    data_file = join(INPUT_DIR, args["in"])
    X, y = generate_feature_vectors(data_file)
    logger.info(f"[LOADED] dataset&labels, length: {len(X)}")

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=247, stratify=y)

    # training
    model, labeled_fps = train_kfp(X_train, y_train)
    logger.info(f"[TRAINED] Random_Forest model of k-FP.")

    # test
    y_pred = test_kfp(model, labeled_fps, X_test)
    logger.info(f"[GOT] predicted labels of test samples.")
    
    # get metrics value
    lines = get_metrics(y_test, y_pred, max(y_test))
    logger.info(f"[CALCULATED] metrics.")
    
    with open(join(OUTPUT_DIR, CURRENT_TIME+args["out"]+".txt"), "w") as f:
        f.writelines(lines)
        logger.info(f"[SAVED] results in the {args['out']}.")


    logger.info(f"{MODULE_NAME}: complete successfully.\n")


if __name__ == "__main__":
    sys.exit(main())

