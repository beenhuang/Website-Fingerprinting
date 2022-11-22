#!/usr/bin/env python3

"""
<file>    k-FP.py
<brief>   brief of thie file

"""
import argparse
import os
import sys
import logging
from os.path import join, basename, abspath, splitext, dirname, pardir, isdir
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, recall_score
import numpy as np

from feature import extract_features


# constants
MODULE_NAME = basename(__file__)

BASE_DIR = abspath(dirname(__file__))
MON_DIRNAME = "Alexa_Monitored"
UNMON_DIRNAME = "HS_Monitored"
# 55*100+2,930

NUM_TREES = 1000
KNN = 6


# [FUNC] parse arugment
def parse_arguments():
    # argument parser
    parser = argparse.ArgumentParser(description="k-FP")

    # 1. INPUT: load ds-*.pkl dataset
    parser.add_argument("-i", "--in", required=True, metavar="<trace file directory>", help="load trace data")
    # 2. OUTPUT: save overhead in the overhead-*.txt file
    #parser.add_argument("-o", "--out", required=True, metavar="<result-file>", help="save overhead to the text file.")

    args = vars(parser.parse_args())

    return args


def load_dataset(data_dir):
    # return values
    dataset, labels = [], []

    mon_dir = join(data_dir, MON_DIRNAME)
    unmon_dir = join(data_dir, UNMON_DIRNAME)

    # monitored data
    for file in os.listdir(mon_dir):
        dataset.append(extract_features(join(mon_dir, file)))
        labels.append(int(file.split("_")[0]))
    
    # unmonitored data
    unmon_label = max(labels)+1

    for file in os.listdir(unmon_dir):
        dataset.append(extract_features(join(unmon_dir, file)))
        labels.append(unmon_label)


    return dataset, labels    


def train_kfp(X_train, y_train):
    model = RandomForestClassifier(n_jobs=-1, n_estimators=NUM_TREES, oob_score=True)
    model.fit(X_train, y_train)

    train_fps = model.apply(X_train)
    kfp_fingerprints = [[train_fps[i], y_train[i]] for i in range(len(train_fps))]

    return model, kfp_fingerprints


def test_kfp(model, kfp_fingerprints, X_test, y_test):
    #
    pred_labels = []

    test_fps = model.apply(X_test)

    for fp in test_fps:
        test_fp = np.array(fp, dtype=np.int32)
        
        predictions=[]
        for elem in kfp_fingerprints:
            kfp_fp = np.array(elem[0], dtype=np.int32)
            pred_label = elem[1]

            hamming_distance = np.sum(test_fp != kfp_fp) / float(kfp_fp.size)

            if hamming_distance == 1.0:
                 continue

            predictions.append((hamming_distance, pred_label))

        top_labels = [p[1] for p in sorted(predictions)[:KNN]]
        pred_label = max(set(top_labels), key=top_labels.count)
        pred_labels.append(pred_label)


    #####  metrics  #######
    accuracy = accuracy_score(y_test, pred_labels)
    recall = recall_score(y_test, pred_labels, average='micro')

    return accuracy, recall


# [MAIN] function
def main():
    print(f"{MODULE_NAME}: start to run")

    # parse arguments
    args = parse_arguments()
    print(f"Arguments: {args}")

    # load dataset&labels
    data_dir = join(BASE_DIR, args["in"])
    dataset, labels = load_dataset(data_dir)
    print(f"[LOADED] dataset&labels, length: {len(dataset)}")

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.1, random_state=247, stratify=labels)

    # run k-fp classifier
    model, kfp_fingerprints = train_kfp(X_train, y_train)
    print(f"[TRAINED] model")

    accuracy, recall = test_kfp(model, kfp_fingerprints, X_test, y_test)
    print(f"accuracy: {accuracy}")
    print(f"recall: {recall}")

    print(f"{MODULE_NAME}: complete successfully")


if __name__ == "__main__":
    sys.exit(main())
