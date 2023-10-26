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

from metrics import *

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

import numpy as np
from keras.optimizers import Adamax
from keras import backend as K
from keras.utils import to_categorical

from dfnet_tf import DFNet

# Training the DF model
EPOCH = 30   # Number of training epoch
BATCH_SIZE = 128 # Batch size
OPTIMIZER = Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# number of outputs = number of classes
NUM_CLASSES = 101 
INPUT_SHAPE = (5000,1)

def train_valid_test_split(X, y, num_class=NUM_CLASSES):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=247, stratify=y)
    
    X_train = np.array(X_train).astype('float32')
    X_test = np.array(X_test).astype('float32')
    X_train = X_train[:, :,np.newaxis]
    X_test = X_test[:, :,np.newaxis]
  
    y_train = np.array(y_train).astype('float32')
    y_test = np.array(y_test).astype('float32')
    y_train = to_categorical(y_train, num_class)
    y_test = to_categorical(y_test, num_class)

    logger.info(f"X_train:{X_train.shape}, y_train:{y_train.shape}")
    logger.info(f"X_test:{X_test.shape}, y_test:{y_test.shape}")

    return X_train, X_test, y_train, y_test

def main(X, y, result_path):
    # split dataset 
    X_train, X_test, y_train, y_test = train_valid_test_split(X, y)

    # create DF model
    model = DFNet.build(input_shape=INPUT_SHAPE, classes=NUM_CLASSES)
    model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])

    # training
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH, validation_split=0.1, verbose=2)
    # test
    y_pred_softmax = model.predict(X_test)

    # get metrics value
    y_true = [np.argmax(x) for x in y_test]
    y_pred = [np.argmax(x) for x in y_pred_softmax]
    lines = openworld_score(y_true, y_pred, max(y_true))

    # save the result
    with open(result_path, "a") as f:
        f.writelines(lines)
        logger.info(f"saved the result.")

    logger.info(f"Complete!")


if __name__ == "__main__":
    CURRENT_TIME = time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())
    BASE_DIR = abspath(join(dirname(__file__)))
    FEATURE_DIR = join(BASE_DIR, "feature")
    RESULT_DIR = join(BASE_DIR, "result")

    try:
        logger, args = logger_and_arguments()
        logger.info(f"Arguments: {args}")

        # load X, y
        feature_path = join(FEATURE_DIR, args["in"])
        with open(feature_path, "rb") as f:
            X, y = pickle.load(f)
            logger.info(f"load dataset:{len(X)}, labels:{len(y)}")

        if not exists(RESULT_DIR):
            os.makedirs(RESULT_DIR)
        result_path = join(RESULT_DIR, args["out"]+".txt")

        main(X, y, result_path)

    except KeyboardInterrupt:
        sys.exit(-1)




