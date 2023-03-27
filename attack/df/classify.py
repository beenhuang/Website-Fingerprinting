#!/usr/bin/env python3

"""
<file>    classify.py
<brief>   classify the dataset using kfp_classifier.py model.
"""

import argparse
import os
import sys
import time
import pickle
import logging
from os.path import join, basename, abspath, splitext, dirname, pardir, isdir, exists
import numpy as np
from sklearn.model_selection import train_test_split
from keras.optimizers import Adamax
from keras import backend as K
from keras.utils import to_categorical

from dfnet import DFNet
from metrics import df_openworld

# Training the DF model
NB_EPOCH = 30   # Number of training epoch
BATCH_SIZE = 128 # Batch size
OPTIMIZER = Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) # Optimizer

# number of outputs = number of classes
NB_CLASSES = 101 
INPUT_SHAPE = (5000,1)


# create logger
def get_logger():
    logging.basicConfig(format="[%(asctime)s]>>> %(message)s", level=logging.INFO, datefmt = "%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(splitext(basename(__file__))[0])
    
    return logger

# parse arugment
def parse_arguments():
    # argument parser
    parser = argparse.ArgumentParser(description="kFP")

    # INPUT
    parser.add_argument("-i", "--in", required=True, help="load feature.pkl file")
    # OUTPUT
    parser.add_argument("-o", "--out", required=True, help="save results")

    args = vars(parser.parse_args())

    return args

def train_valid_test_split(X, y, num_class=101):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=247, stratify=y)
    
    X_train = np.array(X_train).astype('float32')
    X_test = np.array(X_test).astype('float32')
    y_train = np.array(y_train).astype('float32')
    y_test = np.array(y_test).astype('float32')

    X_train = X_train[:, :,np.newaxis]
    X_test = X_test[:, :,np.newaxis]

    y_train = to_categorical(y_train, num_class)
    y_test = to_categorical(y_test, num_class)

    logger.info(f"X_train:{X_train.shape}, y_train:{y_train.shape}")
    logger.info(f"X_test:{X_test.shape}, y_test:{y_test.shape}")

    return X_train, X_test, y_train, y_test


# [MAIN] function
def main(X, y, result_path):
    # split dataset 
    X_train, X_test, y_train, y_test = train_valid_test_split(X, y)

    # create DF model
    model = DFNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
    model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])

    # training
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, validation_split=0.1, verbose=2)

    # test
    y_pred = model.predict(X_test)
    #logger.info(f"y_pred:{y_pred[:,1]}, {y_pred.shape}")
    
    # get metrics value
    lines = df_openworld(y_test, y_pred, 100)
    #logger.info(f"[CALCULATED] metrics.")
    
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




