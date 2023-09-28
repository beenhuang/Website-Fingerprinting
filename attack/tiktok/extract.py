#!/usr/bin/env python3

"""
<file>    extract.py
<brief>   extract feature
"""

import os
import sys
import pickle
import logging
import argparse
from multiprocessing import Pool, cpu_count
from os.path import abspath, dirname, pardir, exists, join, splitext

from preprocess import Preprocess
from tiktok_feature import tiktok_feature

UNMON_LABEL=100

# only need to change the extract function.
def extract(data_dir, feature_dir, file):
    print(f"extracting: {file}")
    # feature
    standard_trace = Preprocess.Wang20000(data_dir, file)  
    feature = tiktok_feature(standard_trace)
    
    # label
    f_name, _ = splitext(file)    
    label = UNMON_LABEL if "-" not in f_name else int(f_name.split("-")[0])
    
    # save
    with open(join(feature_dir,file), "w") as f:
        for e in feature:
            f.write(f"{e}\n")

    return (feature, label)   

### The following code does not need to be modified. ###
# logger and arguments
def logger_and_arguments():
    # get logger
    logging.basicConfig(format="[%(asctime)s]>>> %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger()
    
    # parse arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--in", required=True, help="load trace data")
    parser.add_argument("-a", "--attack", required=True, help="WF attack")
    args = vars(parser.parse_args())

    return logger, args

# features are extracted one by one.
def main(data_dir, feature_dir):
    X, y = [], []
    for file in os.listdir(data_dir):
        feature, label = extract(data_dir, feature_dir, file)
        X.append(feature)
        y.append(label)

    return X, y
    
# multiprocessing
def main_mp(data_dir, feature_dir):
    params = [(data_dir,feature_dir,file) for file in os.listdir(data_dir)]

    with Pool(cpu_count()) as pool:
        result = pool.starmap(extract, params)

    X, y = zip(*result) 

    return X, y    

if __name__ == "__main__":
    BASE_DIR = abspath(join(dirname(__file__), pardir, pardir))
    MAIN_DATA_DIR = join(BASE_DIR, "data")
    MAIN_ATTACK_DIR = join(BASE_DIR, "attack")

    logger, args = logger_and_arguments()
    logger.info(f"Arguments: {args}")

    data_dir = join(MAIN_DATA_DIR, args["in"])   
    feature_dir = join(MAIN_ATTACK_DIR, args["attack"], "feature", args["in"])
    if not exists(feature_dir):
        os.makedirs(feature_dir)

    X, y = main_mp(data_dir, feature_dir) 

    with open(join(feature_dir,"feature.pkl"), "wb") as f:
        pickle.dump((X, y), f)  

    logger.info(f"Complete")    

