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
from var_feature import var_feature

# only need to change the extract function.
def extract(data_dir, feature_dir, file, unmon_label):
    print(f"\t extracting: {file}", end="\r", flush=True)
    
    # feature
    standard_trace = Preprocess.wang20000(data_dir, file)  
    feature = var_feature(standard_trace)
    
    # label
    f_name, _ = splitext(file)    
    label = unmon_label if "-" not in f_name else int(f_name.split("-")[0])
    
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
    parser.add_argument("--wf", required=True, help="WF attack")    
    parser.add_argument("--data_dir", required=True, help="dataset directory")
    parser.add_argument("--unmon_label", required=True, help="unmonitored label")
    args = vars(parser.parse_args())

    return logger, args

# features are extracted one by one.
def main(data_dir, feature_dir, unmon_label):
    X, y = [], []
    for file in os.listdir(data_dir):
        feature, label = extract(data_dir, feature_dir, file, unmon_label)
        X.append(feature)
        y.append(label)

    return X, y
    
# multiprocessing
def main_mp(data_dir, feature_dir, unmon_label):
    params = [(data_dir,feature_dir,file,unmon_label) for file in os.listdir(data_dir)]

    with Pool(cpu_count()) as pool:
        result = pool.starmap(extract, params)

    X, y = zip(*result) 

    return X, y    

if __name__ == "__main__":
    BASE_DIR = abspath(join(dirname(__file__), pardir, pardir))

    logger, args = logger_and_arguments()
    logger.info(f"Arguments: {args}")

    data_dir = join(BASE_DIR, "data", args["data_dir"])   
    feature_dir = join(BASE_DIR, "attack", args["wf"], "feature", args["data_dir"])
    if not exists(feature_dir):
        os.makedirs(feature_dir)

    # main function
    X, y = main_mp(data_dir, feature_dir, int(args["unmon_label"])) 

    with open(join(feature_dir,"feature.pkl"), "wb") as f:
        pickle.dump((X, y), f)  

    logger.info(f"Complete")    

