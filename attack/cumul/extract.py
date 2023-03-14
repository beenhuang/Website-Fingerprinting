#!/usr/bin/env python3

"""
<file>    extract.py
<brief>   extract the CUMUL feature
"""

import os
import sys
import pickle
import argparse
import logging
import numpy as np
import multiprocessing as mp
from os.path import abspath, dirname, join, pardir, splitext, basename, exists

from cumul_feature import cumul_feature

PACKET_SIZE = 514
#PACKET_SIZE = 1

# create logger
def get_logger():
    logging.basicConfig(format="[%(asctime)s]>>> %(message)s", level=logging.INFO, datefmt = "%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(splitext(basename(__file__))[0])
    
    return logger

# parse arugment
def parse_arguments():
    # argument parser
    parser = argparse.ArgumentParser(description="CUMUL")

    # INPUT
    parser.add_argument("-i", "--in", required=True, help="load trace data")

    args = vars(parser.parse_args())

    return args

def preprocess(trace):
    good_trace = []

    for e in trace:
        # in cumul, positive is incoming when positive of Wang's Data is outgoing
        size = int(e.split("\t")[1].strip("\n")) * PACKET_SIZE * -1

        good_trace.append(size)

    #print(f"good_trace: {good_trace}")

    return good_trace

def extract(data_dir, feature_dir, file, umon_label=100):
    # read a trace file.
    file_path = join(data_dir, file)
    with open(file_path, "r") as f:
        trace = f.readlines() 

    # preprocess the trace.    
    good_trace = preprocess(trace)  
    #times, sizes = preprocess2(trace)

    # get feature vector
    feature = cumul_feature(good_trace)

    # get the trace file name
    feature_fname = file.split(".")[0]
    print(f"feature_fname: {feature_fname}")
    
    # get label
    if "-" in feature_fname:
        label = int(feature_fname.split("-")[0])
    else:
        label = umon_label

    
    # save the feature from the trace
    feature_path = join(feature_dir, feature_fname)
    with open(feature_path, "w") as f:
        for element in feature:
            f.write(f"{element}\n")

    return (feature, label)   

# extract features one by one.
def main2(data_dir, feature_dir, feature_pickle="feature.pkl"):
    flist = os.listdir(data_dir)

    X, y = [], []
    for file in tqdm(flist, total=len(flist)):
        feature, label = extract(data_dir, feature_dir, file)

        X.append(feature)
        y.append(label)

    with open(join(feature_dir, feature_pickle), "wb") as f:
        pickle.dump((X, y), f)    
    
    logger.info(f"Complete")  

# use multiprocessing
def main(data_dir, feature_dir, feature_pickle="feature.pkl"):
    flist = os.listdir(data_dir)
    params = [[data_dir, feature_dir, f] for f in flist]

    with mp.Pool(mp.cpu_count()) as pool:
        result = pool.starmap(extract, params)

    X, y = zip(*result) 

    with open(join(feature_dir, feature_pickle), "wb") as f:
        pickle.dump((X, y), f)    
    
    #logger.debug(f"X: {X}")
    #logger.debug(f"y: {y}")
    
    logger.info(f"Complete")     



if __name__ == "__main__":
    MODULE_NAME = basename(__file__)

    BASE_DIR = abspath(join(dirname(__file__), pardir, pardir))
    IN_DATA_DIR = join(BASE_DIR, "data")
    MAIN_FEATURE_DIR = join(BASE_DIR, "attack", "cumul", "feature")

    try:
        logger = get_logger()
        logger.info(f"{MODULE_NAME}: start to run.")

        # parse arguments
        args = parse_arguments()
        logger.info(f"Arguments: {args}")

        data = join(IN_DATA_DIR, args["in"])

        feature_dir = join(MAIN_FEATURE_DIR, args["in"])
        if not exists(feature_dir):
            os.makedirs(feature_dir)

        main(data, feature_dir)    


    except KeyboardInterrupt:
        sys.exit(-1) 
