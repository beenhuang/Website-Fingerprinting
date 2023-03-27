#!/usr/bin/env python3

"""
<file>    extract.py
<brief>   extract the DF feature using df_feature
"""

import os
import sys
import pickle
import itertools
import argparse
import logging
import numpy as np
import multiprocessing as mp
from os.path import abspath, dirname, join, pardir, splitext, basename, exists

from df_feature import df_feature

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

# return [str(time direction), ... ]
def preprocess(trace):
    start_time = float(trace[0].split("\t")[0])
    #logger.debug(f"start_time: {start_time}")
    good_trace = []

    for e in trace:
        e = e.split("\t")
        time = float(e[0]) - start_time
        direction = int(e[1].strip("\n"))

        good_trace.append([time, direction])

    #logger.debug(f"good_trace: {good_trace}")

    return good_trace

def extract(data_dir, feature_dir, file, umon_label=100):
    # read a trace file.
    file_path = join(data_dir, file)
    with open(file_path, "r") as f:
        trace = f.readlines() 

    # preprocess the trace.    
    good_trace = preprocess(trace)  

    # get feature vector
    feature = df_feature(good_trace)

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
    BASE_DIR = abspath(join(dirname(__file__), pardir, pardir))
    MAIN_DATA_DIR = join(BASE_DIR, "data")
    MAIN_FEATURE_DIR = join(BASE_DIR, "attack", "df", "feature")

    try:
        logger = get_logger()
        args = parse_arguments()
        logger.info(f"{basename(__file__)} -> Arguments: {args}")

        data_dir = join(MAIN_DATA_DIR, args["in"])     

        feature_dir = join(MAIN_FEATURE_DIR, args["in"])
        if not exists(feature_dir):
            os.makedirs(feature_dir)

        main(data_dir, feature_dir)    

    except KeyboardInterrupt:
        sys.exit(-1) 
