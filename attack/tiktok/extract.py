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
import multiprocessing as mp
from os.path import abspath, dirname, join, pardir, splitext, basename, exists

from tiktok_feature import tiktok_feature

# logger and arguments
def logger_and_arguments():
    # get logger
    logging.basicConfig(format="[%(asctime)s]>>> %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(splitext(basename(__file__))[0])
    
    # parse arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--in", required=True, help="load trace data")
    parser.add_argument("-a", "--attack", required=True, help="WF attack")
    args = vars(parser.parse_args())

    return logger, args

# trace format: [{timesatmp:float \t direction:int \n}, ...]
# preprocess trace for Wang-20000 dataset
# return format: [[timestamp, direction] ... ]
def preprocess(trace, packet_size=514):
    start_time = float(trace[0].split("\t")[0])
    #logger.info(f"start_time: {start_time}")
    
    good_trace = []
    for e in trace:
        e = e.split("\t")
        time = float(e[0]) - start_time
        direction = int(e[1].strip("\n"))
        good_trace.append([time, direction])

    #logger.debug(f"good_trace: {good_trace}")

    return good_trace

def extract(data_dir, feature_dir, file, umon_label=100):
    print(f"extracting: {file}")
    # read a trace file.
    file_path = join(data_dir, file)
    with open(file_path, "r") as f:
        trace = f.readlines() 

    # 1. get the feature vector for a trace
    good_trace = preprocess(trace)  
    feature = tiktok_feature(good_trace)

    # 2. get the label for a trace
    file_name = file.split(".")[0]    
    if "-" in file_name:
        label = int(file_name.split("-")[0])
    else:
        label = umon_label

    # save the feature from the trace
    feature_path = join(feature_dir, file)
    with open(feature_path, "w") as f:
        for e in feature:
            f.write(f"{e}\n")

    return (feature, label)   

# extract features one by one.
def main2(data_dir, feature_dir, feature_pickle="feature.pkl"):
    flist = os.listdir(data_dir)

    X, y = [], []
    #for file in tqdm(flist, total=len(flist)):
    for file in flist:
        feature, label = extract(data_dir, feature_dir, file)

        X.append(feature)
        y.append(label)

    with open(join(feature_dir, feature_pickle), "wb") as f:
        pickle.dump((X, y), f)    
    
    logger.info(f"Complete")  

# use multiprocessing
def main(data_dir, feature_dir, feature_pickle="feature.pkl"):
    flist = os.listdir(data_dir)
    params = [[data_dir, feature_dir, file] for file in flist]

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
    MAIN_ATTACK_DIR = join(BASE_DIR, "attack")

    try:
        logger, args = logger_and_arguments()
        logger.info(f"{basename(__file__)} -> Arguments: {args}")

        data_dir = join(MAIN_DATA_DIR, args["in"])   

        feature_dir = join(MAIN_ATTACK_DIR, args["attack"], "feature", args["in"])
        if not exists(feature_dir):
            os.makedirs(feature_dir)

        main(data_dir, feature_dir)    

    except KeyboardInterrupt:
        sys.exit(-1) 
