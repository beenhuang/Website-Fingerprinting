#!/usr/bin/env python3

"""
<file>    extract.py
<brief>   extract features
"""

import os
import sys
import pickle
import logging
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count
from os.path import abspath, dirname, pardir, exists, join, splitext

from preprocess import Preprocess
from wa_feature import get_feature

# only need to change the extract function.
def extract(d_dir, f_dir, file, label_unmon=0):
    print(f"extracting: {file}", end="\r", flush=True)

    # get a standard trace.
    std_trace = Preprocess.wang20000(d_dir, file) 
    if len(std_trace) == 0: # standard trace is empty.
        return (std_trace, -1)

    # feature & label
    f_vec = get_feature(std_trace)
    f_name, _ = splitext(file) 
    label = label_unmon if "-" not in f_name else int(f_name.split("-")[0])+1
    
    # save
    with open(join(f_dir, file),'w') as f:
        for e in f_vec:
            f.write(f"{e}\n")

    return (f_vec, label)   

# features are extracted one by one.
def main(d_dir, f_dir):
    X, y = [], []
    for file in os.listdir(d_dir):
        f_vec, label = extract(d_dir, f_dir, file)
        if len(f_vec) == 0: # feature is empty
            continue

        X.append(f_vec)
        y.append(label)

    return X, y
    
# multiprocessing
def main_mp(d_dir, f_dir):
    params = [(d_dir, f_dir, f) for f in os.listdir(d_dir)]

    with Pool(cpu_count()) as pool:
        res = pool.starmap(extract, params)

    # remove empty elements.
    new_res = [e for e in res if len(e[0]) != 0] 

    X, y = zip(*new_res) 

    return X, y    

# logger and arguments
def logger_and_arguments():
    # get logger
    logging.basicConfig(format="[%(asctime)s]>>> %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger()
    
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", required=True, help="WF attack")    
    parser.add_argument("--data_dir", required=True, help="dataset directory")
    args = vars(parser.parse_args())

    return logger, args


if __name__ == "__main__":
    BASE_DIR = abspath(join(dirname(__file__), pardir, pardir))
    logger, args = logger_and_arguments()
    logger.info(f"Arguments: {args}")

    d_dir = join(BASE_DIR, "data", args["data_dir"])   
    f_dir = join(BASE_DIR, "attack", args["attack"], "feature", args["data_dir"])
    if not exists(f_dir):
        os.makedirs(f_dir)

    X, y = main_mp(d_dir, f_dir) # main function

    with open(join(f_dir, "feature.pkl"), "wb") as f:
        #pickle.dump((X, y), f)  
        pickle.dump((np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)), f)

    logger.info(f"Feature extraction completed!")    

