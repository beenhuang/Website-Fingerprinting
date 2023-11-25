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
from df_feature import get_feature

# only need to change the extract function.
def extract(d_dir, f_dir, file, label_unmon=0):
    print(f"extracting:{file}", end="\r", flush=True)

    std_trace = Preprocess.standard_trace(d_dir, file, "\t") # get a standard trace.
    if std_trace == -1: # standard trace is empty.
        return (-1, -1)

    # feature & label
    feat = get_feature(std_trace)
    label = label_unmon if '-' not in file else int(file.split('-')[0])+1
    
    with open(join(f_dir, file),'w') as f: # save feature in the file
        for e in feat:
            f.write(f"{e}\n")

    return (feat, label)   
    
# main function
def main(d_dir, f_dir):
    params = [(d_dir, f_dir, f) for f in os.listdir(d_dir)]
    with Pool(cpu_count()) as pool:
        res = pool.starmap(extract, params)

    new_res = [e for e in res if e[1] != -1] # remove empty elements.
    X, y = zip(*new_res) 

    return X, y    

# logger and arguments
def logger_and_arguments():
    logging.basicConfig(format="[%(asctime)s]>>> %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", required=True, help="dataset directory")
    parser.add_argument("--out", required=True, help="feature directory")
    args = vars(parser.parse_args())

    return logger, args


if __name__ == "__main__":
    BASE_DIR = abspath(join(dirname(__file__), pardir, pardir, pardir))
    logger, args = logger_and_arguments()
    logger.info(f"Arguments:{args}")

    d_dir = join(BASE_DIR, "data", args["in"])   
    f_dir = join(BASE_DIR, "attack", args["out"])
    if not exists(f_dir):
        os.makedirs(f_dir)

    X, y = main(d_dir, f_dir) # main function

    with open(join(f_dir, "feature.pkl"), "wb") as f:
        #pickle.dump((X, y), f)  
        pickle.dump((np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)), f)

    logger.info(f"Feature extraction completed!")    
