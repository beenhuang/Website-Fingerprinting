#!/usr/bin/env python3

"""
<file>    extract.py
<brief>   extract the Tik-Tok feature
"""

import os
import sys
import pickle
import itertools
import argparse
import logging
import numpy as np
from multiprocessing import Pool, cpu_count
from os.path import abspath, dirname, join, pardir, splitext, basename, exists
from tqdm import tqdm
from collections import Counter

from df_feature import get_feature

def preprocess(trace):
    NONPADDING_SENT = 1
    NONPADDING_RECV = -1
    PADDING_SENT = 2
    PADDING_RECV = -2

    start_time = float(trace[0][0])
    good_trace = []

    for e in trace:
        time = round((float(e[0]) - start_time) * 0.000000001, 5)
        direct = int(e[1])
        direct = NONPADDING_SENT if direct == PADDING_SENT else direct
        direct = NONPADDING_RECV if direct == PADDING_RECV else direct

        good_trace.append([time, direct])

    return good_trace

def extract(trace, label):
    print(f"extracting: {label}", end="\r", flush=True)
    std_trace = preprocess(trace)  
    f_vec = get_feature(std_trace)

    return np.array(f_vec, dtype=np.float32), label

# extract features one by one.
def main(data, label):
    X, y = [], []
    for e in zip(data, label):
        f_vec, label = extract(e)

        X.append(f_vec)
        y.append(label)

    return X, y    

# multiprocessing
def main_mp(dl):
    with Pool(cpu_count()) as pool:
        res = pool.starmap(extract, dl)

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
    f_dir = join(BASE_DIR, "attack", args["attack"], "feature", args["data_dir"], "20k")
    if not exists(f_dir):
        os.makedirs(f_dir)

    with open(join(d_dir, "10-10-standard-october-5000.pkl"), "rb") as f:
        data, label = pickle.load(f)
        label = [0 if x == 50 else x+1 for x in label]
        dl = list(zip(data, label))
        #new_dl = [x for x in dl if x[1]==11 or x[1]==10 or x[1]==18 or x[1]==21 or x[1]==22 or x[1]==0]
        new_dl = [x for x in dl if x[1]==1 or x[1]==2 or x[1]==3 or x[1]==4 or x[1]==5 or x[1]==0]
    X, y = main_mp(new_dl) 

    # unmon-data, max is 399989
    unmon_size = 1
    with open('/home/shadow/WF/data/october/awf/oct-awf.pkl', 'rb') as f:
        data_unmon = pickle.load(f)
        #batch_unmon = data_unmon
        batch_unmon = data_unmon[np.random.choice(data_unmon.shape[0], size=unmon_size, replace=False), :]
        print(f"unmon_size:{batch_unmon.shape}")

    data = np.array(X, dtype=np.float32)
    label = np.array(y, dtype=np.int16)

    data_all=np.concatenate((data, batch_unmon),axis=0)
    label = np.append(label, np.zeros(unmon_size, dtype=np.int16))
    print(Counter(label))
    print(data_all.shape)
    print(label.shape)

    with open(join(f_dir, "feature.pkl"), "wb") as f:
        pickle.dump((data_all, label), f)  

    logger.info(f"Complete!")   


