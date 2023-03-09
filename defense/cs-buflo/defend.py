#!/usr/bin/env python3

"""
<file>    defend.py
<brief>   convert original trace to defended trace using tamaraw.
"""

import argparse
import os
import csv
import sys
import logging
import pickle
import multiprocessing as mp
from os.path import abspath, dirname, join, basename, pardir, exists, splitext

from tqdm import tqdm

from csbuflo import CSBuFLO, normalise_timings, TimeTravel

# create logger
def get_logger():
    logging.basicConfig(format="[%(asctime)s]>>> %(message)s", level=logging.INFO, datefmt = "%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(splitext(basename(__file__))[0])
    
    return logger

# parse arugment
def parse_arguments():
    # create argument parser
    parser = argparse.ArgumentParser(description="cf")

    # INPUT
    parser.add_argument("-i", "--in", required=True, help="load trace data")

    # parse arguments
    args = vars(parser.parse_args())

    return args

# return [[time, direction] ... ]
def preprocess(trace):
    start_time = float(trace[0].split("\t")[0])
    #logger.debug(f"start_time: {start_time}")
    good_trace = []

    for e in trace:
        e = e.split("\t")
        #time = (float(e[0]) - start_time) * 1000.0  # To milliseconds
        time = float(e[0]) * 1000.0  # To milliseconds
        direction = int(e[1].strip("\n"))

        good_trace.append([time, direction])

    #logger.debug(f"good_trace: {good_trace}")

    return good_trace

def defend(data_dir, defended_dir, file):
    print(f"Processing the file: {file}")
    # read a trace file.
    file_path = join(data_dir, file)
    with open(file_path, "r") as f:
        trace = f.readlines() 

    # preprocess the trace.    
    good_trace = preprocess(trace)  

    success = False
    while not success:
        try:
            defended = CSBuFLO().defend(good_trace)
            success = True
        except TimeTravel:
            pass

    defended = normalise_timings(defended)        

    # save the tamaraw trace
    defended_f_path = join(defended_dir, file)
    with open(defended_f_path, "w") as f:
        for t, s in defended:
            t /= 1000.0  # To seconds
            s = 1 if s > 0 else -1
            f.write(repr(t) + "\t" + repr(s) + "\n")

# extract features one by one.
def main2(data_dir, defended_dir):
    flist = os.listdir(data_dir)

    for file in tqdm(flist, total=len(flist)):
        defend(data_dir, defended_dir, file)
  
    logger.info(f"Complete")  

# use multiprocessing
def main(data_dir, defended_dir):
    flist = os.listdir(data_dir)
    params = [[data_dir, defended_dir, f] for f in flist]

    with mp.Pool(mp.cpu_count()) as pool:
        pool.starmap(defend, params)

    logger.info(f"Complete")     

if __name__ == "__main__":
    BASE_DIR = abspath(join(dirname(__file__), pardir, pardir))
    MAIN_DATA_DIR = join(BASE_DIR, "data")
    MAIN_DEFENDED_DIR = join(BASE_DIR, "data", "cs-buflo")

    try:
        logger = get_logger()
        args = parse_arguments()
        logger.info(f"{basename(__file__)} -> Arguments: {args}")

        data_dir = join(MAIN_DATA_DIR, args["in"])

        if not exists(MAIN_DEFENDED_DIR):
            os.makedirs(MAIN_DEFENDED_DIR)
            
        defended_dir = join(MAIN_DEFENDED_DIR, args["in"])
        if not exists(defended_dir):
            os.makedirs(defended_dir)

        main(data_dir, defended_dir)    

    except KeyboardInterrupt:
        sys.exit(-1)    

