#!/usr/bin/env python3

"""
<file>    convert.py
<brief>   it converts data file to trace file
"""

import argparse
import os
import sys
import logging
import multiprocessing as mp
from os.path import abspath, dirname, join, basename, pardir, exists, splitext

#from dataset.goodenough import standard_trace, get_file_list
from dataset.wtfpad_protected import standard_trace, get_file_list


# create logger
def get_logger():
    logging.basicConfig(format="[%(asctime)s]>>> %(message)s", level=logging.INFO, datefmt = "%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(splitext(basename(__file__))[0])
    
    return logger

# parse arugment
def parse_arguments():
    # create argument parser
    parser = argparse.ArgumentParser()

    # INPUT
    parser.add_argument("-i", "--in", required=True, help="load trace data")

    # parse arguments
    args = vars(parser.parse_args())

    return args


def convert(data_dir, trace_dir, data_file, trace_file):
    # 1. read a trace file.
    file_path = join(data_dir, data_file)
    with open(file_path, "r") as f:
        original_data = f.readlines() 

    # 2. convert to trace 
    trace = standard_trace(original_data) 

    # 3. save the trace to the trace_file
    trace_path = join(trace_dir, trace_file)
    with open(trace_path, "w") as f:
        for element in trace:
            f.write(f"{element}\n")

    print(f"trace file: {trace_file}")

# extract features one by one.
def main2(data_dir, trace_dir):
    flist = get_file_list(data_dir)

    for file in flist:
        # input file
        data_file = file[0]
        # output file
        trace_file = file[1]
        #print(f"{data_file}, {trace_file}")
        convert(data_dir, trace_dir, data_file, trace_file)

    logger.info(f"Complete")  



def main(data_dir, trace_dir):
    flist = get_file_list(data_dir)

    params = [[data_dir, trace_dir, f[0], f[1]] for f in flist]

    with mp.Pool(mp.cpu_count()) as pool:
        pool.starmap(convert, params)

    logger.info(f"Complete")     

if __name__ == "__main__":
    BASE_DIR = abspath(join(dirname(__file__), pardir))
    RAW_DATA_DIR = join(BASE_DIR, "raw-data")
    TRACE_DIR = join(BASE_DIR, "data")

    try:
        logger = get_logger()
        args = parse_arguments()
        logger.info(f"{basename(__file__)} -> Arguments: {args}")

        raw_data_dir = join(RAW_DATA_DIR, args["in"])
            
        trace_dir = join(TRACE_DIR, args["in"])
        if not exists(trace_dir):
            os.makedirs(trace_dir)

        main(raw_data_dir, trace_dir)    

    except KeyboardInterrupt:
        sys.exit(-1)    

