#!/usr/bin/env python3

"""
<file>    main.py
<brief>   read trace files ...
"""

import argparse
import os
import sys
import logging
import multiprocessing as mp
from os.path import abspath, dirname, join, basename, pardir, exists, splitext
from tqdm import tqdm

# create logger and parse arugment
def logger_and_arguments():
    # get logger
    logging.basicConfig(format="[%(asctime)s]>>> %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(splitext(basename(__file__))[0])
    
    # parse arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--in", required=True, help="load trace data")
    parser.add_argument("-d", "--defense", required=True, help="defense algorithm")
    args = vars(parser.parse_args())

    return logger, args

# return [[time, direction] ... ]
# this function for Wang-20000 dataset
def preprocess(trace, packet_size=514):
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

# run the simulation
def simulate(data_dir, defended_dir, file):
    print(f"Processing the file: {file}")
    # read a trace file.
    file_path = join(data_dir, file)
    with open(file_path, "r") as f:
        trace = f.readlines() 

    # preprocess the trace.    
    good_trace = preprocess(trace)  

    # get the defended trace
    defended_trace = defend(good_trace)

    # save the defended trace
    defended_file_path = join(defended_dir, file)
    with open(defended_file_path, "w") as f:
        for e in defended_trace:
            f.write(str(e[0])+"\t"+str(e[1])+"\n")

# run simulation one by one.
def main2(data_dir, defended_dir):
    flist = os.listdir(data_dir)

    #for file in tqdm(flist, total=len(flist)):
    for file in flist:    
        simulate(data_dir, defended_dir, file)
  
    logger.info(f"Complete")  

# use multiprocessing
def main(data_dir, defended_dir):
    flist = os.listdir(data_dir)
    params = [[data_dir, defended_dir, file] for file in flist]

    with mp.Pool(mp.cpu_count()) as pool:
        pool.starmap(simulate, params)

    logger.info(f"Complete")     

if __name__ == "__main__":
    BASE_DIR = abspath(join(dirname(__file__), pardir, pardir))
    MAIN_DATA_DIR = join(BASE_DIR, "data")
    MAIN_DEFENDED_DIR = join(BASE_DIR, "data", "DEFENSE")

    try:
        logger, args = logger_and_arguments()
        logger.info(f"{basename(__file__)} -> Arguments: {args}")

        data_dir = join(MAIN_DATA_DIR, args["in"])
            
        defended_dir = join(MAIN_DATA_DIR, args["defense"], args["in"])
        if not exists(defended_dir):
            os.makedirs(defended_dir)

        main2(data_dir, defended_dir)    

    except KeyboardInterrupt:
        sys.exit(-1)    

