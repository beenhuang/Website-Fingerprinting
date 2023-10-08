#!/usr/bin/env python3

"""
<file>    main.py
<brief>   main module
"""

import os
import logging
import argparse
from multiprocessing import Pool, cpu_count
from os.path import abspath, dirname, join, pardir, exists

from trafficsilver import defend

# run the simulation
def simulate(data_dir, out_dir, file):
    print(f"\t processing: {file}", end="\r", flush=True)
    defend(join(data_dir, file), out_dir)

# run simulation one by one.
def main(data_dir, out_dir):
    for file in os.listdir(data_dir):    
        simulate(data_dir, out_dir, file)

# use multiprocessing
def main_mp(data_dir, out_dir):
    params = [[data_dir, out_dir, file] for file in os.listdir(data_dir)]
    with Pool(cpu_count()) as pool:
        pool.starmap(simulate, params)

# create a logger and parse arguments
def logger_and_arguments():
    # get logger
    logging.basicConfig(format="[%(asctime)s]>>> %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger()
    
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", required=True, help="data directory.")
    parser.add_argument("--wfd", required=True, help="WF defense.")
    args = vars(parser.parse_args())

    return logger, args
         
if __name__ == "__main__":
    BASE_DIR = abspath(join(dirname(__file__), pardir, pardir))  
    logger, args = logger_and_arguments()
    logger.info(f"Arguments: {args}")

    data_dir = join(BASE_DIR, "data", args["in"])
        
    out_dir = join(BASE_DIR, "data", args["wfd"], args["in"])
    if not exists(out_dir):
        os.makedirs(out_dir)

    main_mp(data_dir, out_dir)    

    logger.info(f"Complete")
