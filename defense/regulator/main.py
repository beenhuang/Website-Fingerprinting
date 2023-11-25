#!/usr/bin/env python3

"""
<file>    main.py
<brief>   main module
"""

import os
import logging
import argparse
from multiprocessing import Pool, cpu_count
from os.path import abspath, dirname, pardir, join, exists

from regulator import RegulaTor
from preprocess import Preprocess

# run the simulation
def simulate(data_dir, out_dir, file):
    print(f'processing:{file}', end='\r', flush=True)
       
    std_trace = Preprocess.bigenough(data_dir, file) # preprocess the trace. 
    regulator = RegulaTor()
    defend_trace = regulator.defend(std_trace) # get the defended trace

    with open(join(out_dir, file), 'w') as f:  # save the defended trace
        for e in defend_trace:
            f.write(str(e[0])+'\t'+str(e[1])+'\n')

# use multiprocessing
def main(data_dir, out_dir):
    params = [[data_dir, out_dir, file] for file in os.listdir(data_dir)]
    with Pool(cpu_count()) as pool:
        pool.starmap(simulate, params)

# create a logger and parse arguments
def logger_and_arguments():
    logging.basicConfig(format="[%(asctime)s]>>> %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", required=True, help="data directory")
    parser.add_argument("--out", required=True, help="defended data directory")
    args = vars(parser.parse_args())

    return logger, args
         
if __name__ == "__main__":
    BASE_DIR = abspath(join(dirname(__file__), pardir, pardir))  
    logger, args = logger_and_arguments()
    logger.info(f"Arguments:{args}")

    data_dir = join(BASE_DIR, 'data', args['in'])
    out_dir = join(BASE_DIR, 'data', args['out'])
    if not exists(out_dir):
        os.makedirs(out_dir)

    main(data_dir, out_dir)    

    logger.info(f"WF Defense Complete!")
