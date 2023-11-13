#!/usr/bin/env python3

"""
<file>    main.py
<brief>   main module
"""

import os
import logging
import argparse
from os.path import abspath, dirname, join, pardir, exists

from decoy import Decoy

# main function
def main(data_dir, out_dir):
    decoy = Decoy(data_dir, out_dir)
    decoy.defend()

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

    logger.info(f"WF Defense Complete")
