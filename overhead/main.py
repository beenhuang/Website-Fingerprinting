#!/usr/bin/env python3

"""
<file>    main.py
<brief>   main module
"""

import os
import time
import logging
import argparse
from multiprocessing import Pool, cpu_count
from os.path import abspath, dirname, join, pardir, exists, splitext

from preprocess import Preprocess
from defense_overhead import defense_overhead

def get_overhead(res):
    undef_out_all = sum([x[0] for x in res]) 
    undef_in_all = sum([x[1] for x in res]) 
    undef_time_all = sum([x[2] for x in res]) 

    def_out_all = sum([x[3] for x in res]) 
    def_in_all = sum([x[4] for x in res]) 
    def_time_all = sum([x[5] for x in res]) 

    return defense_overhead(undef_out_all, undef_in_all, undef_time_all, def_out_all, def_in_all, def_time_all)  

def out_in_time_one_file(data_dir, file):
    std_trace = Preprocess.standard_trace(data_dir, file)
    direct_only = [x[1] for x in std_trace]
    
    p_out = len([x for x in direct_only if x > 0])
    p_in = len([x for x in direct_only if x < 0])
    total_time = std_trace[-1][0]-std_trace[0][0]

    return p_out, p_in, total_time

def out_in_time(undef_dir, def_dir, oh_dir, file):
    print(f"processing: {file}", end="\r", flush=True)

    undef_out, undef_in, undef_time = out_in_time_one_file(undef_dir, file)    
    def_out, def_in, def_time = out_in_time_one_file(def_dir, file)    
    
    f_name, _ = splitext(file) 
    mon_file = int(f_name.split("-")[0])+1 if "-" in f_name else None

    if mon_file != None:
        with open(join(oh_dir, str(mon_file)), 'a') as f:
            f.writelines([f'{f_name}\n', 
                            f'[original] out:{undef_out}, in:{undef_in}, total_time:{undef_time:.4f}\n',
                            f'[defend] out:{def_out}, in:{def_in}, total_time:{def_time:.4f}\n',
                            f'[BW overhead]:{(def_out+def_in-undef_out-undef_in)/float(undef_out+undef_in):.4f}\n',
                            f'[Time overhead]:{(def_time-undef_time)/float(undef_time):.4f}\n\n'])    

    return undef_out, undef_in, undef_time, def_out, def_in, def_time

# use multiprocessing
def main(undef_dir, def_dir, oh_dir):
    params = [[undef_dir, def_dir, oh_dir, f] for f in os.listdir(undef_dir) if exists(join(def_dir, f)) == True]
    with Pool(cpu_count()) as pool:
        res = pool.starmap(out_in_time, params)

    return get_overhead(res)

# create a logger and parse arguments
def logger_and_arguments():
    logging.basicConfig(format="[%(asctime)s]>>> %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    logger = logging.getLogger()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--undef_dir", required=True, help="undefended directory")
    parser.add_argument("--def_dir", required=True, help="defended directory")
    parser.add_argument("--out", required=True, help="output file")
    args = vars(parser.parse_args())

    return logger, args
         
if __name__ == "__main__":
    cur_time = time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())
    BASE_DIR = abspath(join(dirname(__file__), pardir)) 
    logger, args = logger_and_arguments()
    logger.info(f"Arguments:{args}")

    undef_dir = join(BASE_DIR, 'data', args['undef_dir'])
    def_dir = join(BASE_DIR, 'data', args['def_dir']) 

    out_dir = join(BASE_DIR, 'overhead', 'res')
    if not exists(out_dir):
        os.makedirs(out_dir)

    oh_dir = join(BASE_DIR, 'overhead', 'overhead', args['def_dir']) 
    if not exists(oh_dir):
        os.makedirs(oh_dir)
    
    lines = main(undef_dir, def_dir, oh_dir)   
    with open(join(out_dir, args['out']+'.txt'), "a") as f:
        f.write(f">>> {args['def_dir'].split('-')[-1]} experiment\n")
        f.writelines(lines)       

    logger.info(f"Completed!")

