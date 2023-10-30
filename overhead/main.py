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

def outin_time_one_file(data_dir, file):
    std_trace = Preprocess.standard_trace(data_dir, file)
    direct_only = [x[1] for x in std_trace]
    
    p_out = len([x for x in direct_only if x > 0])
    p_in = len([x for x in direct_only if x < 0])
    total_time = std_trace[-1][0]-std_trace[0][0]

    return p_out, p_in, total_time

def outin_time(undef_dir, def_dir, oh_dir, file):
    print(f"processing: {file}", end="\r", flush=True)

    undef_out, undef_in, undef_time = outin_time_one_file(undef_dir, file)    
    def_out, def_in, def_time = outin_time_one_file(def_dir, file)    
    
    f_name, _ = splitext(file) 
    mon_file = int(f_name.split("-")[0])+1 if "-" in f_name else None

    if mon_file != None:
        with open(join(oh_dir, str(mon_file)), 'a') as f:
            f.writelines([f'{f_name}\n', 
                            f'[undefended] out:{undef_out}, in:{undef_in}, total_time:{undef_time:.4f}\n',
                            f'[defended] out:{def_out}, in:{def_in}, total_time:{def_time:.4f}\n',
                            f'[Bandwidth Overhead]:{(def_out+def_in-undef_out-undef_in)/float(undef_out+undef_in):.4f}\n',
                            f'[Time Overhead]:{(def_time-undef_time)/float(undef_time):.4f}\n\n'])    

    return undef_out, undef_in, undef_time, def_out, def_in, def_time

# use multiprocessing
def main_mp(undef_dir, def_dir, oh_dir):
    params = [[undef_dir, def_dir, oh_dir, f] for f in os.listdir(undef_dir)]
    with Pool(cpu_count()) as pool:
        res = pool.starmap(outin_time, params)

    return get_overhead(res)

# create a logger and parse arguments
def logger_and_arguments():
    # get logger
    logging.basicConfig(format="[%(asctime)s]>>> %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    logger = logging.getLogger()
    
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--undefended_dir", required=True, help="undefended directory")
    parser.add_argument("--WF_defense", required=True, help="WF defense")
    parser.add_argument("--out", required=True, help="output file")
    args = vars(parser.parse_args())

    return logger, args
         
if __name__ == "__main__":
    BASE_DIR = abspath(join(dirname(__file__), pardir))  
    cur_time = time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())
    logger, args = logger_and_arguments()
    logger.info(f"Arguments: {args}")

    out_dir = join(BASE_DIR, 'overhead', 'result')
    if not exists(out_dir):
        os.makedirs(out_dir)
    oh_dir = join(BASE_DIR, 'overhead', 'overhead', args['WF_defense'], args['undefended_dir'])
    if not exists(oh_dir):
        os.makedirs(oh_dir)

    undef_dir = join(BASE_DIR, 'data', args['undefended_dir'])
    def_dir = join(BASE_DIR, 'data', args['WF_defense'], args['undefended_dir'])
    
    lines = main_mp(undef_dir, def_dir, oh_dir)   
    with open(join(out_dir, cur_time+"_"+args['out']+'.txt'), "a") as f:
        f.writelines(lines)       

    logger.info(f"Complete")

