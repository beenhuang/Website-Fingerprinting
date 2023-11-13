#!/usr/bin/env python3

"""
<file>    decoy.py
<brief>   
"""

import os
import random
from os.path import join
from multiprocessing import Pool, cpu_count

from preprocess import Preprocess

class Decoy():
    def __init__(self, data_dir, out_dir):
        self.unmon_f = [x for x in os.listdir(data_dir) if '-' not in x]
        self.data_dir = data_dir
        self.out_dir = out_dir

    def defend(self):
        params = [[self.data_dir, self.out_dir, f, random.choice(self.unmon_f)] for f in os.listdir(self.data_dir)]
        with Pool(cpu_count()) as pool:
            pool.starmap(Decoy.simulate, params)

    @staticmethod
    def simulate(data_dir, out_dir, target_f, unmon_f):
        print(f'monitored file:{target_f}, unmonitored file:{unmon_f}', end='\r', flush=True)
       
        tgt_trace = Preprocess.standard_trace(data_dir, target_f, "\t") # preprocess the trace. 
        unmon_trace = Preprocess.standard_trace(data_dir, unmon_f, "\t")
        PAD_CELL = 777
        unmon_trace = [[x[0], x[1]*PAD_CELL] for x in unmon_trace]
        
        merge_trace = sorted(tgt_trace+unmon_trace, key=lambda x:x[0])

        with open(join(out_dir, target_f), 'w') as f:  # save the defended trace
            for e in merge_trace:
                f.write(str(e[0])+'\t'+str(e[1])+'\n')

if __name__ == "__main__":
    pass
