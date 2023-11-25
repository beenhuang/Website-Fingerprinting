#!/usr/bin/env python3

"""
<file>    decoy.py
<brief>   
"""

import os
import random

from preprocess import Preprocess

class Decoy():
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.unmon_f = [x for x in os.listdir(data_dir) if '-' not in x]

    def defend(self, tgt_trace):
        unmon_trace = Preprocess.standard_trace(self.data_dir, random.choice(self.unmon_f), "\t")
        #print(len(unmon_trace))
        PAD_CELL = 777
        unmon_trace = [[x[0], x[1]*PAD_CELL] for x in unmon_trace]
        
        merge_trace = sorted(tgt_trace+unmon_trace, key=lambda x:x[0])

        return merge_trace

if __name__ == "__main__":
    from os.path import join

    data_dir = '/Users/huangbin/desktop/WF/script/data/Wang-20000'
    out_dir = '/Users/huangbin/desktop/decoy'
    file = '0-0.cell'

    std_trace = Preprocess.bigenough(data_dir, file) # preprocess the trace. 
    print(f"std_trace:{len(std_trace)}")
    decoy = Decoy(data_dir)
    defend_trace = decoy.defend(std_trace) # get the defended trace

    with open(join(out_dir, file), 'w') as f:  # save the defended trace
        for e in defend_trace:
            f.write(str(e[0])+'\t'+str(e[1])+'\n')  
