#!/usr/bin/env python3

"""
<file>    huang.py
<brief>   huang defense
"""

import random
import numpy as np
from scipy.stats import uniform, geom, bernoulli

OUT = 1
IN = -1

class HyWF():
    def __init__(self):
        self.n_path = 2
        self.n_cons = 20

    def defend(self, trace):
        out_trace = [x for x in trace if x[1] > 0]
        in_trace = [x for x in trace if x[1] < 0]

        c_trcs = self.__split_trace(out_trace)
        r_trcs = self.__split_trace(in_trace)

        '''
        print(len(out_trace))
        for x in c_trcs:
            print(len(x), end=", ")
        print()  
        print(len(in_trace))  
        for x in r_trcs:
            print(len(x), end=", ")
        print()  
        '''
        
        trcs = []
        for c_trc, r_trc in zip(c_trcs, r_trcs):
            trc = sorted(c_trc+r_trc, key=lambda x:x[0])
            trcs.append(trc)
        
        return trcs

    def __split_trace(self, trace):  
        p = uniform.rvs(size=1)[0]   

        n, c = 0, 0
        trcs = [[] for _ in range(self.n_path)]
        for pkt in trace:
            n += 1
            if n > c: # sample new path and batch
                c = geom.rvs(1/self.n_cons, size=1)[0]
                i = bernoulli.rvs(p, size=1)[0]
                n = 1
            
            trcs[i].append(pkt) # send packet

        return trcs         

if __name__ == "__main__":
    import os
    from os.path import join, exists
    from preprocess import Preprocess

    data_dir = '/Users/huangbin/desktop/WF/script/data/Wang-20000'
    out_dir = '/Users/huangbin/desktop/hywf'
    file = '0-0.cell'

    hy = HyWF()
    std_trace = Preprocess.bigenough(data_dir, file) # preprocess the trace. 
    print(f"std_trace:{len(std_trace)}")
    defend_trace = hy.defend(std_trace) # get the defended trace

    for trc in defend_trace:
        print(len(trc))

    for idx, trc in enumerate(defend_trace):             
        if len(trc) != 0:
            with open(join(out_dir, file+'-'+str(idx)), 'w') as f:  # save the defended trace
                for e in trc:
                    f.write(str(e[0])+'\t'+str(e[1])+'\n')  
