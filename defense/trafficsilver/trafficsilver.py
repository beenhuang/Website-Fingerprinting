#!/usr/bin/env python3

"""
<file>    huang.py
<brief>   huang defense
"""

import random
import numpy as np

OUT = 1
IN = -1

class TrafficSilver():
    def __init__(self):
        self.n_path = 5
        self.batch_range = [50, 70]

    def defend(self, trace):
        out_trace = [x for x in trace if x[1] > 0]
        in_trace = [x for x in trace if x[1] < 0]

        c_trcs = self.__batched_weighted_random(out_trace)
        r_trcs = self.__batched_weighted_random(in_trace)

        trcs = []
        for c_trc, r_trc in zip(c_trcs, r_trcs):
            trc = sorted(c_trc+r_trc, key=lambda x:x[0])
            trcs.append(trc)

        return trcs

    def __batched_weighted_random(self, trace):
        weights = np.random.dirichlet([1]*self.n_path, size=1)[0]

        n, batch = 0, 0 
        trcs = [[] for _ in range(self.n_path)] 
        for pkt in trace:
            n += 1
            if n > batch:
                path = np.random.choice(np.arange(0, self.n_path), p=weights)
                batch = random.randint(self.batch_range[0], self.batch_range[1])
                n = 1
            
            trcs[path].append(pkt) # send packet

        return trcs


    def __batched_random(self, trace):
        n, batch = 0, 0 
        trcs = [[] for _ in range(self.n_path)] 
        for pkt in trace:
            n += 1
            if n > batch:
                path = np.random.choice(np.arange(0, self.n_path))
                batch = random.randint(self.batch_range[0], self.batch_range[1])
                n = 1
            
            trcs[path].append(pkt) # send packet

        return trcs

if __name__ == "__main__":
    import os
    from os.path import join, exists
    from preprocess import Preprocess

    data_dir = '/Users/huangbin/desktop/WF/script/data/Wang-20000'
    out_dir = '/Users/huangbin/desktop/trafficsilver'
    file = '0-0.cell'
    
    ts = TrafficSilver()
    std_trace = Preprocess.bigenough(data_dir, file) # preprocess the trace. 
    print(f"std_trace:{len(std_trace)}")
    defend_trace = ts.defend(std_trace) # get the defended trace

    for trc in defend_trace:
        print(len(trc), end=", ")

    for idx, trc in enumerate(defend_trace):
        spl_dir = out_dir+'-'+str(idx)
        if not exists(spl_dir):
            os.makedirs(spl_dir)        
        
        if len(trc) != 0:
            with open(join(spl_dir, file), 'w') as f:  # save the defended trace
                for e in trc:
                    f.write(str(e[0])+'\t'+str(e[1])+'\n')  
