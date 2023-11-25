#!/usr/bin/env python3

"""
<file>    front.py
<brief>   
"""

import itertools
import random
import numpy as np

OUT = 1
IN = -1
PAD = 777
IPT = 0.0001 # inter-packet time

class DFD():
    def __init__(self):
        # pert: the perturbation rate to be applied
        self.pert_rate = 0.50
        # variation_ratio: the variation parameter, affect the perturbation rate at the begining of each burst, this will only effect the white-box attack accuracy. (0 < variation_ratio < 1.0)
        self.var_rto = 0.5

    def defend(self, trace):
        return self.__dfd_paper(trace)

    def __dfd_code(self, trace):
        n_out, n_in, last_pkt = 0, 0, [0.0, 0]
        def_trace = []
        for pkt in trace:
            def_trace.append(pkt)
            cur_pert = random.uniform(self.pert_rate*self.var_rto, self.pert_rate*(1+self.var_rto))
            
            if pkt[1] == OUT: # outgoing packet
                n_out += 1
                
                if last_pkt[1] == IN:
                    cur_time = pkt[0]
                    for _ in range(int(1+cur_pert*n_in)):
                        cur_time += IPT
                        def_trace.append([cur_time, IN * PAD])
                    n_in = 0
            
            elif pkt[1] == IN: # incoming packet
                n_in += 1
                
                if last_pkt[1] == OUT:
                    cur_time = pkt[0]
                    for _ in range(int(1+cur_pert*n_out)):
                        cur_time += IPT
                        def_trace.append([cur_time, OUT * PAD])
                    n_out = 0

            last_pkt = pkt        

        return sorted(def_trace, key=lambda x:x[0])              

    def __dfd_paper(self, trace):
        n_out, n_in, prev_out, prev_in = 0, 0, 0, 0

        def_trace = []
        for pkt in trace:
            def_trace.append(pkt)

            if pkt[1] == OUT:
                n_out += 1

                if n_in != 0:
                    prev_in = n_in
                    n_in = 0

                if n_out == 2:
                    cur_time = pkt[0]
                    for _ in range(int(prev_out/2)):
                        cur_time += IPT
                        def_trace.append([cur_time, OUT * PAD])

            elif pkt[1] == IN:
                n_in += 1

                if n_out != 0:
                    prev_out = n_out
                    n_out = 0 

                if n_in == 2:
                    cur_time = pkt[0]
                    for _ in range(int(prev_in/2)):
                        cur_time += IPT
                        def_trace.append([cur_time, IN * PAD])

        return sorted(def_trace, key=lambda x:x[0])

if __name__ == '__main__':
    from os.path import join
    from preprocess import Preprocess

    data_dir = '/Users/huangbin/desktop/WF/script/data/Wang-20000'
    out_dir = '/Users/huangbin/desktop/dfd'
    file = '0-0.cell'

    std_trace = Preprocess.bigenough(data_dir, file) # preprocess the trace. 
    #print(f"std_trace:{len(std_trace)}")
    dfd = DFD()
    defend_trace = dfd.defend(std_trace) # get the defended trace

    with open(join(out_dir, file), 'w') as f:  # save the defended trace
        for e in defend_trace:
            f.write(str(e[0])+'\t'+str(e[1])+'\n')  
