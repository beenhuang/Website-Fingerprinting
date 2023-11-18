#!/usr/bin/env python3

"""
<file>    huang.py
<brief>   huang defense
"""

import random
import numpy as np
from scipy.stats import geom, uniform, lognorm, weibull_min, genpareto, rayleigh

OUT = 1
IN = -1

class SWP():
    def __init__(self):
        self.path_range = [2, 3]
        self.batch_range = [50, 70]

    def defend(self, trace):
        spl_traces = self.__splitting(trace)

        def_traces = []
        for trc in spl_traces:
            pad_trc = self.__padding(trc)
            def_traces.append(pad_trc)

        return def_traces  

    def __splitting(self, trace):
        pkt_path = self.__batched_weighted_random(trace, self.path_range, self.batch_range)
        #print(f"pkt_path:{len(pkt_path)}")
        spl_traces = self.__split_trace(trace, pkt_path)
        
        return spl_traces

    def __batched_weighted_random(self, trace, path_range, batch_range):
        n_path = np.random.choice(path_range)
        w_c = np.random.dirichlet([1]*n_path, size=1)[0]
        w_r = np.random.dirichlet([1]*n_path, size=1)[0]

        pkt_in, pkt_out = 0, 0
        last_c_path = np.random.choice(np.arange(0, n_path), p=w_c)
        c_batch = random.randint(batch_range[0], batch_range[1]) 
        last_r_path = np.random.choice(np.arange(0, n_path), p=w_r)
        r_batch = random.randint(batch_range[0], batch_range[1]) 

        pkt_path = []    
        for pkt in trace:
            if (pkt[1] == OUT): # outgoing packet
                pkt_out += 1
                pkt_path.append(last_c_path) # add path

                if (pkt_out == c_batch): # resample path & batch
                    pkt_out = 0
                    last_c_path = np.random.choice(np.arange(0, n_path), p=w_c)
                    c_batch = random.randint(batch_range[0], batch_range[1])
            elif (pkt[1] == IN): # incoming packet
                pkt_in += 1
                pkt_path.append(last_r_path) # add path
                
                if (pkt_in == r_batch): # resample path & batch
                    pkt_in = 0
                    last_r_path = np.random.choice(np.arange(0, n_path), p=w_r)
                    r_batch = random.randint(batch_range[0], batch_range[1]) 
        
        return pkt_path

    def __batched_random(self, trace, path_range, batch_range):
        n_path = np.random.choice(path_range)

        pkt_in, pkt_out = 0, 0
        last_c_path = np.random.choice(np.arange(0, n_path))
        c_batch = random.randint(batch_range[0], batch_range[1]) 
        last_r_path = np.random.choice(np.arange(0, n_path))
        r_batch = random.randint(batch_range[0], batch_range[1]) 

        pkt_path = []    
        for pkt in trace:
            if (pkt[1] == OUT): # outgoing packet
                pkt_out += 1
                pkt_path.append(last_c_path) # add path

                if (pkt_out == c_batch): # resample path & batch
                    pkt_out = 0
                    last_c_path = np.random.choice(np.arange(0, n_path))
                    c_batch = random.randint(batch_range[0], batch_range[1])
            elif (pkt[1] == IN): # incoming packet
                pkt_in += 1
                pkt_path.append(last_r_path) # add path
                
                if (pkt_in == r_batch): # resample path & batch
                    pkt_in = 0
                    last_r_path = np.random.choice(np.arange(0, n_path))
                    r_batch = random.randint(batch_range[0], batch_range[1]) 
        
        return pkt_path

    def __split_trace(self, trace, pkt_path):
        n_path = max(pkt_path)+1

        spl_traces = []
        for p in range(n_path):
            one_trace = [trace[i] for i, p_path in enumerate(pkt_path) if p_path == p]
            spl_traces.append(one_trace) 

        return spl_traces  

    def __padding(self, trace):
        c_br_loc, c_br_pad = self.__client_break_params()
        r_br_loc, r_br_pad = self.__relay_break_params()
        c_ex_pad = self.__client_extend_param()
        r_ex_pad = self.__relay_extend_param()

        ipt_o2o = 0.0001
        delay = 0
        last_pkt_time = 0.0
        last_pkt_direct = OUT
        rtt=0.12

        c_con_in, c_con_out = 0, 0
        defend_trace = []
        for e in trace:
            defend_trace.append(e)
            pkt_time, pkt_direct = e[0], e[1]

            if last_pkt_direct == OUT and pkt_direct == IN: # update rtt
                rtt = float(pkt_time - last_pkt_time)

            # current packet is a outgoing packet
            if pkt_direct == OUT:
                c_con_out += 1

                if c_con_in != 0:
                    c_con_in = 0

                # client extend burst
                if c_con_out == 2 and uniform.rvs(size=1)[0] >= 0.7:  
                    time_c_ex = pkt_time
                    for _ in range(c_ex_pad): # client extend packets
                        time_c_ex += ipt_o2o
                        defend_trace.append([time_c_ex, 888*OUT])
                    c_ex_pad = self.__client_extend_param()                    

                # relay break burst
                if c_con_out == r_br_loc: # r_br_loc
                    time_r_br = pkt_time + rtt/2
                    for _ in range(r_br_pad): # r_br_pack
                        time_r_br += ipt_o2o
                        defend_trace.append([time_r_br, 777*IN])
                    c_con_out = 0       
                    r_br_loc, r_br_pad = self.__relay_break_params()  

                last_pkt_time = pkt_time
                last_pkt_direct = pkt_direct

            elif pkt_direct == IN:
                c_con_in += 1
                if c_con_out != 0:
                    c_con_out = 0 

                # relay extend burst
                if c_con_in == 2 and uniform.rvs(size=1)[0] >= 0.9: 
                    time_r_ex = pkt_time
                    for _ in range(r_ex_pad): # relay extend packets
                        time_r_ex += ipt_o2o
                        defend_trace.append([time_r_ex, 888*IN])
                    r_ex_pad = self.__relay_extend_param()

                # client break burst
                if c_con_in == c_br_loc: # c_pad_loc
                    time_c_br = pkt_time
                    for _ in range(c_br_pad): # c_pad_pack
                        time_c_br += ipt_o2o
                        defend_trace.append([time_c_br, 777*OUT])
                    c_con_in = 0
                    c_br_loc, c_br_pad = self.__client_break_params()  

                last_pkt_time = pkt_time
                last_pkt_direct = pkt_direct
        
        # fake burst
        defend_trace = self.__first_fake_burst(defend_trace)
        defend_trace = sorted(defend_trace, key=lambda x:x[0])

        return defend_trace        

    def __client_break_params(self):
        c_br_loc = int(rayleigh.rvs(loc=5.0100934937808645, scale=5.4042479185778625, size=1)[0])   
        c_br_pad = geom.rvs(p=0.6720704613633172, loc=0.0, size=1)[0]   
        
        c_br_loc = max(c_br_loc, 6) # minimum
        c_br_pad = max(c_br_pad, 1)

        c_br_loc = min(c_br_loc, 15) # maximum
        c_br_pad = min(c_br_pad, 4)

        return c_br_loc, c_br_pad

    def __relay_break_params(self):
        r_br_loc = int(rayleigh.rvs(loc=3.9520158881596017, scale=3.8923620672750783, size=1)[0])   
        r_br_pad = geom.rvs(p=0.45767131312950055, loc=0.0, size=1)[0]   

        r_br_loc = max(r_br_loc, 5) # minimum
        r_br_pad = max(r_br_pad, 1)

        r_br_loc = min(r_br_loc, 18) # maximum
        r_br_pad = min(r_br_pad, 6)

        return r_br_loc, r_br_pad

    def __client_extend_param(self):
        c_ex_pad = geom.rvs(p=0.6718020099927879, size=1)[0]
        c_ex_pad = min(c_ex_pad, 4) # maximum
        
        return c_ex_pad        

    def __relay_extend_param(self):
        r_ex_pad = geom.rvs(p=0.4571386203563751, size=1)[0]
        r_ex_pad = min(r_ex_pad, 5) # maximum

        return r_ex_pad 

    def __first_fake_burst(self, trace):
        cur_time = 1.00
        rtt = 0.0
        
        loop_num = random.randint(1, 5)
        for _ in range(loop_num): # loop times
            # fake outgoing burst
            fake_c_out = geom.rvs(p=0.7620060515717053, size=1)[0]
            fake_c_out = min(fake_c_out, 5)
            for _ in range(fake_c_out): # fake outgoing packets
                cur_time += 0.0001*random.randint(1, 3)
                trace.append([cur_time, 999*OUT])

            # fake incoming burst
            cur_time += random.uniform(0.100, 0.300)# o-i iat
            fake_c_in = int(rayleigh.rvs(loc=2.142881728007377, scale=2.998029777424256, size=1)[0]) # fake incoming packets
            fake_c_in = min(fake_c_in, 10)
            for _ in range(fake_c_in):
                cur_time = cur_time+0.0001*random.randint(1, 3)# i-i iat
                trace.append([cur_time, 999*IN]) 

            cur_time += 0.5   

        return trace  
                
    def __end_fake_burst(self, trace):
        last_time = trace[-1][0]

        for i in range(random.randint(1, 5)): # loop times
            # fake outgoing burst
            if trace[-1][1] == OUT:
                last_time += 0.0001 # o-o iat
            else:
                last_time += 0.002 # i-o iat
           
            fake_c_out = geom.rvs(p=0.7, size=1)[0]
            for _ in range(fake_c_out): # fake outgoing packets
                last_time = last_time+0.0001 # o-o iat
                trace.append([last_time, 555*OUT])

            # fake incoming burst
            last_time += 0.0002 # o-i iat
            fake_c_in = geom.rvs(p=0.2, size=1)[0] # fake incoming packets
            for _ in range(fake_c_in):
                last_time = last_time+0.0001 # i-i iat
                trace.append([last_time, 777*IN]) 

        return trace  


if __name__ == "__main__":
    import os
    from os.path import join, exists
    from preprocess import Preprocess

    data_dir = '/Users/huangbin/desktop/WF/script/data/Wang-20000'
    out_dir = '/Users/huangbin/desktop/swp/test'
    file = '0-0.cell'

    std_trace = Preprocess.bigenough(data_dir, file) # preprocess the trace. 
    print(f"std_trace:{len(std_trace)}")
    swp = SWP()
    defend_trace = swp.defend(std_trace) # get the defended trace

    for idx, trc in enumerate(defend_trace):
        spl_dir = out_dir+'-'+str(idx)
        if not exists(spl_dir):
            os.makedirs(spl_dir)        

        with open(join(spl_dir, file), 'w') as f:  # save the defended trace
            for e in trc:
                f.write(str(e[0])+'\t'+str(e[1])+'\n')  




