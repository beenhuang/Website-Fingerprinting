#!/usr/bin/env python3

"""
<file>    front.py
<brief>   
"""

from numpy.random import randint, uniform, rayleigh

class FRONT():
    def __init__(self):
        self.c_min_pkt = 1
        self.c_max_pkt = 600
        self.p_min_pkt = 1
        self.p_max_pkt = 1400
        self.min_pad_time = 1
        self.max_pad_time = 8
        self.start_pad_time = 0

    def defend(self, trace):
        # sample a number of dummy packets: total number of padding packets.
        c_pad_pkt = randint(self.c_min_pkt, self.c_max_pkt)
        p_pad_pkt = randint(self.p_min_pkt, self.p_max_pkt)

        # sample a padding window: scale parameter of Rayleigh distribution.
        c_scale = uniform(self.min_pad_time, self.max_pad_time)
        p_scale = uniform(self.min_pad_time, self.max_pad_time)

        # schedule dummy packets
        last_time = trace[-1][0] 
        c_pad_times = rayleigh(c_scale, c_pad_pkt)
        c_pad_times.sort()
        c_pad_times = [x+self.start_pad_time for x in c_pad_times if x+self.start_pad_time <= last_time]

        first_in_time = [x[0] for x in trace if x[1]<0][0]
        p_pad_times = rayleigh(p_scale, p_pad_pkt)
        p_pad_times.sort()
        p_pad_times = [x+first_in_time for x in p_pad_times]
        p_pad_times = [x+self.start_pad_time for x in p_pad_times if x+self.start_pad_time <= last_time]

        # merge trace
        PAD_CELL = 777
        c_pad = [[x, 1*PAD_CELL] for x in c_pad_times]
        p_pad = [[x, -1*PAD_CELL] for x in p_pad_times]
        merge_trace = sorted(trace+c_pad+p_pad, key=lambda x:x[0])

        return merge_trace 

if __name__ == '__main__':
    from os.path import join
    from preprocess import Preprocess

    data_dir = '/Users/huangbin/desktop/WF/script/data/Wang-20000'
    out_dir = '/Users/huangbin/desktop/front'
    file = '0-0.cell'

    std_trace = Preprocess.bigenough(data_dir, file) # preprocess the trace. 
    #print(f"std_trace:{len(std_trace)}")
    front = FRONT()
    defend_trace = front.defend(std_trace) # get the defended trace

    with open(join(out_dir, file), 'w') as f:  # save the defended trace
        for e in defend_trace:
            f.write(str(e[0])+'\t'+str(e[1])+'\n')  
