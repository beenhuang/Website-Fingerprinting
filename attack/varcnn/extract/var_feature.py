#!/usr/bin/env python3

"""
<file>    var_feature.py
<brief>   feature vector
"""

OUT = 1
IN = -1

def get_feature(trace, max_size=5000):
    abs_time = [x[0] for x in trace] # absolute timestamp sequence
    prev_time, next_time = abs_time[:-1], abs_time[1:]
    # inter-packet time sequence
    inter_time = [next_time[i]-prev_time[i] for i, _ in enumerate(prev_time)] 

    if(len(inter_time) > max_size):
        inter_time = inter_time[:max_size]
    else: # pad 0.0, time is float type.
        inter_time += [0.0]*(max_size-len(inter_time))    
    
    # direction sequence
    direct_only = [x[1] for x in trace]
    if(len(direct_only) > max_size):
        direct_only = direct_only[:max_size]
    else: # pad 0
        direct_only += [0]*(max_size-len(direct_only))    

    # metadata
    total_pkt = len(trace)
    out_pkt = [x[1] for x in trace].count(OUT)
    in_pkt = [x[1] for x in trace].count(IN)
    total_time = trace[-1][0] - trace[0][0]

    meta = [total_pkt, # 1. total packets
            in_pkt,    # 2. incoming packets    
            out_pkt,   # 3. outgoing packets
            in_pkt / total_pkt, # 4. in/total
            out_pkt / total_pkt, # 5. out/total
            total_time, # 6. total time
            total_time / total_pkt] # # 7. average inter-packet time.
    
    return direct_only+meta+inter_time+meta

if __name__ == "__main__":
    from preprocess import Preprocess

    std_trace = Preprocess.wang20000("/Users/huangbin/desktop/WF/script/data/Wang-20000", "8-88.cell")
    feature = get_feature(std_trace)

    with open("/Users/huangbin/desktop/WF/script/attack/varcnn/8-88.cell", "w") as f:
        for e in feature:
            f.write(f"{e}\n")
