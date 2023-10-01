#!/usr/bin/env python3

"""
<file>    var_feature.py
<brief>   feature vector
"""

OUT = 1
IN = -1

def var_feature(trace, max_size=5000):
    # timestamp sequence
    time_only = [x[0] for x in trace]
    if(len(time_only) > max_size):
        time_only = time_only[:max_size]
    else: # pad 0
        time_only += [0]*(max_size-len(time_only))    
    
    # direction sequence
    direction_only = [x[1] for x in trace]
    if(len(direction_only) > max_size):
        direction_only = direction_only[:max_size]
    else: # pad 0
        direction_only += [0]*(max_size-len(direction_only))    

    # metadata
    total_pkt = len(trace)
    out_pkt = [x[1] for x in trace].count(OUT)
    in_pkt = [x[1] for x in trace].count(IN)
    total_time = trace[-1][0] - trace[0][0]

    meta = [total_pkt, 
            in_pkt, 
            out_pkt, 
            in_pkt / total_pkt, 
            out_pkt / total_pkt, 
            total_time, 
            total_time / total_pkt]

    return (time_only+meta, direction_only+meta) 

if __name__ == "__main__":
    from preprocess import Preprocess

    standard_trace = Preprocess.wang20000("/Users/huangbin/desktop/WF/script/data/Wang-20000", "8-88.cell")
    feature = var_feature(standard_trace)

    #print(feature)

    with open("/Users/huangbin/desktop/WF/script/attack/varcnn/8-88.cell", "w") as f:
        for row in feature:
            for e in row:
                f.write(f"{e}\n")