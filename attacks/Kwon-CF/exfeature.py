#!usr/bin/env python3

"""
<file>    exfeature.py
<brief>   extract features from a trace.
"""

from os.path import abspath, dirname, join, basename
import numpy as np

# constants
DIRECTION_IN = 1
DIRECTION_OUT = -1

MAX_COUNT = 10

###########  get general trce  ##########

# general trace: [[timestamp1, direction1], [ts2, direct2], ... ]
def get_general_trace(trace):
    if(trace[3] == 'general'):
        label = 0
    elif(trace[3] == 'IpClient'):
        label = 1
    elif(trace[3] == 'IpHS'):
        label = 2
    elif(trace[3] == 'RpClient'):
        label = 3
    elif(trace[3] == 'RpHS'):
        label = 4    

    gen_total = [int(x.split(":")[2]) for x in trace[4:]]        

    return gen_total, label

################## transform trace ###################

# index array of outgoing packets
def get_first_10_cells(trace):  
    trace.extend(0 for _ in range(MAX_COUNT-len(trace)))       

    return trace[:10]

# get general in/out trace
def get_inout_in_50(trace):
    gen_in=[x for x in trace[:50] if x == DIRECTION_IN]
    gen_out=[x for x in trace[:50] if x == DIRECTION_OUT]

    return len(gen_in), len(gen_out)


############# main function #################

def extract_features(trace):
    # 210 features
    all_features = []
    
    gen_total, label = get_general_trace(trace)

    first_10 =  get_first_10_cells(gen_total) 
    out_50, in_50 = get_inout_in_50(gen_total)


    # [1] circuit construction sequence
    all_features.extend(first_10)
    # [2] number of outgoing packets within the first 50 cells
    #all_features.append(out_50)
    # [3] number of incoming packets within the first 50 cells
    #all_features.append(in_50)
    # [4] transmission time
    #all_features.append(gen_total[-1][0] - gen_total[0][0])    


    return all_features, label


if __name__ == "__main__":
    BASE_DIR = abspath(dirname(__file__))
    INPUT_DIR = join(BASE_DIR, "client")
    FILE_NAME = "6-33"

    features, label = extract_features(join(INPUT_DIR, FILE_NAME))

    #label = get_trace_label(join(INPUT_DIR, FILE_NAME))
    #print(f"label: {label}")

    print(f"[new] {FILE_NAME}, {len(features)}: ")
    for elem in features:
        print(elem) 

