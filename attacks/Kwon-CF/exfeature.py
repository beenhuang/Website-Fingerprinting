#!usr/bin/env python3

"""
<file>    exfeature.py
<brief>   extract features from a trace.
"""

from os.path import abspath, dirname, join, basename
import numpy as np

# constants
DIRECTION_OUT = 1.0
DIRECTION_IN = -1.0

MAX_COUNT = 100
MAX_LABEL = 53

###########  get general trce  ##########

# general trace: [[timestamp1, direction1], [ts2, direct2], ... ]
def get_general_trace(file):
    trace = []

    #print(f"file: {file}")
    with open(file, "r") as f:
        for line in f.readlines():
            line = line.split("\t")
            trace.append([float(line[0]), int(line[1])])       

    return trace


################## transform trace ###################

# get general in/out trace
def transform_general_inout_trace(trace):
    gen_in = [elem for elem in trace if elem[1] == DIRECTION_IN]
    gen_out = [elem for elem in trace if elem[1] == DIRECTION_OUT]

    return gen_in, gen_out


# index array of outgoing packets
def transform_index_out_trace(trace):
    index_out = [index for index,elem in enumerate(trace) if elem[1] == DIRECTION_OUT]
    
    # if less than MAX_COUNT, then fill -1
    index_out.extend(-1 for _ in range(MAX_COUNT-len(index_out)))       

    return index_out[:MAX_COUNT]


# current_outgoing_index - previous_outgoing_index
def transform_relative_index_out(trace):
    relative_index_out = []
    prev_index = 0

    for index, elem in enumerate(trace):
        if elem[1] == DIRECTION_OUT:
            relative_index_out.append(index - prev_index)
            prev_index = index

    # if less than MAX_COUNT, then fill -1
    relative_index_out.extend(-1 for _ in range(MAX_COUNT-len(relative_index_out)))  

    return relative_index_out[:MAX_COUNT]



def transform_unknown_burst(trace):
    bursts = []
    curburst = 0
    stopped = 0

    for elem in trace:
        if elem[1] == DIRECTION_IN:
            stopped = 0
            curburst -= elem[1]
            #curburst += 1
        if elem[1] == DIRECTION_OUT and stopped == 0:
            stopped = 1
        if elem[1] == DIRECTION_OUT and stopped == 1:
            stopped = 0
            bursts.append(curburst)
 
    return bursts

def get_trace_label(file):
    if "-" in basename(file):
        label = int(basename(file).split("-")[0])
    else:
        label = MAX_LABEL    

    return label

############# main function #################

def extract_features(file):
    # 210 features
    all_features = []
    
    gen_total = get_general_trace(file)

    gen_in, gen_out =  transform_general_inout_trace(gen_total) 
    index_out = transform_index_out_trace(gen_total)
    relative_index_out = transform_relative_index_out(gen_total)
    
    # ?
    unknown_bursts = transform_unknown_burst(gen_total)
    #print(f"bursts: {bursts}")

    # [1] number of packets
    all_features.append(len(gen_total))
    # [2] number of outgoing packets
    all_features.append(len(gen_out))
    # [3] number of incoming packets
    all_features.append(len(gen_in))
    # [4] transmission time
    all_features.append(gen_total[-1][0] - gen_total[0][0])

    # [5] index number of outgoing packets, size:100
    all_features.extend(index_out)
    # [6] relative index number of outgoing packets, size:100
    all_features.extend(relative_index_out)

    # [7] max number of bursts        
    all_features.append(max(unknown_bursts))
    # [8] average of bursts
    all_features.append(sum(unknown_bursts)/len(unknown_bursts))
    # [9] number of bursts
    all_features.append(len(unknown_bursts))
    # [10] length of burst > 5
    all_features.append(len([elem for elem in unknown_bursts if elem > 5]))
    # [11] length of burst > 10
    all_features.append(len([elem for elem in unknown_bursts if elem > 10]))
    # [12] length of burst > 15
    all_features.append(len([elem for elem in unknown_bursts if elem > 15]))


    # the label of the trace
    label = get_trace_label(file)

    #if len(all_features) != 210:
    #    print(f"feature: {len(all_features)}, file: {file}")

    return (all_features, label)


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

