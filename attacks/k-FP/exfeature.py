#!usr/bin/env python3

"""
<file>    feature.py
<brief>   brief of thie file
"""
from os.path import abspath, dirname, join
from math import ceil
import numpy as np
import sys

# constants
DIRECTION_OUT = 1.0
DIRECTION_IN = -1.0

MAX_SIZE = 175

###########  get general trce  ##########

# general trace: [[timestamp1, direction1], [ts2, direct2], ... ]
def get_general_trace(trace):
    # the timestamp of the first package is 0.
    trace[:,0] -= trace[0,0]  
    # nanosecond convert to second
    trace[:,0] *= 0.0000000001

    return trace

################## transform trace ###################

# get IN/OUT general trace
def transform_general_in_out_trace(trace):
    #
    gen_in = trace[trace[:,1] == DIRECTION_IN]
    gen_out = trace[trace[:,1] == DIRECTION_OUT]

    return gen_in, gen_out

#
def transform_iat_in_out_total(trace):
    #
    gen_in, gen_out = transform_general_in_out_trace(trace)

    iat_in = np.diff(gen_in[:,0])
    iat_out = np.diff(gen_out[:,0])
    iat_total = np.diff(trace[:,0])

    return iat_in, iat_out, iat_total

def transform_timestamp_in_out_total(trace):
    #
    gen_in, gen_out =  transform_general_in_out_trace(trace)

    timestamp_in = gen_in[:,0]
    timestamp_out = gen_out[:,0]
    timestamp_total = trace[:,0]

    return timestamp_in, timestamp_out, timestamp_total

def get_first_last_30_trace(trace):
    #
    first30_in = [x for x in trace[:30] if x[1] == DIRECTION_IN]
    first30_out = [x for x in trace[:30] if x[1] == DIRECTION_OUT]

    last30_in = [x for x in trace[-30:] if x[1] == DIRECTION_IN]
    last30_out = [x for x in trace[-30:] if x[1] == DIRECTION_OUT]

    return first30_in, first30_out, last30_in, last30_out

def get_num_out_chunksize_20(trace, chunk_size=20): 
    #
    direct_trace = trace[:,1]
    direct_trace = direct_trace.copy(order="C")
    direct_trace[direct_trace == DIRECTION_IN] = 0
    

    direct_trace.resize(ceil(len(direct_trace)/chunk_size),chunk_size)
    num_out_chunksize_20 = np.sum(abs(direct_trace), axis=1, dtype=np.int32)

    return num_out_chunksize_20   

# number-per-second trace
def transform_nps_trace(trace):
    #
    last_second = ceil(trace[-1][0])

    nps_trace = []

    for i in range(1, int(last_second)+1):
        nps = [x for x in trace if i-1 <= x[0] and x[0] < i]
        nps_trace.append(len(nps))    

    return nps_trace  

# cumulative direction trace
def transform_cumul_direct_in_out(trace):
    #
    cd_in = [idx for idx in range(len(trace)) if trace[idx][1] == DIRECTION_IN]
    cd_out = [idx for idx in range(len(trace)) if trace[idx][1] == DIRECTION_OUT]
   
    return cd_in, cd_out    

#
def transform_chunk_trace(trace, num_chunks):
    #
    avg = len(trace) / float(num_chunks)
    chunk_trace = []
    last = 0.0
    while last < len(trace):
        chunk_trace.append(trace[int(last):int(last + avg)])
        last += avg

    N_chunk_trace = [sum(x) for x in chunk_trace]

    if len(N_chunk_trace) == num_chunks:
        N_chunk_trace.append(0)


    return N_chunk_trace

############# main function #################

def extract_features(trace):
    all_features = []
    #
    gen_total = get_general_trace(trace)

    gen_in, gen_out =  transform_general_in_out_trace(gen_total)  
    iat_in, iat_out, iat_total = transform_iat_in_out_total(gen_total)
    timestamp_in, timestamp_out, timestamp_total = transform_timestamp_in_out_total(gen_total)
    first30_in, first30_out, last30_in, last30_out = get_first_last_30_trace(gen_total)
    chunk_out = get_num_out_chunksize_20(gen_total)
    nps_trace = transform_nps_trace(gen_total)
    cd_in, cd_out = transform_cumul_direct_in_out(gen_total)
    _70_chunk_out = transform_chunk_trace(chunk_out, 70)
    _20_chunk_nps = transform_chunk_trace(nps_trace, 20)


    # [1] iat trace, total: 15 features
    #all_features.append(min(iat_in))
    #all_features.append(min(iat_out))
    #all_features.append(min(iat_total))

    all_features.append(iat_in.max())
    all_features.append(iat_out.max())
    all_features.append(iat_total.max())

    all_features.append(iat_in.mean())
    all_features.append(iat_out.mean())
    all_features.append(iat_total.mean())

    all_features.append(iat_in.std())
    all_features.append(iat_out.std())
    all_features.append(iat_total.std())

    all_features.append(np.percentile(iat_in, 75))
    all_features.append(np.percentile(iat_out, 75))
    all_features.append(np.percentile(iat_total, 75))

    # [2] timestamp trace, total: 27 features [OK]
    all_features.append(np.percentile(timestamp_in, 25))
    all_features.append(np.percentile(timestamp_in, 50))
    all_features.append(np.percentile(timestamp_in, 75))
    all_features.append(np.percentile(timestamp_in, 100))    

    all_features.append(np.percentile(timestamp_out, 25))
    all_features.append(np.percentile(timestamp_out, 50))
    all_features.append(np.percentile(timestamp_out, 75))
    all_features.append(np.percentile(timestamp_out, 100))   

    all_features.append(np.percentile(timestamp_total, 25))
    all_features.append(np.percentile(timestamp_total, 50))
    all_features.append(np.percentile(timestamp_total, 75))
    all_features.append(np.percentile(timestamp_total, 100)) 

    # [3] general trace, total: 30 features [OK]
    all_features.append(len(gen_in))
    all_features.append(len(gen_out))
    all_features.append(len(gen_total))  

    # [4] first/last 30 packets trace, total: 34 features [OK]
    all_features.append(len(first30_in))
    all_features.append(len(first30_out))
    all_features.append(len(last30_in))
    all_features.append(len(last30_out))

    # [5] chuck trace, total: 36 features 
    all_features.append(np.std(chunk_out))
    all_features.append(np.mean(chunk_out))

    # [6] number-per-seconde trace, total: 38 features [OK]
    all_features.append(np.mean(nps_trace))
    all_features.append(np.std(nps_trace))

    # [7] cumulative-direction in/out trace, total: 42 features
    all_features.append(np.mean(cd_out)) 
    all_features.append(np.mean(cd_in)) 
    all_features.append(np.std(cd_out)) 
    all_features.append(np.std(cd_in))  
    
    # [8] chunk & nps trace, total: 47 features [OK]
    all_features.append(np.percentile(chunk_out, 50))
    all_features.append(np.percentile(nps_trace, 50))
    all_features.append(min(nps_trace))
    all_features.append(max(nps_trace))
    all_features.append(max(chunk_out))

    # [9] percentage, total: 49 features [OK]
    all_features.append(len(gen_in)/float(len(gen_total)))
    all_features.append(len(gen_out)/float(len(gen_total)))

    # [10] alternative, total: 53 features
    all_features.extend(_70_chunk_out)
    all_features.extend(_20_chunk_nps) 
    all_features.append(sum(_70_chunk_out))
    all_features.append(sum(_20_chunk_nps))

    # [11] sum iat/timestamp/general trace, total: 56 features
    all_features.append(sum([min(iat_in), min(iat_out), min(iat_total),
                             max(iat_in), max(iat_out), max(iat_total),
                             np.mean(iat_in), np.mean(iat_out), np.mean(iat_total),
                             np.std(iat_in), np.std(iat_out), np.std(iat_total),
                             np.percentile(iat_in, 75), np.percentile(iat_out, 75), np.percentile(iat_total, 75)]))
    all_features.append(sum([np.percentile(timestamp_in, 25),
                            np.percentile(timestamp_in, 50),
                            np.percentile(timestamp_in, 75),
                            np.percentile(timestamp_in, 100),
                            np.percentile(timestamp_out, 25),
                            np.percentile(timestamp_out, 50),
                            np.percentile(timestamp_out, 75),
                            np.percentile(timestamp_out, 100),
                            np.percentile(timestamp_total, 25),
                            np.percentile(timestamp_total, 50),
                            np.percentile(timestamp_total, 75),
                            np.percentile(timestamp_total, 100)])) 
    all_features.append(sum([len(gen_in), len(gen_out), len(gen_total)]))

    # [12] chunk_out/number-per-seconde trace
    all_features.extend(chunk_out)
    all_features.extend(nps_trace)

    
    # when less than max_size, fill 0
    all_features.extend(0 for _ in range(MAX_SIZE-len(all_features))) 
    # when more than max_size, only use features limited max_size   
    features = all_features[:MAX_SIZE]

    return features


if __name__ == "__main__":
    BASE_DIR = abspath(dirname(__file__), pardir, pardir)
    INPUT_DIR = join(BASE_DIR, "data", "standard")
    FILE_NAME = "24_7"

    features = extract_features(join(INPUT_DIR, FILE_NAME))

    print(f"[new] {FILE_NAME}: ")
    for elem in features:
        print(elem)

