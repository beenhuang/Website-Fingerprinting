#!/usr/bin/env python3

"""
<file>    cf_feature.py
<brief>   cf feature proposed by Huang.
"""

import math
import itertools
import numpy as np

# constants
DIRECTION_OUT = 1
DIRECTION_IN = -1


# general trace: [[timestamp1, direction1], [ts2, direct2], ... ]
def general_trace(trace, all_features):   
    #directions = [int(x.split(":")[2]) for x in trace[4:]] 
    #print(f"directions: {directions}, len: {len(directions)}")    
    #print(f"len of trace: {len(trace)}")  

    directions = [x[1] for x in trace]  

    # [1] number of packets
    all_features.append(len(directions))

    return directions


def in_out_trace(directions, all_features):
    in_trace = [x for x in directions if x == DIRECTION_IN]
    out_trace = [x for x in directions if x == DIRECTION_OUT]

    #all_features.append(f"IN-OUT-START")
    all_features.append(len(out_trace))
    all_features.append(len(in_trace))
    all_features.append(len(out_trace)/float(len(directions)))
    all_features.append(len(in_trace)/float(len(directions)))
    #all_features.append(f"IN-OUT-END")


def first_last_outgoing(directions, n, all_features):
    first_n_out = [x for x in directions[:n] if x == DIRECTION_OUT]
    last_n_out = [x for x in directions[-n:] if x == DIRECTION_OUT]

    #print(f"last N: {directions[-n:]}")

    #all_features.append(f"FIRST-LAST-{n}-START")
    all_features.append(len(first_n_out))
    all_features.append(len(last_n_out))
    all_features.append(len(first_n_out)/float(n))
    all_features.append(len(last_n_out)/float(n))
    #all_features.append(f"FIRST-LAST-{n}-END")

def burst_trace(directions, all_features):
    burst = [k*len(list(v)) for k,v in itertools.groupby(directions)]
    #print(f"burst: {burst}")

    burst_out = [b for b in burst if b > 0]
    burst_in = [b for b in burst if b < 0]

    #all_features.append("BURST-START")
    all_features.append(min(burst))
    all_features.append(max(burst))
    all_features.append(len(burst))
    all_features.append(len(burst_out))
    all_features.append(len(burst_in))
    all_features.append(len(burst_out)/float(len(burst)))
    all_features.append(len(burst_in)/float(len(burst)))   
    all_features.append(np.mean(burst))
    all_features.append(np.std(burst))
    #all_features.append("BURST-END")

    return burst


def round_trace(burst, all_features):
    round_trace = [sum(burst[i:i+2]) for i in range(0,len(burst),2)]
    #print(f"round: {round_trace}")

    #all_features.append("ROUND-START")
    all_features.append(min(round_trace))
    all_features.append(max(round_trace))
    all_features.append(len(round_trace))
    all_features.append(np.mean(round_trace))
    all_features.append(np.std(round_trace))
    #all_features.append("ROUND-END")

    return round_trace
 

def accumulate_trace(directions, all_features):
    cumul = list(itertools.accumulate(directions))
    #print(f"accumul: {cumul}, len:{len(cumul)}")

    pos_directions = [x for x in directions if x > 0]

    abs_cumul = list(itertools.accumulate([abs(x) for x in directions]))
    #print(f"accumul: {abs_cumul}, len:{len(abs_cumul)}")

    cumFeatures = np.interp(np.linspace(abs_cumul[0], abs_cumul[-1], 101), abs_cumul, cumul)
    #print(f"cumFeatures: {cumFeatures[1:]}")

    #all_features.append("CUMUL-START")
    all_features.extend(cumFeatures)
    #all_features.append("CUMUL-END")    

def ngram_trace(directions, n, all_features):
    two_gram = list(itertools.pairwise(directions))
    new_two_gram = [sum(x) for x in two_gram]
    #print(f"2gram: {new_two_gram}, len:{len(new_two_gram)}")


def chunk_size_trace(directions, all_features, chunk_size=400):
    chunks = [directions[i:i+chunk_size] for i in range(0, len(directions), chunk_size)]
    #print(f"chunks: {chunks}")

    chunk_out = [chunk.count(DIRECTION_OUT) for chunk in chunks]
    #print(f"chunk_out: {chunk_out}")

    #chunk_in = [chunk.count(DIRECTION_IN) for chunk in chunks]
    #print(f"chunk_in: {chunk_in}")

    #all_features.append("CHUNK-SIZE-START")
    all_features.append(min(chunk_out))
    all_features.append(max(chunk_out))
    all_features.append(np.mean(chunk_out))
    all_features.append(np.std(chunk_out))
    #all_features.append("CHUNK-SIZE-END")  

    return chunk_out


def chunk_num_trace(directions, all_features, num_chunk=20):
    chunk_size =  math.ceil(len(directions)/float(num_chunk))

    chunks = [directions[i:i+chunk_size]for i in range(0, len(directions), chunk_size)]
    #print(f"chunks: {chunks}")
    #print(f"chunk_size: {chunk_size}, len:{len(directions)}")

    chunk_out = [chunk.count(DIRECTION_OUT) for chunk in chunks]
    #print(f"chunk_out: {chunk_out}")

    #chunk_in = [chunk.count(DIRECTION_IN) for chunk in chunks]
    #print(f"chunk_in: {chunk_in}")

    #all_features.append("CHUNK-NUM-START")
    all_features.append(min(chunk_out))
    all_features.append(max(chunk_out))
    all_features.append(np.mean(chunk_out))
    all_features.append(np.std(chunk_out))
    #all_features.append("CHUNK-NUM-END")

    return chunk_out    

# trace format: [[time, direction] ... ]
# outgoing is 1, in incoming is -1
def cf_feature(trace, max_size=4000):
    # all features list
    all_features = []

    # discrete features
    directions = general_trace(trace, all_features)
    in_out_trace(directions, all_features)

    first_last_outgoing(directions, 30, all_features)
    first_last_outgoing(directions, 20, all_features)

    burst = burst_trace(directions, all_features)
    round_t = round_trace(burst, all_features)

    #ngram_trace( directions, 2, all_features)

    chunk_size = chunk_size_trace(directions, all_features, chunk_size=400)
    chunk_num = chunk_num_trace(directions, all_features, num_chunk=20)

    # continuous features
    all_features.extend(burst)
    all_features.extend(round_t)
    all_features.extend(chunk_size)
    all_features.extend(chunk_num)
    accumulate_trace(directions, all_features)

    # when less than max_size, fill 0
    all_features.extend(0 for _ in range(max_size-len(all_features))) 
    
    return all_features[:max_size]
