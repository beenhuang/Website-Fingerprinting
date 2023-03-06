#!/usr/bin/env python3

"""
<file>    exfeature.py
<brief>   extract features from a trace.
"""

import argparse
import os
import csv
import sys
import math
import logging
import pickle
import itertools
import numpy as np
from os.path import abspath, dirname, join, basename, pardir, exists, splitext

# constants
DIRECTION_OUT = 1
DIRECTION_IN = -1

MAX_SIZE = 1000

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
def extract_features(trace, out_file):
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


    with open(out_file, "w") as f:
        for element in all_features:
            f.write(f"{element}\n")

    # when less than max_size, fill 0
    all_features.extend(0 for _ in range(MAX_SIZE-len(all_features))) 
    # when more than max_size, only use features limited max_size   
    #features = all_features[:MAX_SIZE]        
 

    return all_features[:MAX_SIZE]

def get_label(trace):
    if(trace[3] == 'general'):
        label = 0
    elif(trace[3] == 'RpClient'):
        label = 1
    elif(trace[3] == 'RpHS'):
        label = 2      
    elif(trace[3] == 'IpClient'):
        label = 3
    elif(trace[3] == 'IpHS'):
        label = 4
    else:
        print(f"unrecognized label: {trace[3]}")  

    return label          


def main(data_dir, out_file):
    # gen files
    files = [join(data_dir, "general-trace", file) for file in os.listdir(join(data_dir, "general-trace"))]
    # hs files
    hs_files = [join(data_dir, "hs-trace", file) for file in os.listdir(join(data_dir, "hs-trace"))]
    files.extend(hs_files)

    max_inst = [0, 0, 0, 0, 0] # gen, C-RP, H-RP, C-IP, H-IP

    X, y = [], []
    for file in files: 
        with open(file, "r") as f:
            reader = csv.reader(f, delimiter=",")

            for trace in reader:
                label = get_label(trace)
                feature = extract_features(trace, join(OUT_DIR, f"{label}", f"{label}-{max_inst[label]}"))

                max_inst[label] += 1

                X.append(feature)
                y.append(label)   

    with open(out_file, "wb") as f:
        pickle.dump((X, y), f)
        
    print(f"[SAVED] original dataset,labels to the {out_file} file") 


# create logger
def get_logger():
    logging.basicConfig(format="[%(asctime)s]>>> %(message)s", level=logging.INFO, datefmt = "%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(splitext(basename(__file__))[0])
    
    return logger

# parse arugment
def parse_arguments():
    # argument parser
    parser = argparse.ArgumentParser(description="cf")

    # INPUT
    parser.add_argument("-i", "--in", required=True, help="load trace data")

    args = vars(parser.parse_args())

    return args


def preprocess(trace):
    start_time = float(trace[0].split("\t")[0])
    #logger.debug(f"start_time: {start_time}")
    good_trace = []

    for element in trace:
        time = float(element.split("\t")[0]) - start_time
        direction = int(element.split("\t")[1].strip("\n"))

        good_trace.append([time, direction])

    #logger.debug(f"good_trace: {good_trace}")

    return good_trace
   

def main2(data_dir, feature_dir):
    flist = os.listdir(data_dir)

    X, y = [], []
    for f in flist:
        feature_fname = f.split(".")[0]
        logger.info(f"feature_fname: {feature_fname}")

        file_path = join(data_dir, f)
        with open(file_path, "r") as fi:
            trace = fi.readlines() 

        good_trace = preprocess(trace)  
        feature_path = join(feature_dir, feature_fname)

        feature = extract_features(good_trace, feature_path)  

        if "-" in feature_fname:
            label = int(feature_fname.split("-")[0])
        else:
            label = 100
        
        logger.debug(f"label: {label}")

        X.append(feature)
        y.append(label)

    with open(join(feature_dir, "feature.pkl"), "wb") as f:
        pickle.dump((X, y), f)    
    
    logger.info(f"Complete")    

if __name__ == "__main__":
    MODULE_NAME = basename(__file__)
    #CURRENT_TIME = time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())

    BASE_DIR = abspath(join(dirname(__file__), pardir, pardir))
    MAIN_DATA_DIR = join(BASE_DIR, "data")
    MAIN_FEATURE_DIR = join(BASE_DIR, "attack", "cf", "feature")

    try:
        logger = get_logger()
        args = parse_arguments()
        logger.info(f"{MODULE_NAME} -> Arguments: {args}")

        data_dir = join(MAIN_DATA_DIR, args["in"])

        if not exists(MAIN_FEATURE_DIR):
            os.makedirs(MAIN_FEATURE_DIR)
            
        feature_dir = join(MAIN_FEATURE_DIR, args["in"])
        if not exists(feature_dir):
            os.makedirs(feature_dir)

        main2(data_dir, feature_dir)    


    except KeyboardInterrupt:
        sys.exit(-1)    



    #print(f"test")

    #one_file = join(INPUT_DATA_DIR, "general-trace", "general(s1).csv")
    #with open(one_file, "r") as f:
    #        reader = csv.reader(f, delimiter=",")

    #        for trace in reader:
                #print(f"trace: {trace}")
    #            extract_features(trace, out_file)
    #            sys.exit(0)


