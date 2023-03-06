#!/usr/bin/env python3

"""
<file>    exfeature.py
<brief>   extract the CUMUL feature
"""

import os
import sys
import pickle
import itertools
import argparse
import logging
import numpy as np
from os.path import abspath, dirname, join, pardir, splitext, basename, exists

PACKET_SIZE = 514
MAX_LENGTH = 100

# trace_format: [size, ...]
# positive is incoming, negative is outgoing
def cumul_feature(trace, feature_path, featureCount=MAX_LENGTH, separateClassifier=False):
    if featureCount % 2 != 0 :
        sys.exit(f"[ERROR] feature_size is invalid.")

    features = []
            
    total = []
    cum = []
    pos = []
    neg = []
    inSize = 0
    outSize = 0
    inCount = 0
    outCount = 0
            
    # Process trace
    for packetsize in itertools.islice(trace, None): 
        # incoming packets
        if packetsize > 0:
            inSize += packetsize
            inCount += 1
            # cumulated packetsizes
            if len(cum) == 0:
                cum.append(packetsize)
                total.append(packetsize)
                pos.append(packetsize)
                neg.append(0)
            else:
                cum.append(cum[-1] + packetsize)
                total.append(total[-1] + abs(packetsize))
                pos.append(pos[-1] + packetsize)
                neg.append(neg[-1] + 0)
            
        # outgoing packets
        if packetsize < 0:
            outSize += abs(packetsize)
            outCount += 1
            if len(cum) == 0:
                cum.append(packetsize)
                total.append(abs(packetsize))
                pos.append(0)
                neg.append(abs(packetsize))
            else:
                cum.append(cum[-1] + packetsize)
                total.append(total[-1] + abs(packetsize))
                pos.append(pos[-1] + 0)
                neg.append(neg[-1] + abs(packetsize))
            
    # add feature
    #features.append(classLabel)
    features.append(inCount)
    features.append(outCount)
    features.append(outSize)
    features.append(inSize)
            
    if separateClassifier:
        # cumulative in and out
        posFeatures = np.interp(np.linspace(total[0], total[-1], featureCount/2), total, pos)
        negFeatures = np.interp(np.linspace(total[0], total[-1], featureCount/2), total, neg)
        for el in itertools.islice(posFeatures, None):
            features.append(el)
        for el in itertools.islice(negFeatures, None):
            features.append(el)
    else:
        # cumulative in one
        cumFeatures = np.interp(np.linspace(total[0], total[-1], featureCount+1), total, cum)
        for el in itertools.islice(cumFeatures, 1, None):
            features.append(el) 

    with open(feature_path, "w") as f:
        for element in features:
            f.write(f"{element}\n")

    return features           


def preprocess(trace):
    good_trace = []

    for e in trace:
        # in cumul, positive is incoming when positive of Wang's Data is outgoing
        size = int(e.split("\t")[1].strip("\n")) * PACKET_SIZE * -1

        good_trace.append(size)

    logger.debug(f"good_trace: {good_trace}")

    return good_trace
   
def main(data, feature_dir):
    flist = os.listdir(data)

    X, y = [], []
    for f in flist:
        feature_fname = f.split(".")[0]
        logger.info(f"feature_fname: {feature_fname}")

        file_path = join(data, f)
        with open(file_path, "r") as fi:
            trace = fi.readlines() 

        good_trace = preprocess(trace)  
        feature_path = join(feature_dir, feature_fname)

        #feature = extract_features(good_trace, feature_path)  
        feature = cumul_feature(good_trace, feature_path)

        if "-" in feature_fname:
            label = int(feature_fname.split("-")[0])
        else:
            label = 100
        
        logger.info(f"label: {label}")

        X.append(feature)
        y.append(label)

    with open(join(feature_dir, "feature.pkl"), "wb") as f:
        pickle.dump((X, y), f)    
    
    logger.info(f"Complete")    


# create logger
def get_logger():
    logging.basicConfig(format="[%(asctime)s]>>> %(message)s", level=logging.INFO, datefmt = "%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(splitext(basename(__file__))[0])
    
    return logger

# parse arugment
def parse_arguments():
    # argument parser
    parser = argparse.ArgumentParser(description="CUMUL")

    # INPUT
    parser.add_argument("-i", "--in", required=True, help="load trace data")

    args = vars(parser.parse_args())

    return args


if __name__ == "__main__":
    MODULE_NAME = basename(__file__)

    BASE_DIR = abspath(join(dirname(__file__), pardir, pardir))
    IN_DATA_DIR = join(BASE_DIR, "data")
    MAIN_FEATURE_DIR = join(BASE_DIR, "attack", "cumul", "feature")

    try:
        logger = get_logger()
        logger.info(f"{MODULE_NAME}: start to run.")

        # parse arguments
        args = parse_arguments()
        logger.info(f"Arguments: {args}")

        data = join(IN_DATA_DIR, args["in"])

        feature_dir = join(MAIN_FEATURE_DIR, args["in"])
        if not exists(feature_dir):
            os.makedirs(feature_dir)

        main(data, feature_dir)    


    except KeyboardInterrupt:
        sys.exit(-1) 
