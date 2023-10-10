#!/usr/bin/env python3

"""
<file>    preprocess.py
<brief>   preprocess trace data
"""

from os.path import join

class Preprocess():
    # for Wang-20000 dataset
    @staticmethod
    def wang20000(data_dir, file):
        return Preprocess.standard_trace(data_dir, file, "\t")
 
    # for BigEnough dataset
    @staticmethod
    def bigenough(data_dir, file):
        return Preprocess.standard_trace(data_dir, file, "\t")

    @staticmethod
    def standard_trace(data_dir, file, delimiter):
        with open(join(data_dir,file), "r") as f:
            trace = f.readlines()  

        if len(trace) == 0: # file is empty.
            return trace            

        start_time = float(trace[0].split(delimiter)[0])
        
        std_trace = []
        for e in trace:
            e = e.split(delimiter)

            time = float(e[0]) - start_time
            direction = int(e[1].strip("\n"))

            std_trace.append([time, direction])

        return std_trace    


