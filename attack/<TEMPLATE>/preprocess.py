#!/usr/bin/env python3

"""
<file>    preprocess.py
<brief>   preprocess trace data
"""

from os.path import join

class Preprocess():

    # for Wang-20000 dataset
    @staticmethod
    def Wang20000(data_dir, file):
        with open(join(data_dir,file), "r") as f:
            trace = f.readlines()         

        DELIMITER="\t"
        start_time = float(trace[0].split(DELIMITER)[0])
        
        standard_trace = []
        for e in trace:
            e = e.split(DELIMITER)
            time = float(e[0]) - start_time
            direction = int(e[1].strip("\n"))
            standard_trace.append([time, direction])

        return standard_trace



