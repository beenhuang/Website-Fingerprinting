#!/usr/bin/env python3

"""
<file>    wtfpad_protected.py
<brief>   used for WTF-PAD protected dataset
"""


# return ["time \t direction" ...]
# time: float
# direction: int
def standard_trace(lines):
    start_time = float(lines[0].split("\t")[0])
    
    trace = []
    
    # travel
    for e in lines:
        e = e.split("\t")
        time = float(e[0]) - start_time
        
        size = int(e[1].strip("\n"))
        direction = 1 if size > 0 else -1

        good_trace.append(f"{time} {direction}")
    
    return trace

# get data/trace file list
def get_file_list(data_dir):
    flist = os.listdir(data_dir)

    return zip(flist, flist)
