#!/usr/bin/env python3

"""
<file>    goodenough.py
<brief>   used for GoodEnough dataset
"""

import os
from os.path import join, splitext


CIRCPAD_EVENT_NONPADDING_SENT = "circpad_cell_event_nonpadding_sent"
CIRCPAD_EVENT_NONPADDING_RECV = "circpad_cell_event_nonpadding_received"
CIRCPAD_ADDRESS_EVENT = "connection_ap_handshake_send_begin" 

NONPADDING_SENT = 1
NONPADDING_RECV = -1

# return ["time \t direction" ...]
# time: float
# direction: int
def standard_trace(lines, strip=True):
    if strip:
        for idx, line in enumerate(lines):
            if CIRCPAD_ADDRESS_EVENT in line:
                lines = lines[idx:]
                break

    start_time = int(lines[0].split(" ")[0])
    
    trace = []
    
    # travel
    for x in lines:
        # transform from nanosecond to second
        time = float(x.split(" ")[0]) * 0.000000001
        #time = float((int(x.split(" ")[0]) - start_time) * 0.000000001)
 
        # sent nonpadding case
        if CIRCPAD_EVENT_NONPADDING_SENT in x:
            trace.append(f"{time}\t{NONPADDING_SENT}")
        # recv nonpadding case
        elif CIRCPAD_EVENT_NONPADDING_RECV in x:
            trace.append(f"{time}\t{NONPADDING_RECV}")
        # other case    
        else:
            continue

    return trace

# get data/trace file list
def get_file_list(data_dir):
    flist, trace_flist = [], []

    # get directories
    c_mon_dir = join(data_dir, "client-traces", "monitored")
    c_unm_dir = join(data_dir, "client-traces", "unmonitored")
    
    # 1. monitored traces
    for fname in os.listdir(c_mon_dir):
        # append data file
        flist.append(join(c_mon_dir, fname))

        # get trace file name
        fname = splitext(fname)[0].split("-")
        #print(fname)
        site = str(fname[0])
        instance = int(site[-1]) * 20 + int(fname[1])

        if site[:-1] == "" :
            trace_flist.append(f"{0}-{instance}")
        else:
            trace_flist.append(f"{site[:-1]}-{instance}")
   
    # 2. unmonitored traces
    x = 0
    for fname in os.listdir(c_unm_dir):
        # append data file
        flist.append(join(c_unm_dir, fname))
        # append trace file
        trace_flist.append(f"{x}")
        x += 1


    return zip(flist, trace_flist)
