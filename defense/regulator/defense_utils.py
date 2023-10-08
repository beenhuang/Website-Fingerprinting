import pickle
import numpy as np

OUTGOING = 1
INCOMING = -1

def get_trace(file_name, cutoff_time=120, cutoff_length=20000):
    '''Given a filename, returns a list representation for processing'''

    all_lines = []
    with open(file_name) as fptr:
        for line in fptr:
            e = line.split("\t")
            time = float(e[0])
            direction = int(e[1].strip("\n"))

            if int(direction) > 0:
                direction = OUTGOING
            else:
                direction = INCOMING

            trace_line = (time, direction)

            if time < cutoff_time:
                all_lines.append(trace_line)

    if len(all_lines) > cutoff_length:
        all_lines = all_lines[:cutoff_length]

    return all_lines

def get_download_packets(trace):
    '''Takes trace and returns list of timestamps of download packets'''
    output_trace = [packet[0] for packet in trace if packet[1] == -1]
    return output_trace

def get_upload_packets(trace):
    '''Takes trace and returns list of timestamps of download packets'''
    output_trace = [packet[0] for packet in trace if packet[1] == 1]
    return output_trace

def get_time_gaps(trace):
    '''Convert 1-d list of times into 1-d list of gaps'''
    output_trace = []
    current_time = 0.0
    for packet in trace:
        output_trace.append(float(packet) - current_time)
        current_time = float(packet)
    return output_trace
