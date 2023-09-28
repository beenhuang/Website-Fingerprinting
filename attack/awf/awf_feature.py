#!/usr/bin/env python3

"""
<file>    awf_feature.py
<brief>   feature vector for AWF model
"""

# 1 is outgoing, -1 is incoming
# trace format: [[timestamp, direction], ...]
def awf_feature(trace, max_size=2000):
    direction_only = [x[1] for x in trace]

    if(len(direction_only) > max_size):
        direction_only = direction_only[:max_size]
    else: # pad 0 if the length of the trace is less than max_size.
        direction_only += [0]*(max_size-len(direction_only))

    return direction_only
