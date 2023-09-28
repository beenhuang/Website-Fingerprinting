#!/usr/bin/env python3

"""
<file>    df_feature.py
<brief>   feature vectors for DF model
"""

# 1 is outgoing, -1 is incoming
# trace format: [[timestamp, direction, ...], ...]
def df_feature(trace, max_size=5000):
    direction_only = [x[1] for x in trace]

    if(len(direction_only) > max_size):
        direction_only = direction_only[:max_size]
    else: # pad 0
        direction_only += [0]*(max_size-len(direction_only))

    return direction_only
