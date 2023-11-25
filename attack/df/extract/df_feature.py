#!/usr/bin/env python3

"""
<file>    df_feature.py
<brief>   feature vectors for DF model
"""

# 1 is outgoing, -1 is incoming
# trace format: [[timestamp, direction, ...], ...]
def get_feature(trace, max_size=5000):
    direct_only = [x[1] for x in trace]

    if(len(direct_only) > max_size):
        direct_only = direct_only[:max_size]
    else: # pad 0
        direct_only += [0]*(max_size-len(direct_only))

    return direct_only
