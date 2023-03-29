#!/usr/bin/env python3

"""
<file>    dt_feature.py
<brief>   directional timing feature for Tik-Tok
"""

# -1 is IN, 1 is OUT
# directional_timing: time * direction
def dt_feature(trace_data, max_size=5000):
    directional_timing = [x[0]*x[1] for x in trace_data]

    while len(directional_timing)<max_size:
        directional_timing.append(0)

    return directional_timing[:max_size]
