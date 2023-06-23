#!/usr/bin/env python3

"""
<file>    tiktok_feature.py
<brief>   directional timing feature for Tik-Tok
"""

# trace_data: [[timestamp, direction], ...]
# directional timing feature: timestamp * direction
def tiktok_feature(trace_data, start_time=0.0001, max_size=5000):
    # change start time to 0.0001
    new_trace = [[x[0]+start_time, x[1]] for x in trace_data]
    direct_timing = [x[0]*x[1] for x in new_trace]

    while len(direct_timing) < max_size:
        direct_timing.append(0)

    return direct_timing[:max_size]
