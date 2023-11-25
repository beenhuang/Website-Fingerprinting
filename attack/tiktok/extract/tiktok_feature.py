#!/usr/bin/env python3

"""
<file>    tiktok_feature.py
<brief>   directional timing feature for Tik-Tok
"""

# trace format: [[timestamp, direction], ...]
# directional timing feature: timestamp * direction
def get_feature(trace, start_time=0.0001, max_size=5000):
    # Modify the star_time to 0.0001
    direct_timing = [(x[0]+start_time)*x[1] for x in trace]

    if(len(direct_timing) > max_size):
        direct_timing = direct_timing[:max_size]
    else: # pad 0
        direct_timing += [0]*(max_size-len(direct_timing))

    return direct_timing
