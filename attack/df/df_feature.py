#!/usr/bin/env python3

"""
<file>    df_feature.py
<brief>   feature vectors for DF model
"""

# -1 is IN, 1 is OUT
def df_feature(trace_data, max_size=5000):
    directions = [x[1] for x in trace_data]

    while len(directions)<max_size:
        directions.append(0)

    return directions[:max_size]
