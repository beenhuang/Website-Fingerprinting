#!/usr/bin/env python3

"""
<file>    rf_feature.py
<brief>   rf feature proposed by Shen.
"""

# Length of TAM
max_matrix_len = 1800
# Maximum Load Time
maximum_load_time = 80

# wrapper
def get_feature(trace):
    times = [x[0] for x in trace]
    sizes = [x[1] for x in trace]

    return extract_feature(times, sizes)


def extract_feature(times, sizes):
    feature = [[0 for _ in range(max_matrix_len)], [0 for _ in range(max_matrix_len)]]
    
    for i in range(0, len(sizes)):
        if sizes[i] > 0: # incoming
            if times[i] >= maximum_load_time:
                feature[0][-1] += 1
            else:
                idx = int(times[i] * (max_matrix_len - 1) / maximum_load_time)
                feature[0][idx] += 1
        if sizes[i] < 0: # outgoing
            if times[i] >= maximum_load_time:
                feature[1][-1] += 1
            else:
                idx = int(times[i] * (max_matrix_len - 1) / maximum_load_time)
                feature[1][idx] += 1

    return feature

if __name__ == "__main__":
    from preprocess import Preprocess

    standard_trace = Preprocess.Wang20000("/Users/huangbin/desktop/WF/script/data/Wang-20000", "8-88.cell")
    feature = rf_feature(standard_trace)

    with open("/Users/huangbin/desktop/WF/script/attack/rf/8-88.cell", "w") as f:
        for row in feature:
            for e in row:
                f.write(f"{e}\n")