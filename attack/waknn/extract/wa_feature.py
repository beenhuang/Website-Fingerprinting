#!/usr/bin/env python3

"""
<file>    wa_feature.py
<brief>   Wa-kNN feature proposed by Wang.
"""

import numpy

def get_feature(trace, max_size=15):
    times = [x[0] for x in trace]
    sizes = [x[1] for x in trace]

    return extract_feature(times, sizes, max_size)

# times: [time, ...]
# sizes: [size, ...]
def extract_feature(times, sizes, max_size):
    features = []

    #Transmission size features
    features.append(len(sizes))

    count = 0
    for x in sizes:
        if x > 0:
            count += 1
    features.append(count)
    features.append(len(times)-count)

    features.append(times[-1] - times[0])

    #Unique packet lengths
##    for i in range(-1500, 1501):
##        if i in sizes:
##            features.append(1)
##        else:
##            features.append(0)

    #Transpositions (similar to good distance scheme)
    count = 0
    for i in range(0, len(sizes)):
        if sizes[i] > 0:
            count += 1
            features.append(i)
        if count == 500:
            break
    for i in range(count, 500):
        features.append("X")
        
    count = 0
    prevloc = 0
    for i in range(0, len(sizes)):
        if sizes[i] > 0:
            count += 1
            features.append(i - prevloc)
            prevloc = i
        if count == 500:
            break
    for i in range(count, 500):
        features.append("X")


    #Packet distributions (where are the outgoing packets concentrated)
    count = 0
    for i in range(0, min(len(sizes), 3000)):
        if i % 30 != 29:
            if sizes[i] > 0:
                count += 1
        else:
            features.append(count)
            count = 0
    for i in range(int(len(sizes)/30), 100):
        features.append(0)

    #Bursts
    bursts = []
    curburst = 0
    consnegs = 0
    stopped = 0
    for x in sizes:
        if x < 0:
            consnegs += 1
            if (consnegs == 2):
                bursts.append(curburst)
                curburst = 0
                consnegs = 0
        if x > 0:
            consnegs = 0
            curburst += x
    if curburst > 0:
        bursts.append(curburst)
    if (len(bursts) > 0):
        features.append(max(bursts))
        features.append(numpy.mean(bursts))
        features.append(len(bursts))
    else:
        features.append("X")
        features.append("X")
        features.append("X")
##    print bursts
    counts = [0, 0, 0, 0, 0, 0]
    for x in bursts:
        if x > 2:
            counts[0] += 1
        if x > 5:
            counts[1] += 1
        if x > 10:
            counts[2] += 1
        if x > 15:
            counts[3] += 1
        if x > 20:
            counts[4] += 1
        if x > 50:
            counts[5] += 1
    features.append(counts[0])
    features.append(counts[1])
    features.append(counts[2])
    features.append(counts[3])
    features.append(counts[4])
    features.append(counts[5])
    for i in range(0, 100):
        try:
            features.append(bursts[i])
        except:
            features.append("X")

    for i in range(0, 10):
        try:
            features.append(sizes[i] + 1500)
        except:
            features.append("X")

    itimes = [0]*(len(sizes)-1)
    for i in range(1, len(sizes)):
        itimes[i-1] = times[i] - times[i-1]
    if len(itimes) > 0:
        features.append(numpy.mean(itimes))
        features.append(numpy.std(itimes))
    else:
        features.append("X")
        features.append("X")


    #changed_features = []
    #for x in features:
    #    if x == "X":
    #        x = -1
    #    changed_features.append(x)    
    

    changed_features = [x if x != "X" else -1 for x in features] 

    if(len(changed_features) > max_size):
        changed_features = changed_features[:max_size]
    else: # pad -1
        changed_features += [-1]*(max_size-len(changed_features))
           


    return changed_features    
