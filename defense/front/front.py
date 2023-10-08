#!/usr/bin/env python3

"""
<file>    front.py
<brief>   FRONT defense
"""

import numpy as np

OUT_PADDING_PACKET = 777
IN_PADDING_PACKET = -777

# wrapper for FRONT written by Author
def defend(trace):
    client_min_dummy_pkt_num = 1
    client_dummy_pkt_num = 600
    server_min_dummy_pkt_num = 1
    server_dummy_pkt_num = 1400
    min_wnd = 1
    max_wnd = 8
    start_padding_time = 0

    np_trace = np.array(trace)
    return RP(np_trace, client_min_dummy_pkt_num, \
                     client_dummy_pkt_num, \
                     server_min_dummy_pkt_num, \
                     server_dummy_pkt_num, \
                     min_wnd, max_wnd, \
                     start_padding_time)

# format: [[time, pkt],[...]]
# trace, cpkt_num, spkt_num, cwnd, swnd
def RP(trace, client_min_dummy_pkt_num, \
              client_dummy_pkt_num, \
              server_min_dummy_pkt_num, \
              server_dummy_pkt_num, \
              min_wnd, max_wnd, \
              start_padding_time):

    client_wnd = np.random.uniform(min_wnd, max_wnd)
    server_wnd = np.random.uniform(min_wnd, max_wnd)
    if client_min_dummy_pkt_num != client_dummy_pkt_num:
        client_dummy_pkt = np.random.randint(client_min_dummy_pkt_num,client_dummy_pkt_num)
    else:
        client_dummy_pkt = client_dummy_pkt_num
    if server_min_dummy_pkt_num != server_dummy_pkt_num:
        server_dummy_pkt = np.random.randint(server_min_dummy_pkt_num,server_dummy_pkt_num)
    else:
        server_dummy_pkt = server_dummy_pkt_num
    #logger.debug("client_wnd:",client_wnd)
    #logger.debug("server_wnd:",server_wnd)
    #logger.debug("client pkt:", client_dummy_pkt)
    #logger.debug("server pkt:", server_dummy_pkt)

    first_incoming_pkt_time = trace[np.where(trace[:,1] <0)][0][0]
    last_pkt_time = trace[-1][0]    
    
    client_timetable = getTimestamps(client_wnd, client_dummy_pkt)
    client_timetable = client_timetable[np.where(start_padding_time+client_timetable[:,0] <= last_pkt_time)]

    server_timetable = getTimestamps(server_wnd, server_dummy_pkt)
    server_timetable[:,0] += first_incoming_pkt_time
    server_timetable = server_timetable[np.where(start_padding_time+server_timetable[:,0] <= last_pkt_time)]

    
    # print("client_timetable")
    # print(client_timetable[:10])
    client_pkts = np.concatenate((client_timetable, OUT_PADDING_PACKET*np.ones((len(client_timetable),1))),axis = 1)
    server_pkts = np.concatenate((server_timetable, IN_PADDING_PACKET*np.ones((len(server_timetable),1))),axis = 1)


    noisy_trace = np.concatenate( (trace, client_pkts, server_pkts), axis = 0)
    noisy_trace = noisy_trace[ noisy_trace[:, 0].argsort(kind = 'mergesort')]
    return noisy_trace

def getTimestamps(wnd, num):
    # timestamps = sorted(np.random.exponential(wnd/2.0, num))   
    # print(wnd, num)
    # timestamps = sorted(abs(np.random.normal(0, wnd, num)))
    timestamps = sorted(np.random.rayleigh(wnd,num))
    # print(timestamps[:5])
    # timestamps = np.fromiter(map(lambda x: x if x <= wnd else wnd, timestamps),dtype = float)
    return np.reshape(timestamps, (len(timestamps),1))
