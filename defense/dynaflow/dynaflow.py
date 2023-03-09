#!/usr/bin/env python3

"""
<file>    dynaflow.py
<brief>   DynaFlow defense
"""

# dynaflow defense
# input trace format: [[time, direction, size] ...]
# defend(site, inst, switch_sizes, end_sizes, FIRST_TIME_GAP, POSS_TIME_GAPS, SUBSEQ_LENGTH, MEMORY, suffix_2)
def dynaflow(packets, subseq_length=4, first_time_gap=0.012, poss_time_gaps=[0.0012, 0.005], switch_sizes=[400, 1200, 2000, 2800, 3600, 4400, 5200], m=1.2, memory=100, max_packet_size=512):
    # get new sequence 
    new_packets = []     
    past_times = []
    past_packets = []
    index = 0
    time_gap = first_time_gap
    # first packet at time zero 
    curr_time = -1 * time_gap

    # create end sizes
    end_sizes = create_end_sizes(m)

    min_index = 99999999
    while len(packets) != 0 or index not in end_sizes:    

        # get time and direction of next packet
        curr_time = curr_time + time_gap        
        if index % subseq_length == 0:
            curr_dir = 1
        else:
            curr_dir = -1
 
        # add needed packet
        # if possible, packet combination
        packet_size = 0
        num_used_packets = 0
        for i in range(0, len(packets)):
            if packets[i][0] <= curr_time and packets[i][1] == curr_dir and packets[i][2] + packet_size <= max_packet_size:
                num_used_packets += 1
                packet_size += packets[i][2]
                if i == 0:
                    past_times.append(packets[i][0])
                past_packets.append(packets[i])
            else:
                break

        del packets[0:num_used_packets]

        new_packets.append([curr_time, curr_dir])

        # find new time gap if time to switch 
        # update string accordingly
        # const for weighted average 
        const = 400
        if index in switch_sizes:
            time_gap_info = find_new_time_gap(past_times, curr_time, time_gap, poss_time_gaps, memory, const)    
            time_gap = time_gap_info[0]
            #choices += ("T" + str(time_gap_info[1]))

        # move on to next packet 
        index += 1 

        # get length of defended sequence before any extra padding at end
        if len(packets) == 0 and min_index > index:
            min_index = index 


    return new_packets

# Creates list of possible sizes for defended sequence. 
def create_end_sizes(m, max_idx=9999, max_endsize=10000000):
    end_sizes = []
    for i in range(0, max_idx):
        if m ** i > max_endsize:
            break
        end_sizes.append(round(m ** i))
   
    return end_sizes

# change the inter-packet time
def find_new_time_gap(past_times, curr_time, time_gap, poss_time_gaps, memory, block_size=400): 
    """Finds new time gap for defended sequence."""

    # find average time gap
    if len(past_times) >= memory:
        average_time_gap = float(past_times[-1] - past_times[-memory]) / (memory - 1)
    elif len(past_times) > 10:
        average_time_gap = float(past_times[-1] - past_times[0]) / (len(past_times) - 1)
    else:
        average_time_gap = time_gap

    # find expected time gap
    exp_packet_num = block_size + 1 * float(curr_time - past_times[-1]) / average_time_gap
    exp_time_gap = block_size / exp_packet_num * average_time_gap

    # choose next timeg gap
    min_diff = 99999
    for i in range(0, len(poss_time_gaps)):
        if min_diff > abs(exp_time_gap - poss_time_gaps[i]):
            min_diff = abs(exp_time_gap - poss_time_gaps[i])
        else:
            return [poss_time_gaps[i - 1], (i - 1)]
    return [poss_time_gaps[-1], len(poss_time_gaps) - 1]

