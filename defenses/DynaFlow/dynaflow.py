# DynaFlow

import os


"""Creates defended dataset."""
def defend(site, inst, switch_sizes, end_sizes, first_time_gap, poss_time_gaps, subseq_length, memory, suffix_2):
    # get old sequence
    if inst != None:
        site_data = open("%s/%d-%d" % (data_loc, site, inst), "r")
    else:
        site_data = open("%s/%d" % (data_loc, site), "r")

    old_packets = []
    for line in site_data:
        line = line.split()
        # add [time, direction, length] for each packet
        old_packets.append([float(line[0]), int(line[1]), int(line[3])])

    site_data.close()    

    # make a copy of old packet sequence
    packets = old_packets[:]

    # get new sequence 
    choices = ""
    new_packets = []     
    past_times = []
    past_packets = []
    index = 0
    time_gap = first_time_gap
    # first packet at time zero 
    curr_time = -1 * time_gap

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
            if packets[i][0] <= curr_time and packets[i][1] == curr_dir and packets[i][2] + packet_size <= 498:
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
            choices += ("T" + str(time_gap_info[1]))

        # move on to next packet 
        index += 1 

        # get length of defended sequence before any extra padding at end
        if len(packets) == 0 and min_index > index:
            min_index = index 


    # update choices
    choices += ("E" + str(end_sizes.index(index)))    
    choices_stats = open(choices_loc + suffix + suffix_2, "a")
    if inst != None:
        choices_stats.write("%s %s\n" % (site, choices))    
    else:
        choices_stats.write("unmonitored %s\n" % (choices))
    choices_stats.close()

    # write new seq
    new_data_loc = data_loc + suffix + suffix_2
    if not os.path.exists(new_data_loc):
        os.mkdir(new_data_loc)

    if inst != None:
        new_site_data = open("%s/%d-%d" % (new_data_loc, site, inst), "w")
    else:
        new_site_data = open("%s/%d" % (new_data_loc, site), "w")

    for packet in new_packets:
        new_site_data.write("%f %d\n" % (packet[0], packet[1]))
    new_site_data.close()

    return [old_packets[-1][0], new_packets[-1][0], len(old_packets), len(new_packets), min_index]


# Finds new time gap for defended sequence.
def find_new_time_gap(past_times, curr_time, time_gap, poss_time_gaps, memory, block_size):
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


def create_end_sizes(k):
    """Creates list of possible sizes for defended sequence."""   

    end_sizes = []
    for i in range(0, 9999):
        if k ** i > 10000000:
            break
        end_sizes.append(round(k ** i))
    return end_sizes

# closed world
# Runs DynaFlow in closed-world.
def run_closed(suffix_2, switch_sizes, m):
    open(choices_loc + suffix + suffix_2, "w").close()

    end_sizes = create_end_sizes(m)

    oldt, newt, oldbw, newbw = 0, 0, 0, 0
    for site in range(0, 100):
        for inst in range(0, 90):
            print(f"site:{site}, inst:{inst}")
            ret = defend(site, inst, switch_sizes, end_sizes, FIRST_TIME_GAP, POSS_TIME_GAPS, SUBSEQ_LENGTH, MEMORY, suffix_2)
            if ret != None:
                oldt+=ret[0]
                newt+=ret[1]
                oldbw+=ret[2]
                newbw+=ret[3]   

    print(f"old time: {oldt}, new time: {newt}, old packets: {oldbw}, new packets: {newbw}")                     


    toh = float(newt-oldt)/oldt
    bwoh = float(newbw-oldbw)/oldbw

    print(f"closed, switch sizes: {switch_sizes}, m: {m}, \
            first time gap: {FIRST_TIME_GAP}, poss time gaps: {POSS_TIME_GAPS}, \
            memory: {MEMORY}\nTOH: {toh}, BWOH: {bwoh}, S: {toh+bwoh}\n\n")
   
    with open("dynaflow.results", "a") as f:
        f.write(f"closed, switch sizes: {switch_sizes}, m: {m}, \
                  first time gap: {FIRST_TIME_GAP}, poss time gaps: {POSS_TIME_GAPS}, \
                  memory: {MEMORY}\nTOH: {toh}, BWOH: {bwoh}, S: {toh+bwoh}\n\n")

# open world
# Runs DynaFlow in open-world.
def run_open(suffix_2, switch_sizes, m):
    open(choices_loc + suffix + suffix_2, "w").close()

    end_sizes = create_end_sizes(m)

    oldt, newt, oldbw, newbw = 0, 0, 0, 0
    for site in range(0, 100):
        for inst in range(0, 90):
            print(f"site:{site}, inst:{inst}")
            ret = defend(site, inst, switch_sizes, end_sizes, FIRST_TIME_GAP, POSS_TIME_GAPS, SUBSEQ_LENGTH, MEMORY, suffix_2)
            if ret != None:
                oldt+=ret[0]
                newt+=ret[1]
                oldbw+=ret[2]
                newbw+=ret[3]            


    toh = float(newt-oldt)/oldt
    bwoh = float(newbw-oldbw)/oldbw

    print("open-mon, switch sizes: %s, m: %s, first time gap: %s, poss time gaps: %s, memory: %s\nTOH: %s, BWOH: %s, S: %s\n\n" % (switch_sizes, m, FIRST_TIME_GAP, POSS_TIME_GAPS, MEMORY, toh, bwoh, toh+bwoh))
    
    results = open("dynaflow.results", "a")  
    results.write("open-mon, switch sizes: %s, m: %s, first time gap: %s, poss time gaps: %s, memory: %s\nTOH: %s, BWOH: %s, S: %s\n\n" % (switch_sizes, m, FIRST_TIME_GAP, POSS_TIME_GAPS, MEMORY, toh, bwoh, toh+bwoh))
    results.close()


    oldt, newt, oldbw, newbw = 0, 0, 0, 0
    for site in range(0, 9000):
        print(f"site:{site}")
        ret = defend(site, None, switch_sizes, end_sizes, FIRST_TIME_GAP, POSS_TIME_GAPS, SUBSEQ_LENGTH, MEMORY, suffix_2)
        if ret != None:
            oldt+=ret[0]
            newt+=ret[1]
            oldbw+=ret[2]
            newbw+=ret[3]            

    print(f"newbw: {newbw}")
    print(f"oldbw: {oldbw}")

    toh = float(newt-oldt)/oldt
    bwoh = float(newbw-oldbw)/oldbw

    print("open-unmon, switch sizes: %s, m: %s, first time gap: %s, poss time gaps: %s, memory: %s\nTOH: %s, BWOH: %s, S: %s\n\n" % (switch_sizes, m, FIRST_TIME_GAP, POSS_TIME_GAPS, MEMORY, toh, bwoh, toh+bwoh))
    
    results = open("dynaflow.results", "a")  
    results.write("open-unmon, switch sizes: %s, m: %s, first time gap: %s, poss time gaps: %s, memory: %s\nTOH: %s, BWOH: %s, S: %s\n\n" % (switch_sizes, m, FIRST_TIME_GAP, POSS_TIME_GAPS, MEMORY, toh, bwoh, toh+bwoh))
    results.close()


if __name__ == "__main__":
    data_loc = "batches/batch-primes"
    choices_loc = "choices/choices-primes"
    suffix = "-defended"

    # tests 
    FIRST_TIME_GAP = 0.012
    SUBSEQ_LENGTH = 4
    MEMORY = 100

    POSS_TIME_GAPS = [0.0012, 0.005]
    run_closed("-closed-6", [400, 1200, 2000, 2800], 1.2)

    #POSS_TIME_GAPS = [0.0012, 0.005]
    #run_open("-open-6", [400, 1200, 2000, 2800], 1.2)



