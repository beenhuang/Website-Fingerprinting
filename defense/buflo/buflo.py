#!/usr/bin/env python3

"""
<file>    buflo.py
<brief>   BuFLO defense
"""

def buflo(query_data, d, f, T):
    '''
    apply BuFlo countermeasure to traffic data. In BuFlo, it will send a packet of length d every ρ milliseconds until
    communications cease and at least τ milliseconds of time have elapsed
    :param d, determines the size of fixed-length packets
    :param f, determines the rates or frequency (in milliseconds) at which we send packets
    :param T, determines the minimum amount of time (in milliseconds) for which we must send packets.
    :return:
    '''
    outgoing = []
    incoming = []
    for p in query_data:
        
        if p[2] <= d:
            overhead = d - p[2]
            p[2] = d
            # p = np.append(p, overhead)
            p.append(overhead)
        
        if p[3] == 1:
            outgoing.append(p)
        else:
            incoming.append(p)

    t_start_out = outgoing[0][1]
    t_start_in = incoming[0][1]

    n_out = 0
    for p in outgoing:
        p[1] = t_start_out + n_out * (1 / f)
        n_out += 1

    n_in = 0
    for p in incoming:
        p[1] = t_start_in + n_in * (1 / f)
        n_in += 1

    outgoing.reverse()
    for p in outgoing:
        if p[1] <= T:
            outgoing.pop(p)
            print("pop")
    outgoing.reverse()

    incoming.reverse()
    for p in incoming:
        if p[1] <= T:
            incoming.pop(p)
    incoming.reverse()

    buflo_data_list = outgoing + incoming

    # buflo_data_list.sort()
    # echo_df1 = pd.DataFrame(buflo_data_list, columns=['index', 'time', 'size', 'direction', 'overhead'])
    # echo_df1.to_csv('sports_update_5_30s_buflo1' + ".csv")

    buflo_data_list.sort(key=sort_by_time)
    echo_df2 = pd.DataFrame(buflo_data_list, columns=['index', 'time', 'size', 'direction', 'overhead'])
    echo_df2.to_csv('sports_update_5_30s_buflo2' + ".csv")
    print(' f')

# buflo_ordered(path, query_in_matrix, 1000, 50, 100)
def buflo_ordered(csv_path, query_data, d, f, t):
    '''
    apply BuFlo countermeasure to traffic data. In BuFlo, it will send a packet of length d every ρ milliseconds until
    communications cease and at least τ milliseconds of time have elapsed. The order of packets are not changed in this method
    :param d, determines the size of fixed-length packets
    :param f, determines the rates or frequency (in milliseconds) at which we send packets
    :param t, determines the minimum amount of time (in milliseconds) for which we must send packets.
    :return:
    '''
    buflo_path = 'buflo/'
    info_path = 'info/'
    pf = Path(csv_path)
    trace_name = pf.name[0:-4]

    # [?, time, size, 
    start_t = query_data[0][1]  #to record the start time of this query
    end_time = query_data[-1][1]  #to record the end time of this query
    index = 0
    total_packet = t * f # the minimum amount of packets
    total_overhead = 0
    original_size = 0
    buflo_data = []

    for p in query_data:
        original_size = original_size + p[2]
        
    for p in query_data:
        if p[2] <= d and len(p) ==4: # size of this packt is small than d. Pad the packet
            overhead = d - p[2]
            total_overhead = total_overhead + overhead
            p[2] = d
            p.append(overhead)
            p[1] = round(start_t + index * (1 / f),2)
            p[0] = index
            p.append('padded')
            index += 1
        elif len(p) == 4: # size of this packet is larger than d. Chop the packet
            p_left = p[2] - d
            p[2] = d
            p[1] = round(start_t + index * (1 / f) ,2)
            p[0] = index
            p.append(0)
            p.append('chopped')  # a dummy packet will be added
            final_left = p_left % d
            if final_left == 0:
                n_new = int(p_left / d)
            else:
                n_new = int(p_left / d) + 1

            if n_new == 0: # if just one dummy packet are needed
                index += 1
                new_p = [index, round(start_t + index * (1 / f),2), d, p[3], d - p_left, 'new']
                total_overhead = total_overhead + (d - p_left)
                query_data.insert(index, new_p)  # add dummy packet

            while n_new > 0:  # if more dummy packets are needed
                if n_new == 1:
                    index += 1
                    new_p = [index, round(start_t + index * (1 / f),2), d, p[3], d - final_left, 'new']
                    total_overhead = total_overhead + (d - final_left)
                    query_data.insert(index, new_p)  # add a dummy packet
                else:
                    index += 1
                    new_p = [index, round(start_t + index * (1 / f),2), d, p[3], 0, 'new']
                    query_data.insert(index, new_p)  # add a dummy packet
                n_new = n_new - 1
            index += 1
        #if index > total_packet != 0:
        #    query_data = query_data[0:index]
        #   break
    if index < total_packet:
        for i in range(index, total_packet):
            seed = -1 + 2 * random.random()
            direction = float(np.sign(seed))
            dummy_packet = [i+1, round(start_t + (i + 1) * (1 / f),2), d, direction, d, 'dummy']
            total_overhead = total_overhead + d
            query_data.insert(i+1, dummy_packet)
    time_delay = query_data[-1][1] - end_time

    # query_data.append(last_packet)
    echo_df2 = pd.DataFrame(query_data, columns=['index', 'time', 'size', 'direction', 'overhead', 'status'])
    echo_df2.to_csv(buflo_path + trace_name + '_buflo' + ".csv")

    info_packet = [[trace_name, time_delay, total_overhead]]
    # echo_info = pd.DataFrame(info_packet, columns=['time delay', 'overhead'])
    # echo_info.to_csv(info_path + trace_name + '_overhead' + ".csv")
    # file_header = ['trace name', ]
    with open('/home/lhp/PycharmProjects/pcap_csv/info/overhead info_1200_50_20.csv', 'a') as info_in:
        writer = csv.writer(info_in)
        writer.writerows(info_packet)
