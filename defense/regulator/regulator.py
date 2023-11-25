#!/usr/bin/env python3

"""
<file>    regulator.py
<brief>   regulator defense
"""
import numpy as np

class RegulaTor():
    def __init__(self):
        self.orig_rate = 277  # initial rate -> R: light=260, heavy=277
        self.depr_rate = 0.94  # depreciation rate -> D: light=0.86, heavy=0.94
        self.burst_thr = 3.55  # T: light=3.75, heavy=3.55
        self.max_pad_budget = 3550  # max_padding_budget N: light=2080, heavy=3550
        self.up_ratio = 3.95  # download/upload ratio U: light=4.02, heavy=3.95
        self.delay_cap = 1.77  # C: light=2.08, heavy=1.77

    def defend(self, trace):
        download_packets = [x[0] for x in trace if x[1] < 0]
        upload_packets = [x[0] for x in trace if x[1] > 0]

        #get defended traces
        padded_download = self.__regulator_download(download_packets, self.orig_rate, self.depr_rate, self.max_pad_budget, self.burst_thr)
        padded_upload = self.__regulator_upload_full(padded_download, upload_packets, self.up_ratio, self.delay_cap)

        download_packets = [(p, -1) for p in padded_download]
        upload_packets = [(p, 1) for p in padded_upload]
        both_output = sorted(download_packets + upload_packets, key=lambda x: x[0])

        return both_output

    def __regulator_download(self, target_trace, orig_rate, depreciation_rate, max_padding_budget, burst_threshold):
        padding_budget = np.random.randint(0,max_padding_budget)

        output_trace = []
        upload_trace = []
     
        position = 10

        #send packets at a constant rate initially (to construct circuit)
        download_start = target_trace[position]
        added_packets = int(download_start*10)
        for i in range(added_packets):
            pkt_time = i*.1
            output_trace.append(pkt_time)

        output_trace.append(target_trace[position])

        current_time = download_start
        burst_time = target_trace[position]
        padding_packets = 0
        position = 1
        
        while True:
            #calculate target rate
            target_rate = orig_rate * (depreciation_rate**(current_time - burst_time))
            
            if(target_rate < 1):
                target_rate = 1        
            
            #if the original trace has been completely sent
            if(position == len(target_trace)):
                break
            
            #find number of real packets waiting to be sent
            queue_length = 0
            for c in range(position, len(target_trace)):
                if(target_trace[c] < current_time):
                    queue_length += 1
                else:
                    break      

            #if waiting packets exceeds treshold, then begin a new burst
            if(queue_length > (burst_threshold*target_rate)):
                burst_time = current_time
            
            #calculate gap
            gap = 1 / float(target_rate)
            current_time += gap
            
            if(queue_length == 0 and padding_packets >= padding_budget):
                #no packets waiting and padding budget reached
                continue
            elif(queue_length == 0 and padding_packets < padding_budget):
                #no packets waiting, but padding budget not reached
                output_trace.append(current_time)
                padding_packets += 1
            else:
                #real packet to send
                output_trace.append(current_time)
                position += 1
           
        #print(f"target_trace: {len(target_trace)}, output_trace: {len(output_trace)}")   
        return output_trace

    def __regulator_upload_full(self, download_trace, upload_trace, upload_ratio, delay_cap):
        output_trace = []

        #send one upload packet for every $upload_ratio download packets 
        upload_size = int(len(download_trace)/upload_ratio)
        output_trace = list(np.random.choice(download_trace, upload_size))

        #send at constant rate at first
        download_start = download_trace[10]
        added_packets = int(download_start*5)
        for i in range(added_packets):
            pkt_time = i*.2
            output_trace.append(pkt_time)

        #assign each packet to the next scheduled sending time in the output trace
        output_trace = sorted(output_trace)
        delay_packets = []
        packet_position = 0
        for t in upload_trace:
            found_packet = False
            for p in range(packet_position+1, len(output_trace)):
                if(output_trace[p] >= t and (output_trace[p]-t) < delay_cap):
                    packet_position = p
                    found_packet = True
                    break
     
            #cap delay at delay_cap seconds
            if(found_packet == False):
                delay_packets.append(t+delay_cap)     
       
        output_trace += delay_packets

        #print(f"output_trace: {len(output_trace)}")
        return sorted(output_trace)

if __name__ == "__main__":
    from os.path import join
    from preprocess import Preprocess

    data_dir = '/Users/huangbin/desktop/WF/script/data/Wang-20000'
    out_dir = '/Users/huangbin/desktop/regulator'
    file = '0-0.cell'

    std_trace = Preprocess.wang20000(data_dir, file) # preprocess the trace. 
    print(f"std_trace:{len(std_trace)}")
    rt = RegulaTor()
    defend_trace = rt.defend(std_trace) # get the defended trace

    with open(join(out_dir, file), 'w') as f:  # save the defended trace
        for e in defend_trace:
            f.write(str(e[0])+'\t'+str(e[1])+'\n') 
