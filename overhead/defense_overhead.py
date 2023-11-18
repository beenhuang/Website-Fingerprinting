#!/usr/bin/env python3

"""
<file>    defense_overhead.py
<brief>   
"""

def defense_overhead(undef_out, undef_in, undef_time, def_out, def_in, def_time):
    out_bw = (def_out-undef_out)/float(undef_out)
    in_bw = (def_in-undef_in)/float(undef_in)
    
    bw_oh = (def_out+def_in-undef_out-undef_in)/float(undef_in+undef_out)
    time_oh = (def_time-undef_time)/float(undef_time)

    return [f"[Packet] orig_out:{format(undef_out, ',')}, orig_in:{format(undef_in, ',')}, orig_total:{format(undef_out+undef_in, ',')}, orig_out/orig_total:{undef_out/float(undef_out+undef_in):.5f}\n",
            f"def_out:{format(def_out, ',')}, def_in:{format(def_in, ',')}, def_total:{format(def_out+def_in, ',')}, def_out/def_total:{def_out/float(def_out+def_in):.5f}\n",
            f"pad_out:{format(def_out-undef_out, ',')}, pad_in:{format(def_in-undef_in, ',')}, pad_total:{format(def_out+def_in-undef_out-undef_in, ',')}, pad_out/pad_total:{(def_out-undef_out)/float(def_out+def_in-undef_out-undef_in):.5f}\n",
            f"[Bandwidth overhead]\n",
            f"out_bw:{out_bw:.5f}\n",
            f"in_bw:{in_bw:.5f}\n",
            f"bw_oh:{bw_oh:.5f}\n\n",
            f"[Time] orig_time:{undef_time:.5f}, def_time:{def_time:.5f}, add_time:{def_time-undef_time:.5f}\n",
            f"[Time overhead]\n",
            f"time_oh:{time_oh:.5f}\n\n"]

if __name__ == "__main__":
    pass
