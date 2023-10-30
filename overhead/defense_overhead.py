#!/usr/bin/env python3

"""
<file>    defense_overhead.py
<brief>   
"""

def defense_overhead(undef_out, undef_in, undef_time, def_out, def_in, def_time):
    return [f"[Undefended packets] out:{format(undef_out, ',')}, in:{format(undef_in, ',')}, total:{format(undef_out+undef_in, ',')}, out/total:{undef_out/float(undef_out+undef_in):.5f}\n",
            f"[Defended packets]   out:{format(def_out, ',')}, in:{format(def_in, ',')}, total:{format(def_out+def_in, ',')}, out/total:{def_out/float(def_out+def_in):.5f}\n\n",
            f"[Outgoing Bandwidth] out_pad:{format(def_out-undef_out, ',')}, out_BW:{(def_out-undef_out)/float(undef_out):.5f}\n",
            f"[Incoming Bandwidth] in_pad:{format(def_in-undef_in, ',')}, in_BW:{(def_in-undef_in)/float(undef_in):.5f}\n",
            f"[Bandwidth Overhead] total_pad:{format(def_out+def_in-undef_out-undef_in, ',')}, [Bandwidth Overhead]:{(def_out+def_in-undef_out-undef_in)/float(undef_in+undef_out):.5f}\n\n",
            f"[Undefended Time]:{undef_time:.5f}, [Defended Time]:{def_time:.5f}\n",
            f"[Latency Overhead] time_pad:{def_time-undef_time:.5f}, [Time Overhead]:{(def_time-undef_time)/float(undef_time):.5f}\n\n"]

if __name__ == "__main__":
    pass
