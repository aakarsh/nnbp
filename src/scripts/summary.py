#!/usr/bin/python

import glob
import os
from itertools import izip_longest


def print_summary():
    " Print the summary of misprediction results  "
    path = "/var/host/media/removable/My Passport/src/results/MYRESULTS"
    file_list = glob.glob(path+"/*.res")
    trace_time = {}
    for f in file_list:
        filename = os.path.basename(f)
        fh = open(f)
        line = fh.readline()
        
        values = line.split()[1:]
        results ={}
        for (v1,v2,v3) in grouper(3,values):
            results[v1] = v3
        
        if "MISPRED_PER_1K_INST" in results and len(results) > 0:
            trace_time[filename] = float(results["MISPRED_PER_1K_INST"])
        fh.close()

    for key in sorted(trace_time,key=trace_time.get,reverse=True):
        print "%-10f\t%s"%(trace_time[key],key)


def grouper(n,iterable,fill_value = None):
    "Group iterable in groups of n  fill rest with fill_value "
    args = [iter(iterable)]*n
    return izip_longest(fillvalue = fill_value,*args)

if __name__ == "__main__":
    print_summary();
