#!/usr/bin/python

import glob
import os
from itertools import izip_longest


def print_summary():
    " Print the summary of misprediction results  "
    path = "/var/host/media/removable/My Passport/src/results/MYRESULTS"
    file_list = glob.glob(path+"/*.res")
    for f in file_list:
        filename = os.path.basename(f)
        fo = open(f)
        line = fo.readline()
        
        values = line.split()[1:]
        results ={}
        for (v1,v2,v3) in grouper(3,values):
            results[v1] = v3

        if "MISPRED_PER_1K_INST" in results and len(results) > 0:
            print filename,":",results["MISPRED_PER_1K_INST"]


def grouper(n,iterable,fill_value = None):
    "Group iterable in groups of n  fill rest with fill_value "
    args = [iter(iterable)]*n
    return izip_longest(fillvalue = fill_value,*args)

if __name__ == "__main__":
    print_summary();
