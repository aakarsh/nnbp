#!/usr/bin/python
import argparse
import glob
import os
from itertools import izip_longest


    
def summary(result_dir):
    file_list = glob.glob(result_dir+"/*.res")    
    trace_time = {}
    total  = 0
    count = 0
    for f in file_list:
        filename = os.path.basename(f)
        fh = open(f)
        line = fh.readline()
        results ={}
        while line:
            values = line.split()[1:]
            for (v1,v2,v3) in grouper(3,values):                
                results[v1] = v3
            if "TRACE" in results:
                break
            else:
                line = fh.readline()
        
        if "MISPRED_PER_1K_INST" in results and len(results) > 0:           
            trace_time[filename] = float(results["MISPRED_PER_1K_INST"])
            total += trace_time[filename]
            count += 1
            
        fh.close()
    return (trace_time,total,count)
   
    
    
def print_summary(path="/home/aakarsh/src/nnbp/src/results/MYRESULTS"):
    "Print the summary of misprediction results "
    (trace_time,total,count) = summary(path)
    for key in sorted(trace_time,key=trace_time.get,reverse=True):
        print "%-10f\t%s"%(trace_time[key],key)
        
    print "Total : %-10f\nAverage : %-10f\n" % (total,total/count)


def compare_results(r1,r2):
    (trace_time_1,total_1) = summary(r1)
    (trace_time_2,total_2) = summary(r2)
    delta_results = {}
    for key in sorted(trace_time_1,key=trace_time_1.get,reverse=True):
        delta = 0
        if key in trace_time_2:
            delta_results[key] = trace_time_1[key] - trace_time_2[key]
            
    for key in sorted(delta_results,key=delta_results.get,reverse=True):
        print "%10s \t%+10f"% (key,delta_results[key])
        
        
def grouper(n,iterable,fill_value = None):
    "Group iterable in groups of n  fill rest with fill_value "
    args = [iter(iterable)]*n
    return izip_longest(fillvalue = fill_value,*args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize result of running the traces")
    subparsers = parser.add_subparsers(dest="cmd") 
    compare_parser = subparsers.add_parser('compare', help='a help')
    compare_parser.add_argument("-r1")
    compare_parser.add_argument("-r2")
    summary_parser = subparsers.add_parser('summary', help='a help')    
    summary_parser.add_argument('--result',help="directory containing results")
    
    args  = parser.parse_args()
    if args.cmd == "summary": 
        if args.result:
            print "Using result directory %s " % args.result
            print_summary(args.result)
        else:
            print_summary();
    elif args.cmd == "compare":
        compare_results(args.r1,args.r2)
