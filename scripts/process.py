#!/home/bgreene/anaconda3/envs/LES/bin/python
# --------------------------------
# Name: process.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 31 March 2023
# Purpose: process output from LES using functions from LESutils
# Update 27 November 2023: use arguments to submit jobs on Casper
# --------------------------------
import sys
sys.path.append("/home/bgreene/LES-utils/")
import os
from argparse import ArgumentParser
from LESutils import process_raw_sim
from getsims import update_log

# arguments for simulation directory to process
parser = ArgumentParser()
parser.add_argument("-d", required=True, action="store", dest="dcbl", nargs=1,
                    help="Simulation base directory")
parser.add_argument("-s", required=True, action="store", dest="sim", nargs=1,
                    help="Simulation name")
args = parser.parse_args()

# construct simulation directory and ncdir
dout = os.path.join(args.dcbl[0], args.sim[0], "output") + os.sep
dnc = f"{dout}netcdf/"

# these values will be consistent for all CBL sims
# label to give stats file that matchs nhr
nhr_s = "5-6h"
# number of hours from end to process
nhr = 1
# organize file output first?
organize = False
# number of batch jobs, if applicable
nrun = 8
# delete raw output files?
del_raw = True
# overwrite old files if they exist?
overwrite = False
# compute stats?
cstats = True
# rotate coordinates?
rotate = False
# compute timeseries files?
ts2nc = False
# delete remaining raw files?
del_remaining = False
# log file name
flog = "sims.yaml"

print(f"----- Begin processing simulation: {args.sim[0]} -----")
# run process
fstats = process_raw_sim(dout=dout, nhr=nhr, 
                         nhr_s=nhr_s,
                         del_raw=del_raw, 
                         organize=organize,
                         nrun=nrun,
                         overwrite=nrun, 
                         cstats=cstats, 
                         rotate=rotate,
                         ts2nc=ts2nc,
                         del_remaining=del_remaining)
# update sims.yaml with resulting stats file
if fstats is not None:
    update_log(dnc, fstats, args.dcbl[0], flog)
    
print(f"Finished processing simulation: {args.sim[0]}")

print("process.py complete!")