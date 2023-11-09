#!/glade/work/bgreene/conda-envs/LES/bin/python
# --------------------------------
# Name: process.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 31 March 2023
# Purpose: process output from LES using functions from LESutils
# 3 Nov 2023: call nc_rotate on each sim
# --------------------------------
import sys
sys.path.append("..")
import os
import yaml
import numpy as np
from LESutils import out2netcdf, calc_stats, nc_rotate
from argparse import ArgumentParser
from multiprocessing import Process

# arguments for simulation directory to process
parser = ArgumentParser()
parser.add_argument("-d", required=True, action="store", dest="dsbl", nargs=1,
                    help="Simulation base directory")
parser.add_argument("-s", required=True, action="store", dest="sim", nargs=1,
                    help="Simulation name")
args = parser.parse_args()

# construct simulation directory and ncdir
dout = os.path.join(args.dsbl[0], args.sim[0], "output") + os.sep
dnc = f"{dout}netcdf/"

# call out2netcdf
# here, processing hours 8-9 = timesteps 900000 - 1080000
# use params27.yaml
with open(f"{dout}params27.yaml") as fp:
    pp = yaml.safe_load(fp)
# add simlabel to params
pp["simlabel"] = pp["path"].split(os.sep)[-2]
# add nhr_s
pp["nhr_s"] = "8-9h"
# add dnc
pp["dnc"] = dnc

# define timesteps
t0 = 900000
t1 = 988000
t2 = 990000
t3 = 1080000
dt = pp["nwrite"]

# call nc_rotate (this loops over time already)
# split into two processes
# initiate concurrent processes for each half of the timesteps
process1 = Process(target=nc_rotate, 
                    args=(dnc, t0, t1, dt))
process2 = Process(target=nc_rotate, 
                    args=(dnc, t2, t3, dt))
# begin each process
process1.start()
process2.start()
# join each process
process1.join()
process2.join()

# # loop over timesteps and call out2netcdf
# # keep track of f_all for later
# f_all = []
# for tt in timesteps:
#     print(f"Processing timestep: {tt}")
#     f_all.append(f"{dnc}all_{tt:07d}.nc")
#     # out2netcdf(dout, tt, del_raw=False, **pp)

# # run calc_stats using f_all
# print(f"Begin calculating stats for timesteps {timesteps[0]} - {timesteps[-1]}")
# pp["return_fstats"] = True
# fstats = calc_stats(f_use=f_all, **pp)

# # finally, calc stats for hours 8-10 = timesteps 900000 - 1260000
# # to be consistent with previous hour block, load with same freq as in params27.yaml
# # rewrite nhr_s
# pp["nhr_s"] = "8-10h"
# # construct new timesteps for hrs 8-10
# tf2 = 1260000
# timesteps2 = np.arange(t0, tf2+1, pp["nwrite"], dtype=np.int32)
# # loop over timesteps to define new array of f_all2
# f_all2 = [f"{dnc}all_{tt:07d}.nc" for tt in timesteps2]
# # compute new stats file
# print(f"Begin calculating stats for timesteps {timesteps2[0]} - {timesteps2[-1]}")
# pp["return_fstats"] = True
# fstats = calc_stats(f_use=f_all2, **pp)

print(f"Finished processing {args.sim[0]} output for hours 8-9!")