#!/home/bgreene/anaconda3/envs/LES/bin/python
# --------------------------------
# Name: process_quadrant.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 28 June 2023
# Purpose: loop over simulations and calculate quadrant components
# using LESutils and spec code
# --------------------------------
import sys
sys.path.append("..")
import numpy as np
from LESutils import load_stats
from spec import calc_quadrant

# list of simulations to consider
slist = ["cr0.50_384", "cr1.00_384"]
dsim = "/home/bgreene/simulations/SBL/"
dncall = [f"{dsim}{name}/output/netcdf/" for name in slist]

# simulation timesteps to consider
t0 = 1080000
t1 = 1260000
dt = 1000
delta_t = 0.02
fstats = "mean_stats_xyt_9-10h.nc"
timesteps = np.arange(t0, t1+1, dt, dtype=np.int32)

# list of inputs to calc_quadrant
var_pairs = [("u_rot","w"), ("theta","w")]
svarlist = ["uw", "tw"]

for dnc in dncall:
    # determine filenames
    fall = [f"{dnc}all_{tt:07d}_rot.nc" for tt in timesteps]
    # load files
    s = load_stats(dnc+fstats)
    # call calc_quadrant
    calc_quadrant(dnc, fall, s, var_pairs, svarlist)