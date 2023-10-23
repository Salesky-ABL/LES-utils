#!/home/bgreene/anaconda3/envs/LES/bin/python
# --------------------------------
# Name: process_LCS.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 8 June 2023
# Purpose: loop over simulations and calculate LCS using functions
# from LESutils
# --------------------------------
import sys
sys.path.append("..")
import numpy as np
from LESutils import load_stats
from spec import nc_LCS

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

# list of variables to process
# constant reference height (same variable)
const_zr_varlist = ["u_rot", "w", "theta"]
const_zr_savelist = ["u", "w", "t"]
# all reference heights (comparing variables)
all_zr_pairs = [("u_rot", "w"), ("theta", "w"), ("theta", "u_rot")]
all_zr_savelist = ["uw", "tw", "tu"]
# desired reference heights
zzi_list = [0.]
# begin looping over simulations
for dnc in dncall:
    # determine filenames
    fall = [f"{dnc}all_{tt:07d}_rot.nc" for tt in timesteps]
    # load data
    s = load_stats(dnc+fstats)
    # call nc_LCS
    nc_LCS(dnc, fall, s, zzi_list, const_zr_varlist, const_zr_savelist,
           all_zr_pairs, all_zr_savelist)