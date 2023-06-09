#!/home/bgreene/anaconda3/bin/python
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
from LESutils import load_full
from spec import nc_LCS

# list of simulations to consider
slist = ["cr0.10_u08_192", "cr0.25_u08_192", "cr0.33_u08_192",
         "cr0.50_u08_192", "cr1.00_u08_192"]
dsim = "/home/bgreene/simulations/"
dncall = [f"{dsim}{stab}/output/netcdf/" for stab in slist]
# simulation timesteps to consider
t0 = 1080000
t1 = 1260000
dt = 1000
delta_t = 0.02
fstats = "mean_stats_xyt_9-10h.nc"

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
    # load data
    df, s = load_full(dnc, t0, t1, dt, delta_t, True, fstats, True)
    # call nc_LCS
    nc_LCS(dnc, df, s.h, zzi_list, const_zr_varlist, const_zr_savelist,
           all_zr_pairs, all_zr_savelist)