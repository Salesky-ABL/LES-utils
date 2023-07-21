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
from LESutils import load_full
from spec import calc_quadrant

# list of simulations to consider
# slist = ["cr0.10_u08_192", "cr0.25_u08_192", "cr0.33_u08_192",
#          "cr0.50_u08_192", "cr1.00_u08_192"]
slist = ["cr0.50_u08_240"]
dsim = "/home/bgreene/simulations/"
dncall = [f"{dsim}{stab}/output/netcdf/" for stab in slist]

# simulation timesteps to consider
t0 = 1080000
t1 = 1260000
dt = 1000
delta_t = 0.02
fstats = "mean_stats_xyt_9-10h.nc"
use_rot = True

# list of inputs to calc_quadrant
var_pairs = [("u_rot","w"), ("theta","w")]
svarlist = ["uw", "tw"]

for dnc in dncall:
    # load files
    dd = load_full(dnc, t0, t1, dt, delta_t, SBL=True, stats=None, rotate=True)
    # call calc_quadrant
    calc_quadrant(dnc, dd, var_pairs, svarlist)