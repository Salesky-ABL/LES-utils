#!/home/bgreene/anaconda3/bin/python
# --------------------------------
# Name: process_condavg.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 8 June 2023
# Purpose: loop over simulations and calculate conditional averages
# using functions from LESutils and spec
# --------------------------------
import sys
sys.path.append("..")
from LESutils import load_stats
from spec import cond_avg

# list of simulations to consider
# slist = ["cr0.10_u08_192", "cr0.25_u08_192", "cr0.33_u08_192",
        #  "cr0.50_u08_192", "cr1.00_u08_192"]
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

# inputs to cond_avg
cond_var = "u_rot"
cond_thresh = -2.0
zzi = 0.05
varlist = ["u_rot", "w", "theta"]
svarlist = ["u", "w", "t"]

# loop over simulations
for dnc in dncall:
    # load stats file
    s = load_stats(dnc+fstats, SBL=True)
    # grab values from s
    cond_scale = s.ustar0
    varscale_list = [s.ustar0, s.ustar0, s.tstar0]
    cond_jz = abs((s.z/s.h).values - zzi).argmin()
    # call cond_avg
    cond_avg(dnc, t0, t1, dt, use_rot, s, cond_var, cond_thresh, cond_jz,
             cond_scale, varlist, varscale_list, svarlist)

