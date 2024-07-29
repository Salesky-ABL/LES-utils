#!/home/bgreene/anaconda3/envs/LES/bin/python
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
import os
from argparse import ArgumentParser
from LESutils import load_stats
from spec import cond_avg

# arguments for simulation directory to process
parser = ArgumentParser()
parser.add_argument("-d", required=True, action="store", dest="dsbl", nargs=1,
                    help="Simulation base directory")
parser.add_argument("-s", required=True, action="store", dest="sim", nargs=1,
                    help="Simulation name")
args = parser.parse_args()

# construct simulation directory and ncdir
dnc = os.path.join(args.dsbl[0], args.sim[0]) + os.sep
# simulation timesteps to consider
t0 = 1440000
t1 = 1620000
dt = 2000
fstats = "mean_stats_xyt_8-9h.nc"
use_rot = True

# inputs to cond_avg
cond_var = "u_rot"
cond_thresh = -2.0
zzi = 0.05
varlist = ["u_rot", "w", "theta"]
svarlist = ["u", "w", "t"]

# load stats file
s = load_stats(dnc+fstats)
# grab values from s
cond_scale = s.ustar0
varscale_list = [s.ustar0, s.ustar0, s.tstar0]
cond_jz = abs((s.z/s.h).values - zzi).argmin()
# call cond_avg
cond_avg(dnc, t0, t1, dt, use_rot, s, cond_var, cond_thresh, cond_jz,
         cond_scale, varlist, varscale_list, svarlist)
