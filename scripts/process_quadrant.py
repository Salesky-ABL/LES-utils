#!/glade/work/bgreene/conda-envs/LES/bin/python
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
import os
import numpy as np
from argparse import ArgumentParser
from LESutils import load_stats
from spec import calc_quadrant

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

# simulation timesteps to consider
t0 = 900000
t1 = 1260000
dt = 2000
fstats = "mean_stats_xyt_8-10h.nc"
timesteps = np.arange(t0, t1+1, dt, dtype=np.int32)

# list of inputs to calc_quadrant
var_pairs = [("u_rot","w"), ("theta","w")]
svarlist = ["uw", "tw"]

# determine filenames
fall = [f"{dnc}all_{tt:07d}_rot.nc" for tt in timesteps]
# load files
s = load_stats(dnc+fstats)
# call calc_quadrant
calc_quadrant(dnc, fall, s, var_pairs, svarlist)