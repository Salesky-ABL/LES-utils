#!/glade/work/bgreene/conda-envs/LES/bin/python
# --------------------------------
# Name: process_acf.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 1 November 2023
# Purpose: loop over simulations and calculate 2d acf using functions
# from LESutils and spec
# --------------------------------
import sys
sys.path.append("..")
import os
import numpy as np
from argparse import ArgumentParser
from LESutils import load_stats
from spec import autocorr_2d

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

# determine filenames
fall = [f"{dnc}all_{tt:07d}_rot.nc" for tt in timesteps]
# load data
s = load_stats(dnc+fstats)
# call autocorr_2d
autocorr_2d(dnc, fall, s, timeavg=True)