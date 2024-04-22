#!/home/bgreene/anaconda3/envs/LES/bin/python
# --------------------------------
# Name: process_LCS.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 20 July 2023
# Purpose: loop over simulations and calculate spectrograms 
# using functions from LESutils
# --------------------------------
import sys
sys.path.append("..")
import os
import numpy as np
from argparse import ArgumentParser
from LESutils import load_stats
from spec import spectrogram

# arguments for simulation directory to process
parser = ArgumentParser()
parser.add_argument("-d", required=True, action="store", dest="dsbl", nargs=1,
                    help="Simulation base directory")
parser.add_argument("-s", required=True, action="store", dest="sim", nargs=1,
                    help="Simulation name")
args = parser.parse_args()

# construct simulation directory and ncdir
dnc = os.path.join(args.dsbl[0], args.sim[0]) + os.sep
# dnc = f"{dout}netcdf/"
# simulation timesteps to consider
t0 = 450000
t1 = 540000
dt = 1000
fstats = "mean_stats_xyt_5-6h.nc"
timesteps = np.arange(t0, t1+1, dt, dtype=np.int32)

# determine filenames
fall = [f"{dnc}all_{tt:07d}_rot.nc" for tt in timesteps]
# load data
s = load_stats(dnc+fstats)
# call spectrogram
spectrogram(dnc, fall, s, detrend="constant")