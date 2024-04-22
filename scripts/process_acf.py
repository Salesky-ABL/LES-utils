#!/home/bgreene/anaconda3/envs/LES/bin/python
# --------------------------------
# Name: process_acf.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 1 November 2023
# Purpose: loop over simulations and calculate 2d acf using functions
# from LESutils and spec
# 14 Feb 2024: calc 1d autocorr and lengthscales
# --------------------------------
import sys
sys.path.append("..")
import os
import numpy as np
import xarray as xr
from argparse import ArgumentParser
from LESutils import load_stats
from spec import autocorr_2d, autocorr_1d, calc_lengthscale

# arguments for simulation directory to process
parser = ArgumentParser()
parser.add_argument("-d", required=True, action="store", dest="dsim", nargs=1,
                    help="Simulation base directory")
parser.add_argument("-s", required=True, action="store", dest="sim", nargs=1,
                    help="Simulation name")
args = parser.parse_args()

# construct simulation directory and ncdir
dnc = os.path.join(args.dsim[0], args.sim[0]) + os.sep
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
# call autocorr_2d
# autocorr_2d(dnc, fall, s, timeavg=True)
# call autocorr_1d
autocorr_1d(dnc, fall, s, detrend="constant")
# load the resulting file
R = xr.load_dataset(dnc+"R_1d.nc")
# call calc_lengthscale
calc_lengthscale(dnc, R)