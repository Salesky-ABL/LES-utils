#!/glade/work/bgreene/conda-envs/LES/bin/python
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
import os
import numpy as np
from argparse import ArgumentParser
from LESutils import load_stats
from spec import nc_LCS

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
# determine filenames
fall = [f"{dnc}all_{tt:07d}_rot.nc" for tt in timesteps]
# load data
s = load_stats(dnc+fstats)
# call nc_LCS
nc_LCS(dnc, fall, s, zzi_list, const_zr_varlist, const_zr_savelist,
       all_zr_pairs, all_zr_savelist)