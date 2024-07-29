#!/glade/work/bgreene/conda-envs/LES/bin/python
# --------------------------------
# Name: process_pdf.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 8 November 2023
# Purpose: loop over simulations and calculate 1d pdfs of u, w, theta
# from functions in spec.py
# --------------------------------
import sys
sys.path.append("..")
import os
import numpy as np
from argparse import ArgumentParser
from LESutils import load_stats
from spec import get_1d_hist

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

# list of z/zj values to process
zh = [0.1, 0.25, 0.5, 0.9]
# determine filenames
fall = [f"{dnc}all_{tt:07d}_rot.nc" for tt in timesteps]
# load stats file
s = load_stats(dnc+fstats)
# call get_1d_hist
get_1d_hist(dnc, fall, s, s.h, zh)