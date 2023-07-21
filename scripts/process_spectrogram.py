#!/home/bgreene/anaconda3/bin/python
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
from LESutils import load_full
from spec import spectrogram

# list of simulations to consider
slist = ["u15_tw10_qw04_dry3", "u15_tw10_qw04_moist4", 
         "u10_tw20_qw08_dry2", "u01_tw03_qw01_dry3",
         "u01_tw24_qw10_dry2", "u01_tw24_qw10_moist4",
         "u15_tw10_qw01_dry", "u15_tw10_qw01_moist2",
         "u10_tw20_qw02_dry", "u10_tw20_qw02_moist2"]
# slist = ["u15_tw10_qw04_dry3", "u15_tw10_qw04_moist4",
#          "u15_tw10_qw01_dry", "u15_tw10_qw01_moist2"]
# slist = ["u10_tw20_qw08_dry2", "u01_tw03_qw01_dry3",
#          "u01_tw24_qw10_dry2", "u01_tw24_qw10_moist4",
#          "u10_tw20_qw02_dry", "u10_tw20_qw02_moist2"]

dsim = "/home/bgreene/simulations/"
dncall = [f"{dsim}{name}/output/netcdf/" for name in slist]
# simulation timesteps to consider
t0 = 360000
t1 = 432000
dt = 1000
delta_t = 0.05
SBL = False
use_q = True
fstats = "mean_stats_xyt_5-6h.nc"

# begin looping over simulations
for dnc in dncall:
    # load data
    df, s = load_full(dnc, t0, t1, dt, delta_t, SBL=SBL, 
                      stats=fstats, rotate=True)
    # call spectrogram
    spectrogram(dnc, df, detrend="constant", use_q=use_q)