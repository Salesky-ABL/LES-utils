# --------------------------------
# Name: process_ampmod.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 12 June 2023
# Purpose: loop over simulations and calculate amplitude modulation
# coefficients using LESutils code
# --------------------------------
import sys
sys.path.append("..")
from LESutils import load_stats, load_timeseries
from spec import amp_mod

# list of simulations to consider
# slist = ["cr0.10", "cr0.25", "cr0.33", "cr0.50", "cr1.00"]
# dsim = "/home/bgreene/simulations/SBL/"
# dncall = [f"{dsim}{stab}_384/output/netcdf/" for stab in slist]
# fstats = "mean_stats_xyt_8-10h.nc"
slist = ["cr0.10_192_384", "cr0.25_192_384", 
        "cr0.33_192_384", "cr0.50_192_384", "cr1.00_192_384"]
dsim = "/home/bgreene/simulations/SBL_JFM/"
dncall = [f"{dsim}{stab}/" for stab in slist]
fstats = "mean_stats_xyt_8-9h.nc"

# cutoff frequency: fraction of CBL depth
fc_frac = 0.25

for jd, dnc in enumerate(dncall):
    s = load_stats(dnc+fstats)
    detrend = True
    tlab = "1h"
    tuse = None

    ts = load_timeseries(dnc, detrend=detrend, tlab=tlab, tuse=tuse)
    print(f"Begin amp_mod for {slist[jd]}")
    amp_mod(dnc, ts, fc_frac*s.h.values)
    print("Finished amp_mod!")