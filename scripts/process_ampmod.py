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
slist = ["cr0.10_u08_240", "cr0.25_u08_192", "cr0.33_u08_192",
         "cr0.50_u08_240"]#, "cr1.00_u08_192"]
dsim = "/home/bgreene/simulations/"
dncall = [f"{dsim}{stab}/output/netcdf/" for stab in slist]
fstats = "mean_stats_xyt_9-10h.nc"

# cutoff frequency: fraction of sbl depth
fc_frac = 0.25

for jd, dnc in enumerate(dncall):
    s = load_stats(dnc+fstats, SBL=True)
    if jd in [0, 3]:
        tavg = "1h"
    else:
        tavg = "1.0h"

    ts = load_timeseries(dnc, detrend=True, tavg=tavg)
    print("Begin amp_mod...")
    amp_mod(dnc, ts, fc_frac*s.h.values)
    print("Finished amp_mod!")