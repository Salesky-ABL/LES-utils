#!/home/bgreene/anaconda3/bin/python
# --------------------------------
# Name: process.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 31 March 2023
# Purpose: process output from LES using functions from LESutils
# --------------------------------
import sys
sys.path.append("..")
import yaml
import os
from time import sleep
from datetime import datetime
from LESutils import sim2netcdf, calc_stats, calc_stats_long,\
    timeseries2netcdf, load_full, load_stats, load_timeseries,\
    nc_rotate
from spec import autocorr_1d, autocorr_2d, spectrogram, amp_mod

# load yaml file
fyaml = "/home/bgreene/LES-utils/scripts/process.yaml"
with open(fyaml) as f:
    config = yaml.safe_load(f)

# while loop to check if sim complete
ffinal = f"{config['dout']}u_{config['t1']:07d}.out"
ncfinal = f"{config['dnc']}all_{config['t1']:07d}.nc"
if os.path.exists(ncfinal):
    print("Simulation nc files exist...continue.")
else:
    while not os.path.exists(ffinal):
        print(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
        print("Simulation not complete. Waiting another hour.")
        sleep(3600)

# run functions based on config params
# sim2netcdf
if config["sim2nc"]:
    print("Begin sim2netcdf...")
    sim2netcdf(config["dout"], config["dnc"], config["res"], config["dim"], 
               config["scales"], config["t0"], config["t1"], config["dt"], 
               config["use_dissip"], config["use_q"], config["simlab"],
               del_raw=config["del_raw"])
    print("Finished sim2netcdf!")

# calc_stats
if config["calcstats"]:
    print("Begin calc_stats...")
    calc_stats(config["dnc"], config["t0"], config["t1"], config["dt"], 
               config["delta_t"], config["use_dissip"], config["use_q"], 
               config["detrend"], config["tavg"], config["use_rot"])
    print("Finished calc_stats!")

# calc_stats_long
if config["statslong"]:
    print("Begin calc_stats_long...")
    calc_stats_long(config["dnc"], config["t0"], config["t1"], config["dt"],
                    config["delta_t"], config["use_dissip"], config["use_q"],
                    config["tavg"], config["use_rot"])
    print("Finished calc_stats!")
    
# timeseries2netcdf
if config["ts2nc"]:
    print("Begin timeseries2netcdf...")
    timeseries2netcdf(config["dout"], config["dnc"], config["scales"],
                      config["use_q"], config["delta_t"], config["res"][2], 
                      config["dim"][2], config["nhr"], config["tf"], 
                      config["simlab"], del_raw=config["del_raw"])
    print("Finished timeseries2netcdf!")

# nc_rotate
if config["ncrot"]:
    print("Begin nc_rotate...")
    nc_rotate(config["dnc"], config["t0"], config["t1"], config["dt"])
#
# Spectral analysis functions - requires loading files first
#
if (config["ac1d"] or config["ac2d"] or config["spec"]):
    dd, s = load_full(config["dnc"], config["t0"], config["t1"], config["dt"],
                      config["delta_t"], SBL=config["SBL"], 
                      stats=config["fstats"], rotate=config["use_rot"])

# 1d autocorrelation
if config["ac1d"]:
    print("Begin autocorr_1d...")
    autocorr_1d(config["dnc"], dd)
    print("Finished autocorr_1d!")

# 2d autocorrelation
if config["ac2d"]:
    print("Begin autocorr_2d...")
    autocorr_2d(config["dnc"], dd)
    print("Finished autocorr_2d!")

# spectrogram
if config["spec"]:
    print("Begin spectrogram...")
    spectrogram(config["dnc"], dd, use_q=config["use_q"])
    print("Finished spectrogram!")

# amplitude modulation: requires timeseries file
if config["AM"]:
    s = load_stats(config["dnc"]+config["fstats"])
    ts = load_timeseries(config["dnc"], detrend=True, 
                         tavg=f"{config['nhr']}h")
    print("Begin amp_mod...")
    amp_mod(config["dnc"], ts, 0.25*s.h.values)
    print("Finished amp_mod!")

print("process.py complete!")