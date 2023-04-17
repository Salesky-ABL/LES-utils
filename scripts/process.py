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
from LESutils import sim2netcdf, calc_stats, calc_stats_long, timeseries2netcdf, load_full
from spec import autocorr_1d, autocorr_2d, spectrogram

# load yaml file
fyaml = "/home/bgreene/LES-utils/scripts/process.yaml"
with open(fyaml) as f:
    config = yaml.safe_load(f)

# run functions based on config params
# sim2netcdf
if config["sim2nc"]:
    print("Begin sim2netcdf...")
    sim2netcdf(config["dout"], config["dnc"], config["res"], config["dim"], 
               config["scales"], config["t0"], config["t1"], config["dt"], 
               config["use_dissip"], config["use_q"], config["simlab"])
    print("Finished sim2netcdf!")

# calc_stats
if config["calcstats"]:
    print("Begin calc_stats...")
    calc_stats(config["dnc"], config["t0"], config["t1"], config["dt"], 
               config["delta_t"], config["use_dissip"], config["use_q"], 
               config["detrend"], config["tavg"])
    print("Finished calc_stats!")

# calc_stats_long
if config["statslong"]:
    print("Begin calc_stats_long...")
    calc_stats_long(config["dnc"], config["t0"], config["t1"], config["dt"],
                    config["delta_t"], config["use_dissip"], config["use_q"],
                    config["tavg"])
    print("Finished calc_stats!")
    
# timeseries2netcdf
if config["ts2nc"]:
    print("Begin timeseries2netcdf...")
    timeseries2netcdf(config["dout"], config["dnc"], config["scales"],
                      config["use_q"], config["delta_t"], config["res"][2], 
                      config["dim"][2], config["nhr"], config["tf"], 
                      config["simlab"])
    print("Finished timeseries2netcdf!")

#
# Spectral analysis functions - requires loading files first
#
if (config["ac1d"] or config["ac2d"] or config["spec"]):
    dd, s = load_full(config["dnc"], config["t0"], config["t1"], config["dt"],
                      config["delta_t"], SBL=config["SBL"], 
                      stats=config["fstats"])

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
    spectrogram(config["dnc"], dd)
    print("Finished spectrogram!")

print("process.py complete!")