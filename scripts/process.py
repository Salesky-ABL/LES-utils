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
from LESutils import sim2netcdf, calc_stats, calc_stats_long, timeseries2netcdf

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

print("process.py complete!")