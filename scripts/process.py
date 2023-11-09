#!/glade/work/bgreene/conda-envs/LES/bin/python
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
from LESutils import process_raw_sim
from getsims import update_log

# load yaml file
fyaml = "/glade/u/home/bgreene/LES-utils/scripts/process.yaml"
with open(fyaml) as f:
    config = yaml.safe_load(f)

# loop over simulations in config
for sim in config["simlist"]:
    print(f"----- Begin processing simulation: {sim} -----")
    # construct dout
    dout = config["d0"] + sim + "/output/"
    # run process
    fstats = process_raw_sim(dout=dout, nhr=config["nhr"], 
                             nhr_s=config["nhr_s"],
                             del_raw=config["del_raw"], 
                             organize=config["organize"],
                             nrun=config["nrun"],
                             overwrite=config["overwrite"], 
                             cstats=config["cstats"], 
                             rotate=config["rotate"],
                             ts2nc=config["ts2nc"],
                             del_remaining=config["del_remaining"])
    # update sims.yaml with resulting stats file
    # if fstats is not None:
        # update_log(dout+"netcdf/", fstats, config["d0"], "sims.yaml")
        
    print(f"Finished processing simulation: {sim}")

print("process.py complete!")