#!/home/bgreene/anaconda3/envs/LES/bin/python
# --------------------------------
# Name: process_timeseries.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 22 October 2023
# Purpose: Script that combines timeseries npz files into one netcdf
# --------------------------------
import os
import yaml
import sys
sys.path.append("..")
from multiprocessing import Process
from LESutils import timeseries2netcdf

# define function that pre-processes metadata
def tsPreprocess(dout):
    # load yaml file
    with open(dout+"params.yaml") as fp:
        params = yaml.safe_load(fp)
    # add simulation label to params
    params["simlabel"] = params["path"].split(os.sep)[-2]
    # add netcdf path to params
    dnc = f"{dout}netcdf/"
    params["dnc"] = dnc
    # now can call timeseries2netcdf
    timeseries2netcdf(dout, del_npz=False, **params)
    print(f"Finished processing {params['simlabel']}")
    return

# main call
if __name__ == "__main__":
    # define storage directory
    d0 = "/home/bgreene/simulations/SBL/"
    # initiate concurrent processes for each sim
    process1 = Process(target=tsPreprocess, 
                       args=(d0+"cr0.50_384/output/",))
    process2 = Process(target=tsPreprocess, 
                       args=(d0+"cr1.00_384/output/",))
    # begin each process
    process1.start()
    process2.start()
    # join each process
    process1.join()
    process2.join()
