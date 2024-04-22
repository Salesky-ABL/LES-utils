#!/home/bgreene/anaconda3/envs/LES/bin/python
# --------------------------------
# Name: rotate_sim.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 20 October 2023
# Purpose: Script that rotates sim volume files as they rsync over
# --------------------------------
import os
import sys
sys.path.append("..")
import xarray as xr
import numpy as np
import time
from multiprocessing import Process
from LESutils import xr_rotate, nc_rotate_parallel

def rotate_and_wait(dnc, t0, t1, dt):
    """Purpose: load netcdf all_TTTTTTT.nc files and rotate coordinates
    into mean streamwise flow using rot_interp, then save netcdf file with
    new coordinates. Load one file at a time. If next file is not fully synced,
    then wait another 10 minutes.
    :param str dnc: absolute path to netcdf directory
    :param int t0: starting timestep
    :param int t1: ending timestep
    :param int dt: number of files between timesteps
    """
    # determine list of file timesteps from input
    timesteps = np.arange(t0, t1+1, dt, dtype=np.int32)
    # determine files to read from timesteps
    fall = [f"{dnc}all_{tt:07d}.nc" for tt in timesteps]
    # BEGIN TIME LOOP
    for jf, ff in enumerate(fall):
        # check if file exists
        if os.path.exists(ff):
            print(f"Processing file: {ff}")
            # load file
            d = xr.load_dataset(ff)
            # use xr_rotate to rotate this Dataset
            Dsave = xr_rotate(d)
            # save to netcdf file and continue
            fsave = f"{dnc}all_{timesteps[jf]:07d}_rot.nc"
            print(f"Saving file: {fsave.split(os.sep)[-1]}")
            Dsave.to_netcdf(fsave, mode="w")
            # delete original file to save space
            # os.system(f"rm {ff}")
        else:
            print("File does not exist; wait 10 more minutes to sync.")
            time.sleep(600)
    
    print("Finished saving all files!")
    return

# main program
if __name__ == "__main__":
    # define storage directory
    d0 = "/home/bgreene/simulations/CBL/"
    # define list of simulations that still need to be rotated
    simrot = ["u01_tw24_qw02_dq+04", "u01_tw24_qw10_dq-02",
              "u01_tw24_qw10_dq+10","u01_tw24_qw39_dq-08",
              "u04_tw20_qw08_dq-03","u08_tw24_qw10_dq-02",
              "u12_tw01_qw01_dq-08","u14_tw01_qw02_dq+08",
              "u15_tw02_qw03_dq-08","u15_tw03_qw00_dq-01",
              "u15_tw03_qw00_dq-02","u15_tw03_qw00_dq+01",
              "u15_tw03_qw00_dq+02","u15_tw10_qw04_dq-02",
              "u15_tw10_qw04_dq+06"]
    # define sim start and end times
    t0_ = 450000
    t1_ = 540000
    dt_ = 1000
    nproc_ = 15
    # loop over simulations and run nc_rotate_parallel
    for sim in simrot:
        dnc = f"{d0}{sim}/"
        print(f"Begin processing simulation: {sim}")
        tstart = time.time()
        nc_rotate_parallel(dnc, t0_, t1_, dt_, nproc_)
        elapsed = time.time() - tstart
        print("Simulation finished processing!") 
        print(f"Total time: {(elapsed)/60:.2f} min")

    # # initiate concurrent processes for each sim
    # process1 = Process(target=rotate_and_wait, 
    #                     args=(d0+"u15_tw02_qw01_dq+02/", 
    #                           t0_, t1_, dt_))
    # # process2 = Process(target=rotate_and_wait, 
    # #                    args=(d0+"u15_tw03_qw01_dry/", 
    # #                          t0_, t1_, dt_))
    # # process3 = Process(target=rotate_and_wait, 
    # #                    args=(d0+"u15_tw24_qw10_256/", 
    # #                          t0_, t1_, dt_))
    # # begin each process
    # process1.start()
    # # process2.start()
    # # process3.start()
    # # join each process
    # process1.join()
    # # process2.join()
    # # process3.join()