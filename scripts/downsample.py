#!/home/bgreene/anaconda3/envs/LES/bin/python
# --------------------------------
# Name: downsample.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 2 May 2024
# Purpose: compute mean profile and spectral statistics with varying
# file output frequencies to test for optimal frequency of files
# so that we can save storage
# --------------------------------
import os
import sys
sys.path.append("..")
import numpy as np
import xarray as xr
from LESutils import load_stats, calc_stats
from spec import spectrogram, autocorr_1d, calc_lengthscale
from multiprocessing import Process

# define function to call calc_stats with varying intervals
def calc_stat_intervals(dnc, t0, t1, dt_list):
    """Call calc_stats() with varying intervals of file outputs
    to determine effect on final mean profiles
    -Input-
    dnc: absolute path to netcdf directory
    t0: starting timestep
    t1: final timestep
    dt: list of number of files between timesteps
    -Output-
    stats file with format 'mean_stats_xyt_<dt>.nc
    """
    print(f"Begin {dnc}")
    for dt in dt_list:
        print(f"using file output frequency of: {dt}")
        # determine list of file timesteps from input
        timesteps = np.arange(t0, t1+1, dt, dtype=np.int32)
        # determine files to read from timesteps
        fall = [f"{dnc}all_{tt:07d}.nc" for tt in timesteps]
        # construct params for essential info to run
        params = dict(dnc=dnc, nhr_s=str(dt))
        # call calc_stats
        calc_stats(f_use=fall, **params)
    
    return
# ---------------------------------------------
# want to compute spectrogram, acf, and lengthscales for a given sim
def process_spec(dnc, t0, t1, dt_all, fstats_all):
    """Purpose: for given simulation, run spectrogram(), autocorr_1d(), and
    calc_lengthscale(). Streamline calling these functions to pass to multiple
    Processes at once for a given list of simulations.
    :param str dnc: absolute path to netcdf directory
    :param int t0: starting timestep
    :param int t1: ending timestep
    :param int dt: number of files between timesteps
    :param str fstats: filename of mean stats to be loaded
    """
    sname = dnc.split('/')[-1]
    print(f"Processing simulation: {sname}")
    for dt, fstats in zip(dt_all, fstats_all):
        # load stats file
        s = load_stats(dnc+fstats)
        # determine list of file timesteps from input
        timesteps = np.arange(t0, t1+1, dt, dtype=np.int32)
        # determine files to read from timesteps
        # use rotated fields
        fall = [f"{dnc}all_{tt:07d}_rot.nc" for tt in timesteps]
        # call spectrogram
        print("Calling spectrogram()")
        spectrogram(dnc, fall, s, detrend="constant", dt=dt)
        print(f"Finished spectrogram!")
        # call autocorr_1d
        print("Calling autocorr_1d()")
        autocorr_1d(dnc, fall, s, detrend="constant", dt=dt)
        print(f"Finished autocorr_1d!")
        # load resulting file
        R = xr.load_dataset(f"{dnc}R_1d_{dt}.nc")
        # call calc_lengthscale
        print("Calling calc_lengthscale()")
        calc_lengthscale(dnc, R, dt=dt)
        print(f"Finished calc_lengthscale!")
        # complete
        print(f"Finished processing {sname}!")
        return

if __name__ == "__main__":
    # define storage directory
    d0 = "/home/bgreene/simulations/CBL/"
    # define sims
    sims = ["u15_tw03_qw01_dq-08", "u15_tw02_qw01_dq+10",
            "u01_tw24_qw10_dq-06", "u01_tw24_qw10_dq-02"]
    # common sim info
    t0_ = 450000
    t1_ = 540000
    dt_ = [10000]
    fstats_ = [f"mean_stats_xyt_{jt}.nc" for jt in dt_]
    nsim = len(sims)

    # functions do not take more than 1-2% of yeti's memory so can call
    # as many Processes as there are sims
    proc_all = [Process(target=process_spec,
                        args=(f"{d0}{sim}/", t0_, t1_, dt_, fstats_))\
                for sim in sims]
    # proc_all = [Process(target=calc_stat_intervals,
    #                     args=(f"{d0}{sim}/", t0_, t1_, dt_))\
    #             for sim in sims]
    # start processes
    [p.start() for p in proc_all]
    # join processes
    [p.join for p in proc_all]