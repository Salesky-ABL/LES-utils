#!/home/bgreene/anaconda3/envs/LES/bin/python
# --------------------------------
# Name: process2.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 31 March 2023
# Purpose: process output from LES using functions from LESutils
# 3 Nov 2023: call nc_rotate on each sim
# 22 Apr 2024: compute spectra, acf, integral lengthscales for CBL sims
# and just hardcode the simulations; spread across numerous Processes
# --------------------------------
import sys
sys.path.append("..")
import numpy as np
import xarray as xr
from time import time
from LESutils import load_stats, calc_TKE_budget
from spec import spectrogram, autocorr_1d, calc_lengthscale
from multiprocessing import Process

# define function to map to each process
# want to compute spectrogram, acf, and lengthscales for a given sim
def process_spec(dnc, t0, t1, dt, fstats):
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
    tstart0 = time()
    # load stats file
    s = load_stats(dnc+fstats)
    # determine list of file timesteps from input
    timesteps = np.arange(t0, t1+1, dt, dtype=np.int32)
    # determine files to read from timesteps
    # use rotated fields
    fall_rot = [f"{dnc}all_{tt:07d}_rot.nc" for tt in timesteps]
    # unrotated fields
    fall = [f"{dnc}all_{tt:07d}.nc" for tt in timesteps]
    # call spectrogram
    print("Calling spectrogram()")
    tstart1 = time()
    spectrogram(dnc, fall_rot, s, detrend="constant")
    print(f"Finished spectrogram! Time elapsed: {(time()-tstart1)/60:.2f} min")
    # call autocorr_1d
    print("Calling autocorr_1d()")
    tstart2 = time()
    autocorr_1d(dnc, fall_rot, s, detrend="constant")
    print(f"Finished autocorr_1d! Time elapsed: {(time()-tstart2)/60:.2f} min")
    # load resulting file
    R = xr.load_dataset(dnc+"R_1d.nc")
    # call calc_lengthscale
    print("Calling calc_lengthscale()")
    tstart3 = time()
    calc_lengthscale(dnc, R)
    print(f"Finished calc_lengthscale! Time elapsed: {(time()-tstart3)/60:.2f} min")
    # call calc_TKE_budget
    print("Calling calc_TKE_budget()")
    tstart4 = time()
    calc_TKE_budget(dnc, fall, s)
    print(f"Finished calc_TKE_budget! Time elapsed: {(time()-tstart4)/60:.2f} min")
    # complete
    print(f"Finished processing {sname}! Total time: {(time()-tstart0)/60:.2f} min")
    return

if __name__ == "__main__":
    # define storage directory
    d0 = "/home/bgreene/simulations/CBL/"
    # define sims
    sims = [
        "u08_tw005_qw008_dq+20_dt+005",
        "u0025_tw01_qw016_dq+20_dt+005"
    ]
    # common sim info
    t0_ = 450000
    t1_ = 540000
    dt_ = 2000
    fstats_ = "mean_stats_xyt_5-6h.nc"
    nsim = len(sims)

    # functions do not take more than 1-2% of yeti's memory so can call
    # as many Processes as there are sims
    proc_all = [Process(target=process_spec,
                        args=(f"{d0}{sim}/", t0_, t1_, dt_, fstats_))\
                for sim in sims]
    # start processes
    [p.start() for p in proc_all]
    # join processes
    [p.join for p in proc_all]