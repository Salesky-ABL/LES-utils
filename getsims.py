#!/home/bgreene/anaconda3/envs/LES/bin/python
# --------------------------------
# Name: getsims.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 31 August 2023
# Purpose: series of functions for creating and reading from a
# file that tracks salient information about CBL simulations
# --------------------------------
import os
import yaml
import numpy as np
import xarray as xr
from glob import glob
from LESutils import load_stats

def update_log(dnc, fstats, dlog, flog, SBL=False):
    """Purpose: add new simulation to log file, sims.yaml
    -Inputs-
    dnc: string with directory path to simulation netcdf files
    fstats: string stats file name passed to load_stats
    dlog: string name of log file directory
    flog: string name of log file (YAML)
    SBL: boolean flag passed to load_stats (TODO: remove need for this)
    -Outputs-
    none; updates sims.txt
    """
    # define some constants
    Lv = 2.5e6 # J / kg
    cp = 1004.  # J / kg K
    # load stats file
    s = load_stats(dnc+fstats, SBL=SBL)
    # calculate important parameters for saving to log file
    # -zi/L
    ziL = -1 * (s.h / s.L).values
    # grab surface, entrianment zone values of qw
    qw_e = s.qw_cov_tot.sel(z=s.h, method="nearest").values / 1000.
    qw_s = s.qw_cov_tot.isel(z=0).values / 1000.
    # grab surface values of tw
    tw_s = s.tw_cov_tot.isel(z=0).values
    # ratio
    qw_ratio = qw_s/qw_e
    qw_angle = np.arctan(qw_ratio) * 180 / np.pi
    # evaporative fraction
    Ef = (Lv * qw_s) / ((cp*tw_s) + (Lv*qw_s))
    # Bowen ratio
    B = (cp*tw_s) / (Lv*qw_s)

    # create dictionary to hold info to save
    newlog = {}
    # update attrs for each sim to include zi/L and phi_qw
    newlog["ziL"] = float(ziL)
    newlog["phi_qw"] = float(qw_angle)
    newlog["Ef"] = float(Ef)
    newlog["B"] = float(B)
    # combine into one label
    lab1 = f"$-z_i/L={ziL:.1f}$, $\\varphi_{{qw}}={qw_angle:.1f}^\\circ$"
    newlog["lab1"] = lab1
    lab2 = f"$\\varphi_{{qw}}={qw_angle:.1f}^\\circ$, $E_f={Ef:.1f}$"
    newlog["lab2"] = lab2
    lab3 = f"$-z_i/L={ziL:.1f}$, $\\varphi_{{qw}}={qw_angle:.1f}^\\circ$, $E_f={Ef:.1f}$"
    newlog["lab3"] = lab3    

    # write info to log file
    # check if flog exists yet
    if os.path.exists(dlog+flog):
        # open file to write and load data to copy over
        with open(dlog+flog) as fo:
            oldlog = yaml.safe_load(fo)
    else:
        # create empty dictionary named oldlog
        oldlog = {}
    # add newlog to oldlog under simulation label s.label
    oldlog[s.label] = newlog
    # save out updated oldlog
    with open(dlog+flog, "w") as fw:
        yaml.dump(oldlog, fw)

    return