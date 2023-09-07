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

def update_log(dnc, fstats, dlog, flog):
    """Purpose: add new simulation to log file, sims.yaml
    -Inputs-
    dnc: string, directory path to simulation netcdf files
    fstats: string, stats file name passed to load_stats
    dlog: string, name of log file directory
    flog: string, name of log file (YAML)
    -Outputs-
    none; updates sims.yaml
    """
    # define some constants
    Lv = 2.5e6 # J / kg
    cp = 1004.  # J / kg K
    # load stats file
    s = load_stats(dnc+fstats)
    print(f"Updating log for simulation: {s.simlabel}")
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
    qw_angle = np.arctan(qw_ratio) * 180. / np.pi
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
    oldlog[s.simlabel] = newlog
    # save out updated oldlog
    with open(dlog+flog, "w") as fw:
        yaml.dump(oldlog, fw)

    return
# --------------------------------
def getsims(simdir, flog, ziL_in, phi_wq_in, Ef_in):
    """Purpose: scan through log file to find simulations that match
    the desired values of ziL, phi_wq, and Ef within a default tolerance
    -Inputs-
    simdir: string, base directory where simulations and log file are saved
    flog: string, name of log file to load
    ziL_in: float, value(s) of global stability desired
    phi_wq: float, value(s) of entrainment flux ratio angle desired
    Ef: float, value(s) of evaporative fraction desired
    -Outputs-
    simlist: list of strings with simulation names matching desired qualities
    """
    # first load yaml log file
    print(f"Reading file: {simdir}{flog}")
    with open(simdir+flog) as f:
        log = yaml.safe_load(f)

    # check ziL, phi_wq, and Ef to see if they are single values or a list
    # if not, make into a list so looping syntax still works
    if not (isinstance(ziL_in, list) | isinstance(ziL_in, np.ndarray)):
        ziL_in = [ziL_in]
    if not (isinstance(phi_wq_in, list) | isinstance(phi_wq_in, np.ndarray)):
        phi_wq_in = [phi_wq_in]
    if not (isinstance(Ef_in, list) | isinstance(Ef_in, np.ndarray)):
        Ef_in = [Ef_in]
    
    # first find list of sims that match ziL_in
    ziL_match = []
    # define ziL tolerance: +/- 20%
    ziL_tol = 0.20
    # begin looping over input values of ziL
    for ziL in ziL_in:
        # get ranges of tolerance
        ziL_lo = (1. - ziL_tol) * ziL
        ziL_hi = (1. + ziL_tol) * ziL
        # loop over sims
        for sim in log.keys():
            # grab this value of ziL and take abs
            ziL_sim = abs(log[sim]["ziL"])
            # compare this value of ziL with tolerance
            if ((ziL_sim >= ziL_lo) & (ziL_sim <= ziL_hi)):
                # store sim name in ziL_match
                ziL_match.append(sim)
    # check if ziL_match is empty
    if len(ziL_match) < 1:
        print("No matching sims found for any input values of -zi/L.")
        print("Try different input values or adjust tolerance of search.")
        print("Returning.")
        return ziL_match

    # search through ziL_match to find sims that match phi_wq_in
    phi_wq_match = []
    # define phi_wq tolerance: +/- 20%
    phi_wq_tol = 0.20
    # begin looping over input values of phi_wq
    for phi_wq in phi_wq_in:
        # get ranges of tolerance
        phi_wq_lo = (1. - phi_wq_tol) * abs(phi_wq)
        phi_wq_hi = (1. + phi_wq_tol) * abs(phi_wq)
        # loop over sims in ziL_match
        for sim in ziL_match:
            # grab this value of phi_wq
            phi_wq_sim = abs(log[sim]["phi_qw"])
            # compare with tolerance
            if ((phi_wq_sim >= phi_wq_lo) & (phi_wq_sim <= phi_wq_hi)):
                phi_wq_match.append(sim)
    # check if phi_wq_match is empty
    if len(phi_wq_match) < 1:
        print("No matching sims found for any input values of phi_wq and -zi/L.")
        print("Try different input values or adjust tolerance of search.")
        print("Returning.")
        return phi_wq_match
    
    # finally, scan through phi_wq_match to find sims that match values of Ef
    Ef_match = []
    # define Ef tolerance: +/- 10%
    Ef_tol = 0.10
    # begin looping over input values of Ef
    for Ef in Ef_in:
        # get ranges of tolerance
        Ef_lo = (1. - Ef_tol) * Ef
        Ef_hi = (1. + Ef_tol) * Ef
        # loop over sims in phi_wq_match
        for sim in phi_wq_match:
            # grab this value of Ef
            Ef_sim = log[sim]["Ef"]
            # compare with tolerance
            if ((Ef_sim >= Ef_lo) & (Ef_sim <= Ef_hi)):
                Ef_match.append(sim)
    # check if Ef_match is empty
    if len(Ef_match) < 1:
        print("No matching sims found for any input values of Ef, phi_wq, and -zi/L.")
        print("Try different input values or adjust tolerance of search.")
        print("Returning.")
    
    # return Ef_match as final list of sims matching combinations of input
    print(f"Total matches found: {len(Ef_match)}")
    return Ef_match