#!/home/bgreene/anaconda3/bin/python
# --------------------------------
# Name: spec.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 1 Febuary 2023
# Purpose: collection of functions for use in
# spectral analysis of LES output including:
# autocorrelation, spectral interpolation, power spectral density
# --------------------------------
import xrft
import xarray as xr
import numpy as np
from scipy.fft import fft, ifft, fft2, ifft2, fftshift
from dask.diagnostics import ProgressBar
from scipy.signal import detrend
# --------------------------------
# Begin Defining Functions
# --------------------------------
def autocorr_1d(dnc, df, detrend="constant"):
    """Input 4D xarray Dataset with loaded LES data to calculate
    autocorrelation function along x-direction, then average in
    y and time. Calculate for u, v, w, theta, u_rot, v_rot.
    Save netcdf file in dnc.

    :param str dnc: absolute path to netcdf directory for saving new file
    :param Dataset df: 4d (time,x,y,z) xarray Dataset for calculating
    :param str detrend: how to detrend along x-axis before processing,\
        default="constant" (also accepts "linear")
    """
    # construct Dataset to save
    Rsave = xr.Dataset(data_vars=None, attrs=df.attrs)
    # add additional attr for detrend_first
    Rsave.attrs["detrend_type"] = str(detrend)

    # variables to loop over for calculations
    vall = ["u", "v", "w", "theta", "u_rot", "v_rot"]

    # loop over variables
    for v in vall:
        # grab data
        din = df[v]
        # detrend
        # subtract mean by default, or linear if desired
        dfluc = xrft.detrend(din, dim="x", detrend_type=detrend)
        # normalize by standard deviation
        dnorm = dfluc / dfluc.std(dim="x")
        # calculate PSD using xrft
        PSD = xrft.power_spectrum(dnorm, dim="x", true_phase=True, 
                                  true_amplitude=True)
        # take real part of ifft to return ACF
        R = xrft.ifft(PSD, dim="freq_x", true_phase=True, true_amplitude=True, 
                      lag=0).real
        # average in y and time
        R_ytavg = R.mean(dim=("y","time"))
        # store in Rsave
        Rsave[v] = R_ytavg
    
    # save nc file
    fsave = f"{dnc}R_1d.nc"
    print(f"Saving file: {fsave}")
    with ProgressBar():
        Rsave.to_netcdf(fsave, mode="w")
    
    print("Finished computing 1d autocorrelation functions!")
    return
# --------------------------------
def autocorr_2d(dnc, df, timeavg=True):
    """Input 4D xarray Dataset with loaded LES data to calculate
    2d autocorrelation function in x-y planes, then average in
    time (if desired). Calculate for u, v, w, theta, u_rot, v_rot.
    Save netcdf file in dnc.

    :param str dnc: absolute path to netcdf directory for saving new file
    :param Dataset df: 4d (time,x,y,z) xarray Dataset for calculating
    :param bool timeavg: flag to average acf in time, default=True
    """
    # construct Dataset to save
    Rsave = xr.Dataset(data_vars=None, attrs=df.attrs)
    # variables to loop over for calculations
    vall = ["u", "v", "w", "theta", "u_rot", "v_rot"]

    # loop over variables
    for v in vall:
        # grab data
        din = df[v]
        # subtract x,y mean
        dfluc = xrft.detrend(din, dim=("x","y"), detrend_type="constant")
        # normalize by standard deviation
        dnorm = dfluc / dfluc.std(dim=("x","y"))
        # calculate PSD using xrft
        PSD = xrft.power_spectrum(dnorm, dim=("x","y"), true_phase=True, 
                                  true_amplitude=True)
        # take real part of ifft to return ACF
        R = xrft.ifft(PSD, dim=("freq_x","freq_y"), true_phase=True, 
                      true_amplitude=True, lag=(0,0)).real
        # average in time if desired, then store in Rsave
        if timeavg:
            Rsave[v] = R.mean(dim=("time"))
        else:
            Rsave[v] = R
    
    # save nc file
    fsave = f"{dnc}R_2d.nc"
    print(f"Saving file: {fsave}")
    with ProgressBar():
        Rsave.to_netcdf(fsave, mode="w")
    
    print("Finished computing 2d autocorrelation functions!")
    return