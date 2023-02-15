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
import xarray as xr
import numpy as np
from scipy.fft import fft, ifft, fft2, ifft2
from dask.diagnostics import ProgressBar
from scipy.signal import detrend
# --------------------------------
# Begin Defining Functions
# --------------------------------
def autocorr_1d(dnc, df, detrend_first=False):
    """Input 4D xarray Dataset with loaded LES data to calculate
    autocorrelation function along x-direction, then average in
    y and time. Calculate for u, v, w, theta, u_rot, v_rot.
    Save netcdf file in dnc.

    :param str dnc: absolute path to netcdf directory for saving new file
    :param Dataset df: 4d (time,x,y,z) xarray Dataset for calculating
    :param bool detrend_first: linear detrend along x-axis before processing,\
        default=False
    """
    # variables to loop over for calculations
    vall = ["u", "v", "w", "theta", "u_rot", "v_rot"]
    # define dictionary of empty arrays to store data
    acf_all = {}
    for v in vall:
        acf_all[v] = np.zeros((df.nx, df.nz), dtype=np.float64)

    # BIG LOOP over time
    for jt in range(df.time.size):
        # loop over variables
        for v in vall:
            # grab data for processing
            din = df[v].isel(time=jt).to_numpy()
            # detrend
            if detrend_first:
                d = detrend(din, axis=0, type="linear")
                d /= d.std(axis=0)
            else:
                d = (din - din.mean(axis=0)) / d.std(axis=0)
            # forward FFT
            f = fft(d, axis=0)
            # calculate PSD
            PSD = np.abs(f) ** 2.
            # ifft to get acf
            R = np.real( ifft(PSD, axis=0) ) / df.nx
            # calculate mean along y-axis and assign to acf_all
            acf_all[v] += np.mean(R, axis=1)
    
    # loop over vall and normalize by nt to get time average
    for v in vall:
        acf_all[v] /= df.time.size

    # construct Dataset to save
    Rsave = xr.Dataset(data_vars=None, coords=dict(x=df.x, z=df.z), 
                       attrs=df.attrs)
    # add additional attr for detrend_first
    Rsave.attrs["detrend_first"] = detrend_first
    # loop over vars in vall and store
    for v in vall:
        Rsave[v] = xr.DataArray(data=acf_all[v], dims=("x","z"), 
                                coords=dict(x=Rsave.x, z=Rsave.z))
    
    # save nc file
    fsave = f"{dnc}R_1d.nc"
    print(f"Saving file: {fsave}")
    with ProgressBar():
        Rsave.to_netcdf(fsave, mode="w")
    
    print("Finished computing autocorrelation functions!")
    return
# --------------------------------
def autocorr_2d(dnc, df):
    """Input 4D xarray Dataset with loaded LES data to calculate
    2d autocorrelation function in x-y planes, then average in
    time. Calculate for u, v, w, theta, u_rot, v_rot.
    Save netcdf file in dnc.

    :param str dnc: absolute path to netcdf directory for saving new file
    :param Dataset df: 4d (time,x,y,z) xarray Dataset for calculating
    """
    # variables to loop over for calculations
    vall = ["u", "v", "w", "theta", "u_rot", "v_rot"]
    # define dictionary of empty arrays to store data
    acf_all = {}
    for v in vall:
        acf_all[v] = np.zeros((df.nx, df.ny, df.nz), dtype=np.float64)

    # BIG LOOP over time
    for jt in range(df.time.size):
        # loop over variables
        for v in vall:
            # grab data for processing
            din = df[v].isel(time=jt).to_numpy()
            # normalize
            d = (din - din.mean(axis=(0,1))) / d.std(axis=(0,1))
            # forward FFT
            f = fft2(d, axes=(0,1))
            # calculate PSD
            PSD = np.abs(f) ** 2.
            # ifft to get acf
            R = np.real( ifft(PSD, axes=(0,1)) ) / df.nx / df.ny
            # calculate mean along y-axis and assign to acf_all
            acf_all[v] += R
    
    # loop over vall and normalize by nt to get time average
    for v in vall:
        acf_all[v] /= df.time.size

    # construct Dataset to save
    Rsave = xr.Dataset(data_vars=None, coords=dict(x=df.x, y=df.y, z=df.z), 
                       attrs=df.attrs)
    # loop over vars in vall and store
    for v in vall:
        Rsave[v] = xr.DataArray(data=acf_all[v], dims=("x","y","z"), 
                                coords=dict(x=Rsave.x, y=Rsave.y, z=Rsave.z))
    
    # save nc file
    fsave = f"{dnc}R_2d.nc"
    print(f"Saving file: {fsave}")
    with ProgressBar():
        Rsave.to_netcdf(fsave, mode="w")
    
    print("Finished computing autocorrelation functions!")
    return