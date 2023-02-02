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
from numpy.fft import fft, ifft
from dask.diagnostics import ProgressBar
# --------------------------------
# Begin Defining Functions
# --------------------------------
def autocorr_1d(dnc, df):
    """Input 4D xarray Dataset with loaded LES data to calculate
    autocorrelation function along x-direction, then average in
    y and time. Calculate for u, v, w, theta, u_rot, v_rot.
    Save netcdf file in dnc.

    :param str dnc: absolute path to netcdf directory for saving new file
    :param Dataset df: 4d (time,x,y,z) xarray Dataset for calculating
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
            # empty PSD array
            PSD = np.zeros((df.nx, df.ny, df.nz), dtype=np.float64)
            # grab data for processing
            d = df[v].isel(time=jt).to_numpy()
            # forward FFT
            f = fft(d, axis=0)
            # calculate PSD
            for jx in range(1, df.nx//2):
                PSD[jx,:,:] = np.real( f[jx,:,:] * np.conj(f[jx,:,:]) )
                PSD[df.nx-jx,:,:] = np.real( f[df.nx-jx,:,:] * np.conj(f[df.nx-jx,:,:]) )
            # normalize by variance
            PSD /= np.var(d, axis=0)
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