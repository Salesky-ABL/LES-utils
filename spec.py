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
import os
import xrft
import xarray as xr
import numpy as np
from scipy.fft import fftfreq, fftshift
from scipy.signal import hilbert
from dask.diagnostics import ProgressBar
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
# --------------------------------
def spectrogram(dnc, df, detrend="constant"):
    """Input 4D xarray Dataset with loaded LES data to calculate
    power spectral density along x-direction, then average in
    y and time. Calculate for u', w', theta', q', u'w', theta'w',
    q'w', theta'q'.
    Save netcdf file in dnc.

    :param str dnc: absolute path to netcdf directory for saving new file
    :param Dataset df: 4d (time,x,y,z) xarray Dataset for calculating
    :param str detrend: how to detrend along x-axis before processing,\
        default="constant" (also accepts "linear")
    """
    # construct Dataset to save
    Esave = xr.Dataset(data_vars=None, attrs=df.attrs)
    # add additional attr for detrend_first
    Esave.attrs["detrend_type"] = str(detrend)

    # variables to loop over for calculations
    vall = ["u_rot", "w", "theta", "q"]
    vsave = ["uu", "ww", "tt", "qq"]

    # loop over variables
    for v, vs in zip(vall, vsave):
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
        # average in y and time, take only real values
        PSD_ytavg = PSD.mean(dim=("time","y"))
        # store in Esave
        Esave[vs] = PSD_ytavg.real

    # variables to loop over for cross spectra
    vall2 = [("u_rot", "w"), ("theta", "w"), ("q", "w"), ("theta", "q")]
    vsave2 = ["uw", "tw", "qw", "tq"]

    # loop over variables
    for v, vs in zip(vall2, vsave2):
        # grab data
        din1, din2 = df[v[0]], df[v[1]]
        # detrend
        # subtract mean by default, or linear if desired
        dfluc1 = xrft.detrend(din1, dim="x", detrend_type=detrend)
        dfluc2 = xrft.detrend(din2, dim="x", detrend_type=detrend)
        # normalize by standard deviation
        dnorm1 = dfluc1 / dfluc1.std(dim="x")
        dnorm2 = dfluc2 / dfluc2.std(dim="x")
        # calculate cross spectrum using xrft
        PSD = xrft.cross_spectrum(dnorm1, dnorm2, dim="x", scaling="density",
                                  true_phase=True, true_amplitude=True)
        # average in y and time, take only real values
        PSD_ytavg = PSD.mean(dim=("time","y"))
        # store in Esave
        Esave[vs] = PSD_ytavg.real
    
    # only save positive frequencies
    Esave = Esave.where(Esave.freq_x > 0., drop=True)
    
    # save file and return
    fsave = f"{dnc}spectrogram.nc"
    # delete old file for saving new one
    if os.path.exists(fsave):
        os.system(f"rm {fsave}")
    print(f"Saving file: {fsave}")
    with ProgressBar():
        Esave.to_netcdf(fsave, mode="w")
    
    print("Finished computing spectrograms!")
    return
# --------------------------------
def spectrogram_rot(dnc, t0, t1, dt, nx, nz, detrend="constant"):
    """Purpose: calculate spectrograms using rotated and irregular
    grids in x,y,z. Average in y_hat, interpolate to consistent k_x 
    grid then average in time.
    Calculate for u', w', theta', q', u'w', theta'w', q'w', theta'q'.
    Save netcdf file in dnc.
    :param str dnc: absolute path to netcdf directory
    :param int t0: starting timestep
    :param int t1: ending timestep
    :param int dt: number of files between timesteps
    :param int nx: number of grid points in x-dimension
    :param int nz: number of grid points in z-dimension
    """
    # determine list of file timesteps from input
    timesteps = np.arange(t0, t1+1, dt, dtype=np.int32)
    # determine files to read from timesteps
    fall = [f"{dnc}all_{tt:07d}_rot.nc" for tt in timesteps]
    # number of timesteps for making new arrays
    nt = len(timesteps)
    # use nx to determine index for positive frequencies
    jfreq = nx//2 + 1
    # also can use to determine number of frequencies to store
    nfreq = nx - jfreq
    # initialize dictionary with empty arrays to store for time averaging
    specall = {}
    # variables to loop over for calculations: spectra
    vall = ["u_rot", "w", "theta", "q"]
    vsave = ["uu", "ww", "tt", "qq"]
    # variables to loop over for cross spectra
    vall2 = [("u_rot", "w"), ("theta", "w"), ("q", "w"), ("theta", "q")]
    vsave2 = ["uw", "tw", "qw", "tq"]
    # loop over vsave + vsave2 and initialize empty arrays
    for var in vsave+vsave2:
        specall[var] = np.zeros((nfreq, nz, nt), dtype=np.float64)
    # BEGIN TIME LOOP
    for jt, ff in enumerate(fall):
        # load file
        print(f"Loading file: {ff}")
        d = xr.load_dataset(ff)
        # look at how dx change with height
        dx = np.zeros(nz)
        # use this info to calculate wavenumbers of *rotated grid*
        khat = np.zeros((nx, nz), dtype=np.float64)
        for jz in range(nz):
            dx[jz] = np.diff(d.rx.isel(z=jz))[0]
            khat[:,jz] = fftshift(fftfreq(nx, dx[jz]))
        # reference wavenumbers from base grid
        kref = fftshift(fftfreq(d.nx, d.dx))
        # LOOP OVER VARIABLES FOR SPECTRA
        for v, vs in zip(vall, vsave):
            # grab data
            din = d[v].drop(["xhat","yhat"])
            # subtract mean by default, or linear if desired
            dfluc = xrft.detrend(din, dim="jx", detrend_type=detrend)
            # normalize by standard deviation
            dnorm = dfluc / dfluc.std(dim="jx")
            # calculate PSD using xrft
            PSD = xrft.power_spectrum(dnorm, dim="jx", true_phase=True, 
                                      true_amplitude=True)
            # average in y, convert to numpy, only take real
            PSD_yavg = PSD.mean(dim="jy").to_numpy().real
            # interpolate to wavenumber of original grid for consistency
            PSD_int = np.zeros((nfreq, nz), dtype=np.float64)
            # loop over heights and interp
            for jz in range(nz):
                PSD_int[:,jz] = np.interp(kref[jfreq:], khat[jfreq:,jz],
                                          PSD_yavg[jfreq:,jz], 
                                          left=np.nan, right=np.nan)
            # store PSD_int in corresponding array in specall
            specall[vs][:,:,jt] = PSD_int
        # LOOP OVER VARIABLES FOR COSPECTRA
        for v, vs in zip(vall2, vsave2):
                # grab data
                din1 = d[v[0]].drop(["xhat","yhat"])
                din2 = d[v[1]].drop(["xhat","yhat"])
                # detrend
                # subtract mean by default, or linear if desired
                dfluc1 = xrft.detrend(din1, dim="jx", detrend_type=detrend)
                dfluc2 = xrft.detrend(din2, dim="jx", detrend_type=detrend)
                # normalize by standard deviation
                dnorm1 = dfluc1 / dfluc1.std(dim="jx")
                dnorm2 = dfluc2 / dfluc2.std(dim="jx")
                # calculate cross spectrum using xrft
                PSD = xrft.cross_spectrum(dnorm1, dnorm2, dim="jx", 
                                          scaling="density",
                                          true_phase=True, true_amplitude=True)
                 # average in y, convert to numpy, only take real
                PSD_yavg = PSD.mean(dim="jy").to_numpy().real
                # interpolate to wavenumber of original grid for consistency
                PSD_int = np.zeros((nfreq, nz), dtype=np.float64)
                # loop over heights and interp
                for jz in range(nz):
                    PSD_int[:,jz] = np.interp(kref[jfreq:], khat[jfreq:,jz],
                                              PSD_yavg[jfreq:,jz], 
                                              left=np.nan, right=np.nan)
                # store PSD_int in corresponding array in specall
                specall[vs][:,:,jt] = PSD_int
    # OUTSIDE TIME LOOP
    # need to average everything in specall across time dimension (axis=2)
    # also convert to DataArray and store in Esave to output as netcdf file
    Esave = xr.Dataset(data_vars=None, attrs=d.attrs)
    # add additional attr for detrend_first
    Esave.attrs["detrend_type"] = str(detrend)
    # loop over all vsave+vsave2 and average
    print("Average in time...")
    for var in vsave+vsave2:
        Esave[var] = xr.DataArray(data=np.nanmean(specall[var], axis=2),
                                  coords=dict(freq_x=kref[jfreq:], z=d.z))
    # save file and return
    fsave = f"{dnc}spectrogram_rot.nc"
    # delete old file for saving new one
    if os.path.exists(fsave):
        os.system(f"rm {fsave}")
    print(f"Saving file: {fsave}")
    with ProgressBar():
        Esave.to_netcdf(fsave, mode="w")
    
    print("Finished computing rotated spectrograms!")
    return
# --------------------------------
def amp_mod(dnc, ts, s):
    """Input timeseries xarray Dataset and stats file 
    with loaded LES data to calculate amplitude modulation 
    coefficients for all of: u, v, w, theta, q, uw, tw, qw, tq.
    Automatically use cutoff wavelength of lambda_c = z_i,
    convert to temporal frequency with Taylor's hypothesis.
    Save netcdf file in dnc.

    :param str dnc: absolute path to netcdf directory for saving new file
    :param Dataset ts: timeseries xarray Dataset for calculating
    :param Dataset s: mean statistics file for reference
    """
    print("Begin calculating amplitude modulation coefficients")
    # lengthscale of filter to separate large and small scales
    delta = s.h.values
    # cutoff frequency from Taylor
    f_c = 1./(delta/ts.u_mean_rot)
    # names of variables to loop over: first half are detrended
    var_ts = ["udr", "vdr", "wd", "td", "qd", "uw", "tw", "qw", "tq"]
    # initialize empty dictionary to hold FFT timseries
    f_all = {}
    print("Lowpass filter signals to separate large and small scale components")
    # loop over var_ts and take forward FFT
    for v in var_ts:
        f_all[v] = xrft.fft(ts[v], dim="t", true_phase=True, true_amplitude=True)
        # now loop over heights and lowpass filter
        for jz in range(ts.nz):
            # can do this all in one line
            # set high frequencies equal to zero -- sharp spectral filter
            # only keep -f_c < freq_t < f_c
            f_all[v][:,jz] = f_all[v][:,jz].where((f_all[v].freq_t < f_c.isel(z=jz)) &\
                                                  (f_all[v].freq_t >-f_c.isel(z=jz)),
                                                  other=0.)
            
    # inverse FFT to get large-scale component of signal
    # initialize empty dictionaries for large and small scale components
    ts_l = {}
    ts_s = {}
    for v in var_ts:
        ts_l[v] = xrft.ifft(f_all[v], dim="freq_t", true_phase=True, true_amplitude=True,
                            lag=f_all[v].freq_t.direct_lag).real
        # reset time coordinate
        ts_l[v]["t"] = ts.t
        # calculate small-scale component by subtracting large-scale from full
        ts_s[v] = ts[v] - ts_l[v]

    print("Calculate envelope of small-scale signal from Hilbert transform")
    # calculate envelope of small-scale signal from Hilbert transform
    # dictionary of envelopes
    Env = {}
    for v in var_ts:
        Env[v] = xr.DataArray(data=np.abs(hilbert(ts_s[v], axis=0)),
                              coords=dict(t=ts_s[v].t, z=ts_s[v].z))
        
    # lowpass filter the small-scale envelopes
    # dictionary of FFT envelopes
    f_E = {}
    # FFT the envelopes
    for v in var_ts:
        f_E[v] = xrft.fft(Env[v], dim="t", true_phase=True, 
                          true_amplitude=True, detrend="linear")
        # loop over heights and lowpass filter (same as above)
        for jz in range(ts.nz):
            f_E[v][:,jz] = f_E[v][:,jz].where((f_E[v].freq_t < f_c.isel(z=jz)) &\
                                              (f_E[v].freq_t >-f_c.isel(z=jz)),
                                              other=0.)
    
    # inverse FFT the filtered envelopes
    E_filt = {}
    for v in var_ts:
        E_filt[v] = xrft.ifft(f_E[v], dim="freq_t", true_phase=True,
                              true_amplitude=True, 
                              lag=f_E[v].freq_t.direct_lag).real
        # reset time coordinate
        E_filt[v]["t"] = ts.t

    # AM Coefficients -----------
    print("Computing AM coefficients")
    # new dataset to hold correlation coefficients
    R = xr.Dataset(data_vars=None,
                   coords=dict(z=ts.z),
                   attrs=ts.attrs)
    # add delta as attr
    R.attrs["cutoff"] = delta
    # list of variable names for new save notation (can zip with var_ts to match)
    vsave = ["u", "v", "w", "t", "q", "uw", "tw", "qw", "tq"]
    # correlation between large scale u and filtered envelope of small-scale variable
    for vts, vv in zip(var_ts, vsave):
        key = f"ul_E{vv}"
        R[key] = xr.corr(ts_l["udr"], E_filt[vts], dim="t")
    # correlation between large scale w and filtered envelope of small-scale variable
    for vts, vv in zip(var_ts, vsave):
        key = f"wl_E{vv}"
        R[key] = xr.corr(ts_l["wd"], E_filt[vts], dim="t")
    # correlation between large scale theta and filtered envelope of small-scale variable
    for vts, vv in zip(var_ts, vsave):
        key = f"tl_E{vv}"
        R[key] = xr.corr(ts_l["td"], E_filt[vts], dim="t")
    # correlation between large scale q and filtered envelope of small-scale variable
    for vts, vv in zip(var_ts, vsave):
        key = f"ql_E{vv}"
        R[key] = xr.corr(ts_l["qd"], E_filt[vts], dim="t")

    # save file
    fsavenc = f"{dnc}AM_coefficients.nc"
    # delete old file for saving new one
    if os.path.exists(fsavenc):
        os.system(f"rm {fsavenc}")
    print(f"Saving file: {fsavenc}")
    with ProgressBar():
        R.to_netcdf(fsavenc, mode="w")

    return