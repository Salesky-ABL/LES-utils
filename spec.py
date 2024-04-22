#!/home/bgreene/anaconda3/envs/LES/bin/python
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
def acf1d(din, detrend="constant", poslags=False):
    """Input 3D xarray DataArray and calculate acf in x-dim, average in y
    Return DataArray(xlag, z)

    :param DataArray din: data to compute acf
    :param str detrend: how to detrend along x-axis before processing,\
        default="constant" (also accepts "linear")
    :param bool poslags: return only positive lags, default=False
    """
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
    Ry = R.mean(dim=("y")).compute()
    # check poslags
    if poslags:
        return Ry.where(Ry.x >= 0., drop=True)
    else:
        return Ry

def autocorr_1d(dnc, fall, s, detrend="constant"):
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
    Rsave = xr.Dataset(data_vars=None, attrs=s.attrs)
    # add additional attr for detrend_first
    Rsave.attrs["detrend_type"] = str(detrend)
    # get number of files
    nf = len(fall)
    # variables to loop over for calculations
    vall = ["u", "v", "w", "theta", "u_rot", "v_rot", "q"]

    # begin looping over each file and load one by one
    for jf, ff in enumerate(fall):
        # load file
        print(f"Loading file: {ff}")
        d = xr.load_dataset(ff)
        # loop over variables
        for v in vall:
            # call acf1d
            R = acf1d(d[v], detrend=detrend)
            # if this is the first file, create new data variable
            if jf == 0:
                Rsave[v] = R
            # otherwise, add to existing array for averaging in time later
            else:
                Rsave[v] += R
    # compute time average
    for v in vall:
        Rsave[v] /= float(nf)

    # save nc file
    fsave = f"{dnc}R_1d.nc"
    print(f"Saving file: {fsave}")
    with ProgressBar():
        Rsave.to_netcdf(fsave, mode="w")
    
    print("Finished computing 1d autocorrelation functions!")
    return
# --------------------------------
def autocorr_2d(dnc, fall, s, timeavg=True):
    """Input list of files to compute the
    2d autocorrelation function in x-y planes, then average in
    time (if desired). Calculate for u, v, w, theta, u_rot, v_rot.
    Save netcdf file in dnc.
    -Input-
    dnc: string, absolute path to netcdf directory for saving new file
    fall: list of strings of all the files to be loaded one-by-one
    s: xr.Dataset, stats file corresponding to simulation for attrs
    timeavg: flag to average acf in time, default=True
    """
    # construct Dataset to save
    Rsave = xr.Dataset(data_vars=None, attrs=s.attrs)
    # variables to loop over for calculations
    vall = ["u", "v", "w", "theta", "u_rot", "v_rot"]
    # get number of files
    nf = len(fall)
    # construct dictionary to hold instantaneous values if timeavg=False
    Rsave2 = xr.Dataset(data_vars=None, attrs=s.attrs)

    # begin looping over each file and load one by one
    for jf, ff in enumerate(fall):
        # load file
        print(f"Loading file: {ff}")
        d = xr.load_dataset(ff)
        # loop over variables
        for v in vall:
            # grab data
            din = d[v]
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
            # store in Rsave
            # if this is the first file, create new data variable
            if jf == 0:
                Rsave[v] = R
            # otherwise, add to existing array for averaging in time later
            else:
                Rsave[v] += R
            # store values if not using timeavg
            if ((not timeavg) & (jf == 0)):
                temp = np.zeros((R.x.size, R.y.size, R.z.size, nf),
                                dtype=np.float64)
                # store R in temp
                temp[:,:,:,jf] = R
                # convert temp to DataArray, store in Rsave2
                Rsave2[v][:,:,:,jf] = xr.DataArray(data=temp, 
                                                   dims=("x","y","z","time"),
                                                   coords=dict(x=R.x, y=R.y, z=R.z, 
                                                               time=np.arange(nf)))
            # store in Rinst if not using timeaverage
            if ((not timeavg) & (jf >= 0)):
                Rsave2[v][:,:,:,jf] = R

    # average in time if desired, then store in Rsave
    if timeavg:
        for v in vall:
            Rsave[v] /= float(nf)
    # save nc file
    if timeavg:
        fsave = f"{dnc}R_2d_8-10h.nc"
        print(f"Saving file: {fsave}")
        with ProgressBar():
            Rsave.to_netcdf(fsave, mode="w")
    else:
        fsave = f"{dnc}R_2d_inst.nc"
        print(f"Saving file: {fsave}")
        with ProgressBar():
            Rsave2.to_netcdf(fsave, mode="w")
    
    print("Finished computing 2d autocorrelation functions!")
    return
# --------------------------------
def spectrogram(dnc, fall, s, detrend="constant"):
    """Calculate power spectral density along x-direction, then 
    average in y and time for a given list of files.
    Calculate for u', w', theta', q', u'w', theta'w', q'w', theta'q'.
    Save a single netcdf file in dnc.
    -Input-
    dnc: string, absolute path to netcdf directory for saving new file
    fall: list of strings of all the files to be loaded one-by-one
    s: xr.Dataset, stats file corresponding to simulation for attrs
    detrend: string, how to detrend along x-axis before processing,\
        default="constant" (also accepts "linear")
    """
    # define the variables to loop over for spectra
    vall = ["u_rot", "w", "theta", "q", "thetav"]
    vsave = ["uu", "ww", "tt", "qq", "tvtv"]
    # variables to loop over for cross spectra
    vall2 = [("u_rot", "w"), ("theta", "w"), ("q", "w"), ("theta", "q")]
    vsave2 = ["uw", "tw", "qw", "tq"]
    # construct Dataset to save
    Esave = xr.Dataset(data_vars=None, attrs=s.attrs)
    # add additional attr for detrend_first
    Esave.attrs["detrend_type"] = str(detrend)
    # get number of files
    nf = len(fall)

    # begin looping over each file and load one by one
    for jf, ff in enumerate(fall):
        # load file
        print(f"Loading file: {ff}")
        d = xr.load_dataset(ff)
        # calculate rotated u, v based on xy mean
        if not bool(d.rotate):
            uxy = d.u.mean(dim=("x","y"))
            vxy = d.v.mean(dim=("x","y"))
            angle = np.arctan2(vxy, uxy)
            d["u_rot"] = d.u*np.cos(angle) + d.v*np.sin(angle)
            d["v_rot"] =-d.u*np.sin(angle) + d.v*np.cos(angle)
        # calculate thetav
        d["thetav"] = d.theta * (1. + 0.61*d.q/1000.)
        # loop over variables for spectra
        for v, vs in zip(vall, vsave):
            print(f"Computing spectra: {vs}")
            # grab data
            din = d[v]
            # calculate PSD using xrft
            PSD = xrft.power_spectrum(din, dim="x", true_phase=True, 
                                      true_amplitude=True, detrend_type=detrend)
            # average in y and time, take only real values
            PSD_yavg = PSD.mean(dim=("y"))
            # store in Esave
            # if this is the first file, create new data variable
            if jf == 0:
                Esave[vs] = PSD_yavg.real
            # otherwise, add to existing array for averaging in time later
            else:
                Esave[vs] += PSD_yavg.real

        # loop over variables for cross spectra
        for v, vs in zip(vall2, vsave2):
            print(f"Computing cross spectra: {vs}")
            # grab data
            din1, din2 = d[v[0]], d[v[1]]
            # calculate cross spectrum using xrft
            PSD = xrft.cross_spectrum(din1, din2, dim="x", scaling="density",
                                      true_phase=True, true_amplitude=True,
                                      detrend_type=detrend)
            # average in y and time, take only real values
            PSD_yavg = PSD.mean(dim=("y"))
            # store in Esave
            # if this is the first file, create new data variable
            if jf == 0:
                Esave[vs] = PSD_yavg.real
            # otherwise, add to existing array for averaging in time later
            else:
                Esave[vs] += PSD_yavg.real
    
    # only save positive frequencies
    Esave = Esave.where(Esave.freq_x > 0., drop=True)
    # average in time by dividing by number of files
    for vs in vsave+vsave2:
        Esave[vs] /= float(nf)
    
    # save file and return
    # check if rotate attr exists
    if "rotate" in d.attrs.keys():
        if bool(d.rotate):
            fsave = f"{dnc}spectrogram_rot.nc"
        else:
            fsave = f"{dnc}spectrogram.nc"
    else:
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
def amp_mod(dnc, ts, delta):
    """Input timeseries xarray Dataset and stats file 
    with loaded LES data to calculate amplitude modulation 
    coefficients for all of: u, v, w, theta, q, uw, tw, qw, tq.
    convert to temporal frequency with Taylor's hypothesis.
    Save netcdf file in dnc.

    :param str dnc: absolute path to netcdf directory for saving new file
    :param Dataset ts: timeseries xarray Dataset for calculating
    :param float delta: cutoff wavelength to use for scale separation
    """
    print("Begin calculating amplitude modulation coefficients")
    # cutoff frequency from Taylor
    f_c = ts.u_mean_rot / delta
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
        R.attrs["use_q"] = "True"
    # save file
    fsavenc = f"{dnc}AM_coefficients.nc"
    # delete old file for saving new one
    if os.path.exists(fsavenc):
        os.system(f"rm {fsavenc}")
    print(f"Saving file: {fsavenc}")
    with ProgressBar():
        R.to_netcdf(fsavenc, mode="w")

    return
# --------------------------------
def LCS(dat1, dat2, zi=None, zzi=None):
    """Calculate linear coherence spectra between two 3D DataArrays
    at a selected level, or at all heights. Return DataArray.
    :param DataArray dat1: first variable to compare
    :param DataArray dat2: second variable to compare (optional)
    :param float zi: height of ABL depth in m; default=None
    :param float zzi: if None, compute at all levels; if float, use reference\
        height zzi in terms of z/zi. Default=None
    """
    # compute forward FFT of dat1 (will use regardless of compute mode)
    F1 = xrft.fft(dat1, dim="x", true_phase=True, true_amplitude=True,
                  detrend="linear")
    # check zzi: if not None, compute LCS @ z/zi=zzi; else, compute for all z
    if ((zzi is not None) & (zi is not None)):
        izr = abs((dat1.z/zi).values - zzi).argmin()
        # calculate Gamma^2
        G2 = np.absolute((F1 * F1.isel(z=izr).conj()).mean(dim=("y"))) ** 2./\
                ((np.absolute(F1)**2.).mean(dim=("y")) *\
                 (np.absolute(F1.isel(z=izr))**2.).mean(dim=("y")))
        # Finished - return
        return G2
    else:
        # calc between 2 variables at all heights
        F2 = xrft.fft(dat2, dim="x", true_phase=True, true_amplitude=True,
                      detrend="linear")
        G2 = np.absolute((F1 * F2.conj()).mean(dim=("y"))) ** 2. /\
                ((np.absolute(F1)**2.).mean(dim=("y")) *\
                 (np.absolute(F2)**2.).mean(dim=("y")))
        # Finished - return
        return G2
# --------------------------------
def nc_LCS(dnc, fall, s, zzi_list, const_zr_varlist, const_zr_savelist, 
           all_zr_pairs, all_zr_savelist):
    """Purpose: use LCS function to calculate LCS at desired values of zr
    and combinations of variables. Reads one file at a time.
    -Inputs-
    dnc: string, absolute path to directory for saving files
    fall: list of strings of all desired filenames to load/process
    s: xr.Dataset, mean statistics file corresponding to fall
    zzi_list: list of values of z/zi to use as zr
    const_zr_varlist: list of variables to calculate at\
        corresponding list of zr in zzi_list
    const_zr_savelist: list of variable names to use\
        as keys for saving in Dataset with const_zr_varlist
    all_zr_pairs: list of variable pairs for desired\
        LCS calculation at all reference heights
    all_zr_savelist: list of variable names to use\
        as keys for saving in Dataset with all_zr_pairs
    """
    # initialize empty Dataarray with same attrs as s
    Gsave = xr.Dataset(data_vars=None, attrs=s.attrs)
    # get number of files
    nf = len(fall)
    # list of all variables for averaging later
    vavg = []
    # begin looping over each file, one by one
    for jf, ff in enumerate(fall):
        # load file
        print(f"Loading file: {ff}")
        d = xr.load_dataset(ff)
        # calculate rotated u, v based on xy mean
        if not bool(d.rotate):
            uxy = d.u.mean(dim=("x","y"))
            vxy = d.v.mean(dim=("x","y"))
            angle = np.arctan2(vxy, uxy)
            d["u_rot"] = d.u*np.cos(angle) + d.v*np.sin(angle)
            d["v_rot"] =-d.u*np.sin(angle) + d.v*np.cos(angle)
        # calculate thetav
        d["thetav"] = d.theta * (1. + 0.61*d.q/1000.)
        # first calculate LCS for each value of z/zi and each variable
        # begin loop over zzi
        for j, jzi in enumerate(zzi_list):
            # compute closest value of z/zi given jzi (jzi=value from 0-1)
            # jz will be corresponding z index closest to jzi
            jz = abs((d.z/s.h).values - jzi).argmin()
            # add z[jz] as attr in Gsave (for first file only)
            if jf == 0:
                Gsave.attrs[f"zr{j}"] = d.z.isel(z=jz).values
            # loop over variables and calculate LCS
            for var, vsave in zip(const_zr_varlist, const_zr_savelist):
                print(f"Computing LCS at one height: {vsave}")
                # compute
                # if this is first file, create new variable
                if jf == 0:
                    Gsave[f"{vsave}{j}"] = LCS(d[var], None, s.h.values, jzi)
                    vavg.append(f"{vsave}{j}")
                else:
                    Gsave[f"{vsave}{j}"] += LCS(d[var], None, s.h.values, jzi)
        # compute LCS for 2 variables at all heights
        for pair, vsave in zip(all_zr_pairs, all_zr_savelist):
            print(f"Computing LCS for 2 variables at all heights: {vsave}")
            # if this is first file, create new variable
            if jf == 0:
                Gsave[vsave] = LCS(d[pair[0]], d[pair[1]])
                vavg.append(vsave)
            else:
                Gsave[vsave] += LCS(d[pair[0]], d[pair[1]])
    # only keep positive frequencies
    Gsave = Gsave.where(Gsave.freq_x > 0., drop=True)
    # average in time by dividing by number of files
    for vs in vavg:
        Gsave[vs] /= float(nf)
    # save file
    # check if rotate attr exists
    if "rotate" in d.attrs.keys():
        if bool(d.rotate):
            fsave = f"{dnc}G2_8-10h_rot.nc"
        else:
            fsave = f"{dnc}G2.nc"
    else:
        fsave = f"{dnc}G2.nc"
    # delete old file if exists
    if os.path.exists(fsave):
        os.system(f"rm {fsave}")
    print(f"Saving file: {fsave}")
    with ProgressBar():
        Gsave.to_netcdf(fsave, mode="w")
    print("Finished nc_LCS!")
    return
# --------------------------------
def cond_avg(dnc, t0, t1, dt, use_rot, s, cond_var, cond_thresh, cond_jz,
             cond_scale, varlist, varscale_list, svarlist):
    """Purpose: calculate conditionally averaged fields
    For example, if cond_var is u, cond_thresh is -2:
    u'(x,y,z=jz) < -2*sigma_u
    Only load one file at a time
    Return xarray Dataset
    :param str dnc: directory for loading files
    :param int t0: starting timestep
    :param int t1: final timestep to load
    :param int dt: number of timesteps in between files
    :param bool use_rot: use rotated fields
    :param xr.Dataset s: statistics Dataset for useful parameters
    :param str cond_var: variable name used as condition
    :param float cond_thresh: number of standard deviations as threshold for condition.\
        if negative, cond is < cond_thresh; if positive, cond is > cond_thresh
    :param int cond_jz: z index desired to enforce condition
    :param float cond_scale: value to normalize condition by, e.g. ustar, tstar
    :param list<str> varlist: list of variables to conditionally average
    :param list<str> varscale_list: list of scales to normalize varaibles in varlist
    :param list<str> svarlist: list of variable save names corresponding to varlist
    """
    #  timesteps for loading files
    timesteps = np.arange(t0, t1+1, dt, dtype=np.int32)
    # determine files to read from timesteps
    if use_rot:
        fall = [f"{dnc}all_{tt:07d}_rot.nc" for tt in timesteps]
    else:
        fall = [f"{dnc}all_{tt:07d}.nc" for tt in timesteps]
    nf = len(fall)
    # grab some useful params from stats file
    dx = s.dx
    nx = s.nx
    ny = s.ny
    nz = s.nz
    #
    # determine cutoff value: read middle volume file
    #
    dd = xr.load_dataset(fall[nf//2])
    # if cond_var is u_rot, check for use_rot to see if variable exists
    if cond_var == "u_rot":
        # if use_rot, just use u_rot for next step; otherwise, 
        # need to calculate u_rot
        if not use_rot:
            # calculate u_rot
            u_mean = dd.u.mean(dim=("x","y"))
            v_mean = dd.v.mean(dim=("x","y"))
            angle = np.arctan2(v_mean, u_mean)
            dd["u_rot"] = dd.u*np.cos(angle) + dd.v*np.sin(angle)
            dd["v_rot"] =-dd.u*np.sin(angle) + dd.v*np.cos(angle)    
        # now use u_rot to calculate mean and std of u'/u*
        dd["u_p"] = (dd.u_rot - dd.u_rot.mean(dim=("x","y"))) / s.ustar0
        # calculate mean and std of u'/u*
        mu = dd.u_p.mean(dim=("x","y"))
        std = dd.u_p.std(dim=("x","y"))
        # real quick, also create keyname for u_rot
        cond_svar = "u"
    # calculate condition mu and std for other possible variables
    else:
        cond_p = (dd[cond_var] - dd[cond_var].mean(dim=("x","y"))) / cond_scale
        mu = cond_p.mean(dim=("x","y"))
        std = cond_p.std(dim=("x","y"))
        cond_svar = cond_var
    # calculate alpha cutoffs based on mu, std, cond_thresh, and cond_jz
    alpha = mu[cond_jz] + cond_thresh*std[cond_jz]
    #
    # Prepare arrays for conditional averaging
    #
    # max points to include
    n_delta = int(s.h/dx)
    # number of points upstream and downstream to include
    n_min = 2*n_delta
    n_max = 2*n_delta
    # initialize conditionally averaged arrays: one for each variable in varlist
    # do this in a dictionary
    condall = {}
    for svar in svarlist:
        # take time to create keys here; will use same for saving out in Dataset later
        # see if condition is hi or lo
        if cond_thresh > 0:
            hilo = "hi"
        else:
            hilo = "lo"
        key = f"{svar}_cond_{cond_svar}_{hilo}"
        condall[key] = np.zeros((n_min+n_max, nz), dtype=np.float64)
    # initialize counter for number of points satisfying condition
    ncond = 0
    #
    # BEGIN LOOP OVER FILES
    #
    for ff in fall:
        # load file
        print(f"Loading file: {ff}")
        dd = xr.load_dataset(ff)
        # rotate velocities if not use_rot and u_rot in varlist
        if ((not use_rot) & ("u_rot" in varlist)):
            u_mean = dd.u.mean(dim=("x","y"))
            v_mean = dd.v.mean(dim=("x","y"))
            angle = np.arctan2(v_mean, u_mean)
            dd["u_rot"] = dd.u*np.cos(angle) + dd.v*np.sin(angle)
            dd["v_rot"] =-dd.u*np.sin(angle) + dd.v*np.cos(angle)
        # compute normalized values, store in dd
        # create big arrays for each variable in var so we dont have to deal
        # with periodicity
        varbig = {}
        for var, svar, scale in zip(varlist, svarlist, varscale_list):
            # normalized values
            key = f"{svar}_p"
            dd[key] = (dd[var] - dd[var].mean(dim=("x","y"))) / scale
            # big arrays
            varbig[key] = np.zeros((4*nx,ny,nz), dtype=np.float64)
            for jx in range(4):
                varbig[key][jx*nx:(jx+1)*nx,:,:] = dd[key][:,:,:].to_numpy()
        #
        # Calculate conditional averages
        #
        # define key for normalized condition variable
        keycond = f"{cond_svar}_p"
        # loop over y
        for jy in range(ny):
            # loop over x
            for jx in range(nx):
                # include points if meeting condition
                is_hi = ((hilo == "hi") & (dd[keycond][jx,jy,cond_jz] > alpha))
                is_lo = ((hilo == "lo") & (dd[keycond][jx,jy,cond_jz] < alpha))
                if (is_hi | is_lo):
                    # increment counter
                    ncond += 1
                    # loop over svarlist
                    for svar in svarlist:
                        kbig = f"{svar}_p"
                        kcond = f"{svar}_cond_{cond_svar}_{hilo}"
                        values = varbig[kbig][(jx+nx-n_min):(jx+nx+n_max),jy,:]
                        condall[kcond][:,:] += values
    # FINISHED TIME LOOP
    print("Finished processing all timesteps")
    # normalize each by number of samples
    for key in condall.keys():
        condall[key][:,:] /= ncond
    # store these variables in Dataset for saving
    # new array coordinates
    xnew = np.linspace(-1*n_min*dx, n_max*dx, (n_min+n_max))
    # empty dataset
    dsave = xr.Dataset(data_vars=None, coords=dict(x=xnew, z=s.z), attrs=s.attrs)
    # store each variable as DataArray
    for key in condall.keys():
        dsave[key] = xr.DataArray(data=condall[key], dims=("x","z"),
                                    coords=dict(x=xnew, z=s.z))
    # include attrs for each variable
    # jz values
    dsave.attrs["jz"] = cond_jz
    # number of times meeting condition
    dsave.attrs[f"n_{cond_svar}_{hilo}"] = ncond
    # threshold value
    dsave.attrs[f"alpha_{cond_svar}_{hilo}"] = alpha.values
    # save and return
    fsave = f"{dnc}cond_avg_{cond_svar}_{hilo}_8-10h.nc"
    # delete old file for saving new one
    if os.path.exists(fsave):
        os.system(f"rm {fsave}")
        print(f"Saving file: {fsave}")
    with ProgressBar():
        dsave.to_netcdf(fsave, mode="w")

    print(f"Finished processing conditional averaging on {cond_svar}_{hilo}")
    return
# --------------------------------
def calc_quadrant(dnc, fall, s, var_pairs=[("u_rot","w"), ("theta","w")], 
                  svarlist=["uw", "tw"], hole_factor=None):
    """Purpose: Calculate quadrant components of given pairs of variables and
    also calculate their mixing efficiences, eta. Calc one file at a time.
    -Inputs-
    str dnc: directory for saving files
    fall: list of all files to load for analysis
    s: xr.Dataset, mean statistics file corresponding to fall
    var_pairs: list of variable pairs to calculate covars
    svarlist: list of variable save names corresponding to varlist
    hole_factor: float, proportion of the xy-mean relative covar to use as\
        a filter theshold for including in quadrant means, default=None
    -Output-
    saves netcdf file in dnc
    """
    # get instantaneous perturbations for each unique variable in var_pairs
    # create list of unique variables from pairs
    punique = []
    for p in var_pairs:
        for pp in p:
            if pp not in punique:
                punique.append(pp)
    # initialize empty dataset for storing
    quad = xr.Dataset(data_vars=None,
                      coords=dict(z=s.z),
                      attrs=s.attrs)
    # get number of files
    nf = len(fall)
    # list of all variables for averaging later
    vavg = []
    # begin looping over each file, one by one
    for jf, ff in enumerate(fall):
         # load file
        print(f"Loading file: {ff}")
        d = xr.load_dataset(ff)
        # calculate rotated u, v based on xy mean
        if not bool(d.rotate):
            uxy = d.u.mean(dim=("x","y"))
            vxy = d.v.mean(dim=("x","y"))
            angle = np.arctan2(vxy, uxy)
            d["u_rot"] = d.u*np.cos(angle) + d.v*np.sin(angle)
            d["v_rot"] =-d.u*np.sin(angle) + d.v*np.cos(angle)
        # calculate thetav
        d["thetav"] = d.theta * (1. + 0.61*d.q/1000.)
        # store necessary fields in dictionary (reset each new file)
        varp = {}
        for var in punique:
            varp[var] = (d[var] - d[var].mean(dim=("x","y"))).compute()
        # calculate four quadrants
        # loop over variable pairs
        for pair, vlab in zip(var_pairs, svarlist):
            print(f"Calculating quadrant components for: {vlab}")
            dat1 = varp[pair[0]]
            dat2 = varp[pair[1]]
            # dat1 > 0, dat2 > 0
            d_pp = dat1.where(dat1 > 0.) * dat2.where(dat2 > 0.)
            # dat1 > 0, dat2 < 0
            d_pn = dat1.where(dat1 > 0.) * dat2.where(dat2 < 0.)
            # dat1 < 0, dat2 > 0
            d_np = dat1.where(dat1 < 0.) * dat2.where(dat2 > 0.)
            # dat1 < 0, dat2 < 0
            d_nn = dat1.where(dat1 < 0.) * dat2.where(dat2 < 0.)
            # apply hole_factor
            if hole_factor is not None:
                d1d2 = (dat1 * dat2).mean(dim=("x","y")).compute()
                d_pp = d_pp.where(abs(d_pp) >= hole_factor*abs(d1d2))
                d_pn = d_pn.where(abs(d_pn) >= hole_factor*abs(d1d2))
                d_np = d_np.where(abs(d_np) >= hole_factor*abs(d1d2))
                d_nn = d_nn.where(abs(d_nn) >= hole_factor*abs(d1d2))
            # calculate means and store
            # if this is first file, create new variable
            if jf == 0:
                quad[f"{vlab}_pp"] = d_pp.mean(dim=("x","y"), skipna=True)
                quad[f"{vlab}_pn"] = d_pn.mean(dim=("x","y"), skipna=True)
                quad[f"{vlab}_np"] = d_np.mean(dim=("x","y"), skipna=True)
                quad[f"{vlab}_nn"] = d_nn.mean(dim=("x","y"), skipna=True)
                vavg += [f"{vlab}_{jp}" for jp in ["pp", "pn", "np", "nn"]]
            else:
                quad[f"{vlab}_pp"] += d_pp.mean(dim=("x","y"), skipna=True)
                quad[f"{vlab}_pn"] += d_pn.mean(dim=("x","y"), skipna=True)
                quad[f"{vlab}_np"] += d_np.mean(dim=("x","y"), skipna=True)
                quad[f"{vlab}_nn"] += d_nn.mean(dim=("x","y"), skipna=True)
        print("Finished calculating quadrant components")
        # begin calculating mixing efficiencies eta
        for pair, vlab in zip(var_pairs, svarlist):
            print(f"Calculating resolved covariance for: {vlab}")
            # calculate full covariance between variable pair
            # use detrended variables stored in varp
            dat1 = varp[pair[0]]
            dat2 = varp[pair[1]]
            # multiply together
            d1d2 = dat1 * dat2
            # compute cov_res by averaging in xy
            cov_res = d1d2.mean(dim=("x","y")).compute()
            # use hole_factor if it exists and override
            if hole_factor is not None:
                d1d2 = d1d2.where(abs(d1d2) >= hole_factor*abs(cov_res))
                cov_res = d1d2.mean(dim=("x","y"), skipna=True)
            # save this as individual variable to calculate full eta after
            # time avg loop ends
            if jf == 0:
                quad[f"{vlab}_cov_res"] = cov_res
                vavg.append(f"{vlab}_cov_res")
            else:
                quad[f"{vlab}_cov_res"] += cov_res
    # outside of time loop
    # calculate time avg by dividing everything by nf
    for vs in vavg:
        quad[vs] /= float(nf)
    # now can calculate mixing efficiencies using full mean values
    for pair, vlab in zip(var_pairs, svarlist):
        print(f"Calculating mixing efficiency for: {vlab}")
        # NOTE: in SBL, all variables increase with height so will use
        # quadrants II and IV in denom; will need to change in future!
        # example: eta_uw = |<u'w'>| / |(<u-w+> + <u+w->)|
        scov = f"{vlab}_cov_res"
        quad[f"eta_{vlab}"] = np.abs(quad[scov]) / np.abs(quad[f"{vlab}_np"] +\
                                                          quad[f"{vlab}_pn"])
    print("Finished calculating mixing efficiencies")
    # finally, calculate sum of quadrants
    for pair, vlab in zip(var_pairs, svarlist):
        print(f"Calculating quadrant sum for: {vlab}")
        quad[f"{vlab}_sum"] = np.abs(quad[f"{vlab}_pp"]) +\
                              np.abs(quad[f"{vlab}_pn"]) +\
                              np.abs(quad[f"{vlab}_np"]) +\
                              np.abs(quad[f"{vlab}_nn"])
    print("Finished calculating quadrant sums")
    # save quad Dataset as netcdf
    vlabs = "_".join(svarlist)
    if hole_factor is not None:
        # add hole_factor as attr
        quad.attrs["hole_factor"] = hole_factor
        # add as part of name
        fsave = f"{dnc}{vlabs}_quadrant_H{hole_factor:.1f}_8-10h.nc"
    else:
        fsave = f"{dnc}{vlabs}_quadrant_8-10h.nc"
    # delete old file for saving new one
    if os.path.exists(fsave):
        os.system(f"rm {fsave}")
    print(f"Saving file: {fsave}")
    with ProgressBar():
        quad.to_netcdf(fsave, mode="w")
    
    return
# --------------------------------
def get_1d_hist(dnc, fall, s, zref, jzref):
    """Return 1 dimensional arrays of all perturbation values of: u_rot, w, theta
    at input levels of z/zref (i.e., z/zi, z/zj). Return one netcdf file to use
    for constructing pdfs and joint pdfs.
    -Input-
    dnc: directory for saving files
    fall: list of all files to load for analysis
    s: xr.Dataset, mean statistics file corresponding to fall
    zref: reference height such as h or zj for normalizing z
    jzref: list of values of z/zref to extract data from
    -Output-
    saves netcdf file in dnc
    """
    # define indices for values of z/zj
    jz = [abs((s.z/zref).values - zzj).argmin() for zzj in jzref]
    # loop through all files
    for jf, ff in enumerate(fall):
        # load file
        print(f"Loading file: {ff}")
        d = xr.load_dataset(ff)
        # calculate rotated u
        uxy = d.u.mean(dim=("x","y"))
        vxy = d.v.mean(dim=("x","y"))
        angle = np.arctan2(vxy, uxy)
        u = d.u*np.cos(angle) + d.v*np.sin(angle)
        # grab other variables of interest: w, theta
        w = d.w
        theta = d.theta
        # compute perturbations
        up = u - u.mean(dim=("x","y"))
        wp = w - w.mean(dim=("x","y"))
        tp = theta - theta.mean(dim=("x","y"))
        # collapse into 1d arrays at selected values of jz
        ulong = up.isel(z=jz).stack(index=("x","y")).reset_index("index", drop=True)
        wlong = wp.isel(z=jz).stack(index=("x","y")).reset_index("index", drop=True)
        tlong = tp.isel(z=jz).stack(index=("x","y")).reset_index("index", drop=True)
        # if this is the first file, create new array to hold all timesteps
        if jf == 0:
            uall = ulong
            wall = wlong
            tall = tlong
        # otherwise, concatenate onto end along the "index" dimension
        else:
            uall = xr.concat([uall, ulong], dim="index")
            wall = xr.concat([wall, wlong], dim="index")
            tall = xr.concat([tall, tlong], dim="index")
    # combine into Dataset
    dsave = xr.Dataset()
    dsave["u"] = uall
    dsave["w"] = wall
    dsave["theta"] = tall
    # save values of z/zj in attrs
    dsave.attrs["zzj"] = jzref
    # save netcdf file
    fsave = f"{dnc}u_w_theta_pdf_8-10h.nc"
    print(f"Saving file: {fsave}")
    dsave.to_netcdf(fsave, "w")
    print("Finished running get_1d_hist()!")

    return

# --------------------------------
def calc_lengthscale(dnc, R):
    """Input xarray Dataset with loaded autocorrelation data to
    calculate integral lengthscales up to first zero crossing for
    each parameter saved at every height.
    Save netcdf file in dnc.

    :param str dnc: absolute path to netcdf directory for saving new file
    :param Dataset R: 1d autocorrelation dataset computed from autocorr_1d
    """
    # grab only positive lags
    R = R.where(R.x >= 0., drop=True)
    # create empty dataset for storing/saving
    Lsave = xr.Dataset()
    # loop over variables
    for var in list(R.keys()):
        # empty array to fill in height
        LL = np.zeros(R.z.size, dtype=np.float64)
        # loop over heights
        for jz in range(R.z.size):
            # find first zero crossing
            izero = np.where(R[var].isel(z=jz) <= 0.)[0]
            # make sure izero isn't empty
            if np.size(izero) > 0:
                izero = izero[0]
            else:
                izero = 1
            # now integrate and store in array
            LL[jz] = R[var].isel(z=jz, x=range(izero)).integrate("x")
        # store LL in Lsave
        Lsave[var] = xr.DataArray(data=LL, coords=dict(z=R.z))
    # save file and return
    fsave = f"{dnc}lengthscale.nc"
    print(f"Saving file: {fsave}")
    with ProgressBar():
        Lsave.to_netcdf(fsave, mode="w")

    return