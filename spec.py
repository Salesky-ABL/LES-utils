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
def spectrogram(dnc, df, detrend="constant", use_q=True):
    """Input 4D xarray Dataset with loaded LES data to calculate
    power spectral density along x-direction, then average in
    y and time. Calculate for u', w', theta', q', u'w', theta'w',
    q'w', theta'q'.
    Save netcdf file in dnc.

    :param str dnc: absolute path to netcdf directory for saving new file
    :param Dataset df: 4d (time,x,y,z) xarray Dataset for calculating
    :param str detrend: how to detrend along x-axis before processing,\
        default="constant" (also accepts "linear")
    :param bool use_q: flag to use specific humidity data, default=True
    """
    # construct Dataset to save
    Esave = xr.Dataset(data_vars=None, attrs=df.attrs)
    # add additional attr for detrend_first
    Esave.attrs["detrend_type"] = str(detrend)

    # variables to loop over for calculations
    vall = ["u_rot", "w", "theta"]
    vsave = ["uu", "ww", "tt"]
    if use_q:
        vall.append("q")
        vsave.append("qq")

    # loop over variables
    for v, vs in zip(vall, vsave):
        # grab data
        din = df[v]
        # calculate PSD using xrft
        PSD = xrft.power_spectrum(din, dim="x", true_phase=True, 
                                  true_amplitude=True, detrend_type=detrend)
        # average in y and time, take only real values
        PSD_ytavg = PSD.mean(dim=("time","y"))
        # store in Esave
        Esave[vs] = PSD_ytavg.real

    # variables to loop over for cross spectra
    vall2 = [("u_rot", "w"), ("theta", "w")]
    vsave2 = ["uw", "tw"]
    if use_q:
        vall2 += [("q", "w"), ("theta", "q")]
        vsave2 += ["qw", "tq"]

    # loop over variables
    for v, vs in zip(vall2, vsave2):
        # grab data
        din1, din2 = df[v[0]], df[v[1]]
        # calculate cross spectrum using xrft
        PSD = xrft.cross_spectrum(din1, din2, dim="x", scaling="density",
                                  true_phase=True, true_amplitude=True,
                                  detrend_type=detrend)
        # average in y and time, take only real values
        PSD_ytavg = PSD.mean(dim=("time","y"))
        # store in Esave
        Esave[vs] = PSD_ytavg.real
    
    # only save positive frequencies
    Esave = Esave.where(Esave.freq_x > 0., drop=True)
    
    # save file and return
    # check if rotate attr exists
    if "rotate" in df.attrs.keys():
        if bool(df.rotate):
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
    var_ts = ["udr", "vdr", "wd", "td", "uw", "tw"]
    # check if moisture variables exist; if so, add those to list
    if "qd" in list(ts.keys()):
        var_ts += ["qd", "qw", "tq"]
        use_q = True
    else:
        use_q = False
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
    vsave = ["u", "v", "w", "t", "uw", "tw"]
    if use_q:
        vsave += ["q", "qw", "tq"]
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
    if use_q:
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
    """Calculate linear coherence spectra between two 4D DataArrays
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
        G2 = np.absolute((F1 * F1.isel(z=izr).conj()).mean(dim=("y","time"))) ** 2./\
                ((np.absolute(F1)**2.).mean(dim=("y","time")) *\
                 (np.absolute(F1.isel(z=izr))**2.).mean(dim=("y","time")))
        # Finished - return
        return G2
    else:
        # calc between 2 variables at all heights
        F2 = xrft.fft(dat2, dim="x", true_phase=True, true_amplitude=True,
                      detrend="linear")
        G2 = np.absolute((F1 * F2.conj()).mean(dim=("y","time"))) ** 2. /\
                ((np.absolute(F1)**2.).mean(dim=("y","time")) *\
                 (np.absolute(F2)**2.).mean(dim=("y","time")))
        # Finished - return
        return G2
# --------------------------------
def nc_LCS(dnc, df, zi, zzi_list, const_zr_varlist, const_zr_savelist, 
           all_zr_pairs, all_zr_savelist):
    """Purpose: use LCS function to calculate LCS at desired values of zr
    and combinations of variables. Save netcdf.
    :param str dnc: absolute path to directory for saving files
    :param xarray.Dataset df: full dataset with dimensions (time,x,y,z)
    :param float zi: ABL depth
    :param list<float> zzi_list: list of values of z/zi to use as zr
    :param list<str> const_zr_varlist: list of variables to calculate at\
        corresponding list of zr in zzi_list
    :param list<str> const_zr_savelist: list of variable names to use\
        as keys for saving in Dataset with const_zr_varlist
    :param list<str> all_zr_pairs: list of variable pairs for desired\
        LCS calculation at all reference heights
    :param list<str> all_zr_savelist: list of variable names to use\
        as keys for saving in Dataset with all_zr_pairs
    """
    # initialize empty Dataarray with same attrs as df
    Gsave = xr.Dataset(data_vars=None, attrs=df.attrs)
    # first calculate LCS for each value of z/zi and each variable
    # begin loop over zzi
    for j, jzi in enumerate(zzi_list):
        # compute closest value of z/zi given jzi (jzi=value from 0-1)
        # jz will be corresponding z index closest to jzi
        jz = abs((df.z/zi).values - jzi).argmin()
        # add z[jz] as attr in Gsave
        Gsave.attrs[f"zr{j}"] = df.z.isel(z=jz).values
        # loop over variables and calculate LCS
        for var, vsave in zip(const_zr_varlist, const_zr_savelist):
            # compute
            Gsave[f"{vsave}{j}"] = LCS(df[var], None, zi, jzi)
    # compute LCS for 2 variables at all heights
    for pair, vsave in zip(all_zr_pairs, all_zr_savelist):
        Gsave[vsave] = LCS(df[pair[0]], df[pair[1]])
    # only keep positive frequencies
    Gsave = Gsave.where(Gsave.freq_x > 0., drop=True)
    # save file
    # check if rotate attr exists
    if "rotate" in df.attrs.keys():
        if bool(df.rotate):
            fsave = f"{dnc}G2_rot.nc"
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
    n_min = 3*n_delta
    n_max = 3*n_delta
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
    fsave = f"{dnc}cond_avg_{cond_svar}_{hilo}.nc"
    # delete old file for saving new one
    if os.path.exists(fsave):
        os.system(f"rm {fsave}")
        print(f"Saving file: {fsave}")
    with ProgressBar():
        dsave.to_netcdf(fsave, mode="w")

    print(f"Finished processing conditional averaging on {cond_svar}_{hilo}")
    return 