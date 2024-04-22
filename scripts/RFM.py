#!/home/bgreene/anaconda3/envs/LES/bin/python
# --------------------------------
# Name: RFM.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 13 July 2023
# Purpose: Calculate random error profiles using the relaxed filtering
# method (Salesky et al. 2012, Dias et al. 2018, Greene and Salesky 2023)
# Updated from RFMnc.py originally written during my PhD for improved
# flexibility and applicability to CBL
# --------------------------------
import os
import sys
sys.path.append("..")
import xrft
import numpy as np
import xarray as xr
from argparse import ArgumentParser
from scipy.fft import fftshift
from scipy.optimize import curve_fit
from dask.diagnostics import ProgressBar
from LESutils import load_stats
from spec import acf1d, calc_lengthscale
# --------------------------------
def calc_inst_var(d1, d2, SGS=None):
    """Calculate 'instantaneous' variance/covariance between 2 parameters
    plus optional SGS term by calculating perturbations (x,y) of d1, d2 then
    multiplying resulting fields
    -Input-
    d1: xr.DataArray, parameter 1 to use
    d2: xr.DataArray, parameter 2 to use
    SGS: xr.DataArray, optional subgrid term to include
    -Output-
    single DataArray in (x,y,z) 
    """
    d1p = (d1 - d1.mean(dim=("x","y"))).compute()
    d2p = (d2 - d2.mean(dim=("x","y"))).compute()
    d1d2_cov = d1p * d2p
    if SGS is not None:
        return d1d2_cov + SGS
    else:
        return d1d2_cov
# --------------------------------
def calc_4_order(dnc, fall, s):
    """Calculate 4th-order variances (u'w'u'w', u'u'u'u', etc.) for use
    in fitting RFM curves as well as computing LP errors later.
    Also calculate 1d autocorrelation function for _all_ parameters here. 
    Save netcdf file in dnc.
    -Inputs-
    dnc: string, absolute path to netcdf directory for saving new file
    fall: list of strings of all desired filenames to load/process
    s: xr.Dataset, mean stats file corresponding to fall
    -Output-
    Saves the following files:
    variances_4_order.nc
    autocorr.nc
    """
    # initialize empty Dataset for 4-order variances
    var4 = xr.Dataset(data_vars=None, attrs=s.attrs)
    # initialize empty Dataset for autocorrs
    R = xr.Dataset(data_vars=None, attrs=s.attrs)
    # get number of files
    nf = len(fall)
    # define list of parameters for looping over for autocorrelations
    vacf = ["u", "u_rot", "v", "v_rot", "w", "theta",
            "thetav", "q",
            "uw", "vw", "tw", "qw", "tvw",
            "uu", "uur", "vv", "vvr", "ww", "tt", "tvtv", "qq",
            "uwuw", "vwvw", "twtw", "tvwtvw", "qwqw",
            "uuuu", "uuuur", "vvvv", "vvvvr", "wwww", 
            "tttt", "tvtvtvtv", "qqqq"]
    # begin looping over all files in fall
    for jf, ff in enumerate(fall):
        # load file
        print(f"Loading file: {ff}")
        d = xr.load_dataset(ff)
        # compute thetav
        d["thetav"] = d.theta * (1. + 0.61*d.q/1000.)
        # begin by computing "instantaneous" variances and covariances, store in df
        print("Calculating 'instantaneous' variances and covariances")
        # use the calc_inst_var function
        # u'w'
        d["uw"] = calc_inst_var(d.u, d.w, d.txz)
        # v'w'
        d["vw"] = calc_inst_var(d.v, d.w, d.tyz)
        # t'w'
        d["tw"] = calc_inst_var(d.theta, d.w, d.tw_sgs)
        # q'w'
        d["qw"] = calc_inst_var(d.q, d.w, d.qw_sgs)
        # tv'w'
        tvw_sgs = d.tw_sgs + 0.61*d.thetav.isel(z=0).mean()*d.qw_sgs/1000.
        d["tvw"] = calc_inst_var(d.thetav, d.w, tvw_sgs)
        # need both rotated and unrotated to get wspd, wdir stats later
        # u'u'
        d["uu"] = calc_inst_var(d.u, d.u)
        # u'u' rot
        d["uur"] = calc_inst_var(d.u_rot, d.u_rot)
        # v'v'
        d["vv"] = calc_inst_var(d.v, d.v)
        # v'v' rot
        d["vvr"] = calc_inst_var(d.v_rot, d.v_rot)
        # w'w'
        d["ww"] = calc_inst_var(d.w, d.w)
        # t't'
        d["tt"] = calc_inst_var(d.theta, d.theta)
        # q'q'
        d["qq"] = calc_inst_var(d.q, d.q)
        # tv'tv'
        d["tvtv"] = calc_inst_var(d.thetav, d.thetav)
        #
        # Using instantaneous vars/covars, calculate inst 4th order variances
        #
        print("Calculating 4th order variances")
        # u'w'u'w'
        d["uwuw"] = calc_inst_var(d.uw, d.uw)
        # v'w'v'w'
        d["vwvw"] = calc_inst_var(d.vw, d.vw)
        # t'w't'w'
        d["twtw"] = calc_inst_var(d.tw, d.tw)
        # u'u'u'u'
        d["uuuu"] = calc_inst_var(d.uu, d.uu)
        # u'u'u'u' rot
        d["uuuur"] = calc_inst_var(d.uur, d.uur)
        # v'v'v'v'
        d["vvvv"] = calc_inst_var(d.vv, d.vv)
        # v'v'v'v' rot
        d["vvvvr"] = calc_inst_var(d.vvr, d.vvr)
        # w'w'w'w'
        d["wwww"] = calc_inst_var(d.ww, d.ww)
        # t't't't'
        d["tttt"] = calc_inst_var(d.tt, d.tt)
        # q'w'q'w'
        d["qwqw"] = calc_inst_var(d.qw, d.qw)
        # q'q'q'q'
        d["qqqq"] = calc_inst_var(d.qq, d.qq)
        # tv'tv'tv'tv'
        d["tvtvtvtv"] = calc_inst_var(d.tvtv, d.tvtv)
        # tv'w'tv'w'
        d["tvwtvw"] = calc_inst_var(d.tvw, d.tvw)
        # compute fourth order variances by computing mean of "inst" vars
        # if this is first file, create new variable in var4
        # otherwise, add to existing (will divide by total files later)
        if jf == 0:
            # u'w'u'w' = var{u'w'} = (u'w' - <u'w'>_xyt)**2
            var4["uwuw_var"] = d.uwuw.mean(dim=("x","y")).compute()
            # v'w'v'w'
            var4["vwvw_var"] = d.vwvw.mean(dim=("x","y")).compute()
            # t'w't'w'
            var4["twtw_var"] = d.twtw.mean(dim=("x","y")).compute()
            # u'u'u'u'
            var4["uuuu_var"] = d.uuuu.mean(dim=("x","y")).compute()
            # u'u'u'u' rot
            var4["uuuur_var"] = d.uuuur.mean(dim=("x","y")).compute()
            # v'v'v'v'
            var4["vvvv_var"] = d.vvvv.mean(dim=("x","y")).compute()
            # v'v'v'v' rot
            var4["vvvvr_var"] = d.vvvvr.mean(dim=("x","y")).compute()
            # w'w'w'w'
            var4["wwww_var"] = d.wwww.mean(dim=("x","y")).compute()
            # t't't't'
            var4["tttt_var"] = d.tttt.mean(dim=("x","y")).compute()
            # q'w'q'w'
            var4["qwqw_var"] = d.qwqw.mean(dim=("x","y")).compute()
            # q'q'q'q'
            var4["qqqq_var"] = d.qqqq.mean(dim=("x","y")).compute()
            # tv'tv'tv'tv'
            var4["tvtvtvtv_var"] = d.tvtvtvtv.mean(dim=("x","y")).compute()
            # tv'w'tv'w'
            var4["tvwtvw_var"] = d.tvwtvw.mean(dim=("x","y")).compute()
        else:
            # u'w'u'w' = var{u'w'} = (u'w' - <u'w'>_xyt)**2
            var4["uwuw_var"] += d.uwuw.mean(dim=("x","y")).compute()
            # v'w'v'w'
            var4["vwvw_var"] += d.vwvw.mean(dim=("x","y")).compute()
            # t'w't'w'
            var4["twtw_var"] += d.twtw.mean(dim=("x","y")).compute()
            # u'u'u'u'
            var4["uuuu_var"] += d.uuuu.mean(dim=("x","y")).compute()
            # u'u'u'u' rot
            var4["uuuur_var"] += d.uuuur.mean(dim=("x","y")).compute()
            # v'v'v'v'
            var4["vvvv_var"] += d.vvvv.mean(dim=("x","y")).compute()
            # v'v'v'v' rot
            var4["vvvvr_var"] += d.vvvvr.mean(dim=("x","y")).compute()
            # w'w'w'w'
            var4["wwww_var"] += d.wwww.mean(dim=("x","y")).compute()
            # t't't't'
            var4["tttt_var"] += d.tttt.mean(dim=("x","y")).compute()
            # q'w'q'w'
            var4["qwqw_var"] += d.qwqw.mean(dim=("x","y")).compute()
            # q'q'q'q'
            var4["qqqq_var"] += d.qqqq.mean(dim=("x","y")).compute()
            # tv'tv'tv'tv'
            var4["tvtvtvtv_var"] += d.tvtvtvtv.mean(dim=("x","y")).compute()
            # tv'w'tv'w'
            var4["tvwtvw_var"] += d.tvwtvw.mean(dim=("x","y")).compute()
        #
        # calculate autocorrelations for everything
        #
        print("Begin calculating autocorrelations...")
        # loop over all variables in vacf
        for v in vacf:
            # use acf1d and compute
            # if this is first file, create new variable
            if jf == 0:
                R[v] = acf1d(d[v], detrend="constant", poslags=True)
            else:
                R[v] += acf1d(d[v], detrend="constant", poslags=True)

    # OUTSIDE BIG LOOP
    # finish up var4 data
    # loop over variables in var4 and average in time by dividing by number of files
    for v in list(var4.keys()):
        var4[v] /= float(nf)
    # save file
    fsaveV = f"{dnc}variances_4_order.nc"
    if os.path.exists(fsaveV):
        os.system(f"rm {fsaveV}")
    print(f"Saving file: {fsaveV}")
    with ProgressBar():
        var4.to_netcdf(fsaveV, mode="w")

    # finish up autocorr data
    # loop over variables in R and average in time
    for v in list(R.keys()):
        R[v] /= float(nf)
    # save file
    fsaveR = f"{dnc}autocorr.nc"
    if os.path.exists(fsaveR):
        os.system(f"rm {fsaveR}")
    print(f"Saving file: {fsaveR}")
    with ProgressBar():
        R.to_netcdf(fsaveR, mode="w")

    return
# --------------------------------
def construct_filter(delta_x, nx, Lx):
    """Construct filter transfer function: boxcar in physical space. Return
    as numpy array (can easily convert to xarray later)
    -Inputs-
    delta_x: array of filter widths for use with RFM
    nx: number of points in x-direction
    Lx: size of domain in x-direction
    -Output-
    array of filter transfer functions at specified filter widths
    """
    nfilt = len(delta_x)
    # construct box filter transfer function
    dk = 2.*np.pi/Lx
    # empty array shape(nfilt, nx)
    filt = np.zeros((nfilt, nx), dtype=np.float64)
    # loop over filter sizes
    for i, idx in enumerate(delta_x):
        # set zero frequency
        filt[i,0] = 1.
        # loop over frequencies
        for j in range(1, nx//2+1):
            filt[i,j] = np.sin(j*dk*idx/2.) / (j*dk*idx/2.)
            filt[i,nx-j] = np.sin(j*dk*idx/2.) / (j*dk*idx/2.)
    # fftshift so that zero frequency is in center (convention used by xrft)
    # then return
    return fftshift(filt, axes=1)
# --------------------------------
def relaxed_filter(f, filt, delta_x):
    """Perform filtering of f over the range of filter widths within filt
    and calculate variances at each width. Return DataArray of resulting
    -Inputs-
    y- and time-averaged variances.
    f: xr.Dataset, 3d data to filter and calculate variances
    filt: array of filter transfer functions as shape(nfilt, nx)
    delta_x: array of filter widths associated with filt
    """
    # grab some dimensions
    nfilt = len(delta_x)
    nx, ny, nz = f.x.size, f.y.size, f.z.size
    # forward FFT of f
    f_fft = xrft.fft(f, dim="x", true_phase=True, true_amplitude=True)
    # convert filt to DataArray
    filt = xr.DataArray(data=filt, 
                        coords=(dict(delta_x=delta_x, freq_x=f_fft.freq_x)))
    # apply filter in x-wavenumber space: loop over time and filter widths
    # create new DataArray
    f_fft_filt = xr.DataArray(data=np.zeros((nx, ny, nz, nfilt), 
                                            dtype=np.complex128),
                              coords=dict(freq_x=f_fft.freq_x,
                                          y=f.y, z=f.z, delta_x=delta_x))   
    # begin loop over filter widths
    for i in range(nfilt):
        # multiply f_fft by filter function for filter width i
        f_fft_filt[:,:,:,i] = f_fft * filt.isel(delta_x=i)
    # after looping over filter widths, calculate inverse fft on x-axis
    f_ifft_filt = xrft.ifft(f_fft_filt, dim="freq_x", true_phase=True, 
                            true_amplitude=True,
                            lag=f_fft.freq_x.direct_lag)
    # compute variance along x, take mean in y, and return
    return f_ifft_filt.var(dim="x").mean(dim="y").compute()
# --------------------------------
def calc_RFM_var(dnc, fall, s, filt, delta_x):
    """Calculate variances of parameters at each filter width delta_x.
    Save netcdf file in dnc.
    -Inputs-
    dnc: absolute path to netcdf directory for saving new file
    fall: list of strings of all the files to be loaded one-by-one
    filt: filter transfer functions as shape(nfilt, nx)
    delta_x: array of filter widths associated with filt
    -Output-
    RFM.nc
    """
    # initialize empty Dataset for RFM variances
    RFMvar = xr.Dataset(data_vars=None, attrs=s.attrs)
    # get number of files
    nf = len(fall)
    # define list of parameters for looping over for RFM
    vfilt = ["u", "u_rot", "v", "v_rot", "w", "theta", "thetav", "q",
             "uw", "vw", "tw", "qw", "tvw",
             "uu", "uur", "vv", "vvr", "ww", 
             "tt", "qq", "tvtv"]
    # begin looping over all files in fall
    for jf, ff in enumerate(fall):
        # load file
        print(f"Loading file: {ff}")
        d = xr.load_dataset(ff)    
        # calculate thetav
        d["thetav"] = d.theta * (1. + 0.61*d.q/1000.)
        # begin by computing "instantaneous" variances and covariances, store in f
        print("Calculating 'instantaneous' variances and covariances")
        # use the calc_inst_var function
        # u'w'
        d["uw"] = calc_inst_var(d.u, d.w, d.txz)
        # v'w'
        d["vw"] = calc_inst_var(d.v, d.w, d.tyz)
        # t'w'
        d["tw"] = calc_inst_var(d.theta, d.w, d.tw_sgs)
        # u'u'
        d["uu"] = calc_inst_var(d.u, d.u)
        # u'u' rot
        d["uur"] = calc_inst_var(d.u_rot, d.u_rot)
        # v'v'
        d["vv"] = calc_inst_var(d.v, d.v)
        # v'v' rot
        d["vvr"] = calc_inst_var(d.v_rot, d.v_rot)
        # w'w'
        d["ww"] = calc_inst_var(d.w, d.w)
        # t't'
        d["tt"] = calc_inst_var(d.theta, d.theta)
        # tv'tv'
        d["tvtv"] = calc_inst_var(d.thetav, d.thetav)
        # tv'w'
        tvw_sgs = d.tw_sgs + 0.61*d.thetav.isel(z=0).mean()*d.qw_sgs/1000.
        d["tvw"] = calc_inst_var(d.thetav, d.w, tvw_sgs)
        # q'q'
        d["qq"] = calc_inst_var(d.q, d.q)
        # q'w'
        d["qw"] = calc_inst_var(d.q, d.w, d.qw_sgs)
        # begin loop
        for v in vfilt:
            # use relaxed_filter and compute
            print(f"Computing RFM for: {v}")
            # if this is first file, create new variable
            if jf == 0:
                RFMvar[v] = relaxed_filter(d[v], filt, delta_x)
            else:
                RFMvar[v] += relaxed_filter(d[v], filt, delta_x)

    # OUTSIDE LOOP
    # average in time
    for v in list(RFMvar.keys()):
        RFMvar[v] /= float(nf)
    # save RFMvar
    fsaveRFM = f"{dnc}RFM.nc"
    if os.path.exists(fsaveRFM):
        os.system(f"rm {fsaveRFM}")
    print(f"Saving file: {fsaveRFM}")
    with ProgressBar():
        RFMvar.to_netcdf(fsaveRFM, mode="w")

    return
# --------------------------------
def power_law(delta_x, C, p):
    # function to be used with curve_fit
    return C * (delta_x ** (-p))
# --------------------------------
def fit_RFM(dnc, RFM_var, var4, L, s, dx_fit_1, dx_fit_2):
    """Determine the coefficients C and p in the RFM power law
    based on the 2nd/4th-order variances and the RFM variances.
    Save netcdf files for C and p individually.
    -Inputs-
    dnc: directory for saving netcdf file
    RFM_var: xr.Dataset, output from the RFM_var routine
    var4: xr.Dataset, 4th order variances from calc_4_order
    L: xr.Dataset, integral lengthscales
    s: xr.Dataset, mean statistics (mainly for 2nd order var/cov)
    dx_fit_1: pair of values for min and max filter widths\
        to fit power law on 1st order moments
    dx_fit_2: same as dx_fit_1 but for second order moments
    -Output-
    fit_C.nc
    fit_p.nc
    """
    # NOTE: only computing within the ABL! Grab indices here
    nzabl = s.nzabl
    zabl = s.z.isel(z=range(nzabl))
    # create empty datasets C and p
    C = xr.Dataset(data_vars=None, coords=dict(z=zabl), attrs=RFM_var.attrs)
    p = xr.Dataset(data_vars=None, coords=dict(z=zabl), attrs=RFM_var.attrs)
    # define lists of desired parameters and their corresponding variances
    # 1st order - vars from stat file
    vRFM1 = ["u", "u_rot", "v", "v_rot", "theta", "q", "thetav"]
    vvar1 = ["u_var", "u_var_rot", "v_var", "v_var_rot", "theta_var", "q_var", "thetav_var"]
    # 2nd order - vars from var4 file
    vRFM2 = ["uw", "vw", "tw", "uu", "uur", "vv", "vvr", "ww", "tt", "qw", "qq", "tvtv", "tvw"]
    vvar2 = ["uwuw_var", "vwvw_var", "twtw_var", "uuuu_var", "uuuur_var",
             "vvvv_var", "vvvvr_var", "wwww_var", "tttt_var", "qwqw_var", "qqqq_var",
             "tvtvtvtv_var", "tvwtvw_var"]
    # loop through first order moments
    for v, var in zip(vRFM1, vvar1):
        print(f"Calculating power law coefficients for: {v}")
        # construct empty DataArrays for v to store in C and p
        C[v] = xr.DataArray(np.zeros(nzabl, dtype=np.float64),
                            coords=dict(z=zabl))
        p[v] = xr.DataArray(np.zeros(nzabl, dtype=np.float64),
                            coords=dict(z=zabl))
        # loop over heights within abl
        for jz in range(nzabl):
            # grab delta_x ranges based on dx_fit_1 and dx_fit_2
            ix1 = np.where((RFM_var.delta_x  >= dx_fit_1[0]) &\
                           (RFM_var.delta_x  <= dx_fit_1[1]))[0]
            # grab ranges of data for fitting
            # x: delta_x
            xfit = RFM_var.delta_x.isel(delta_x=ix1) #/ L[v].isel(z=jz)
            # y: filtered variance / ensemble variance
            yfit = RFM_var[v].isel(z=jz, delta_x=ix1) / s[var].isel(z=jz)
            # fit x and y to power law
            (C[v][jz], p[v][jz]), _ = curve_fit(f=power_law,
                                                xdata=xfit,
                                                ydata=yfit,
                                                p0=[0.001, 0.001])
    # repeat for second order moments
    for v, var in zip(vRFM2, vvar2):
        print(f"Calculating power law coefficients for: {v}")
        # construct empty DataArrays for v to store in C and p
        C[v] = xr.DataArray(np.zeros(nzabl, dtype=np.float64),
                            coords=dict(z=zabl))
        p[v] = xr.DataArray(np.zeros(nzabl, dtype=np.float64),
                            coords=dict(z=zabl))
        # loop over heights within abl
        for jz in range(nzabl):
            ix2 = np.where((RFM_var.delta_x >= dx_fit_2[0]) &\
                           (RFM_var.delta_x <= dx_fit_2[1]))[0]
            # grab ranges of data for fitting
            # x: delta_x
            xfit = RFM_var.delta_x.isel(delta_x=ix2) #/ L[v].isel(z=jz)
            # y: filtered variance / ensemble variance
            yfit = RFM_var[v].isel(z=jz, delta_x=ix2) / var4[var].isel(z=jz)
            # fit x and y to power law
            (C[v][jz], p[v][jz]), _ = curve_fit(f=power_law,
                                                xdata=xfit,
                                                ydata=yfit,
                                                p0=[0.001, 0.001])
    #
    # Save C and p as netcdf files
    #
    # C
    fsaveC = f"{dnc}fit_C.nc"
    if os.path.exists(fsaveC):
        os.system(f"rm {fsaveC}")
    print(f"Saving file: {fsaveC}")
    with ProgressBar():
        C.to_netcdf(fsaveC, mode="w") 
    # p
    fsavep = f"{dnc}fit_p.nc"
    if os.path.exists(fsavep):
        os.system(f"rm {fsavep}")
    print(f"Saving file: {fsavep}")
    with ProgressBar():
        p.to_netcdf(fsavep, mode="w")
    
    return
# --------------------------------
def calc_error(dnc, T1, T2, var4, s, C, p, L):
    """Calculate the relative random errors given the RFM coefficients
    C and p for the desired averaging times T1, T2 for 1st and 2nd order.
    For comparison, also calculate Lumley-Panofsky formulation.
    Save netcdf file in dnc.
    -Input-
    dnc: directory for saving netcdf file
    T1: averaging time of 1st order moments
    T2: averaging time of 2nd order moments
    var4: 4th order variances from calc_4_order
    s: mean statistics (mainly for 2nd order var/cov)
    C: coefficients from fit_RFM
    p: coefficients from fit_RFM
    L: integral lengthscales to calc LP errors
    -Output-
    err.nc
    err_LP.nc
    """
    # grab abl indices
    nzabl = s.nzabl
    # use Taylor hypothesis to convert time to space
    X1 = s.uh.isel(z=range(nzabl)) * T1
    X2 = s.uh.isel(z=range(nzabl)) * T2
    # create xarray Datasets for MSE and err
    MSE = xr.Dataset(data_vars=None, coords=dict(z=C.z), attrs=C.attrs)
    err = xr.Dataset(data_vars=None, coords=dict(z=C.z), attrs=C.attrs)
    # define lists of desired parameters and their corresponding variances
    # 1st order - vars from stat file
    vRFM1 = ["u", "u_rot", "v", "v_rot", "theta", "q", "thetav"]
    vvar1 = ["u_var", "u_var_rot", "v_var", "v_var_rot", "theta_var", "q_var", "thetav_var"]
    vavg1 = ["u_mean", "u_mean_rot", "v_mean", "v_mean_rot", "theta_mean", "q_mean", "thetav_mean"]
    # 2nd order - vars from var4 file
    vRFM2 = ["uw", "vw", "tw", "uu", "uur", "vv", "vvr", "ww", "tt", "qw", "qq", "tvtv", "tvw"]
    vvar2 = ["uwuw_var", "vwvw_var", "twtw_var", "uuuu_var", "uuuur_var",
             "vvvv_var", "vvvvr_var", "wwww_var", "tttt_var", "qwqw_var", "qqqq_var", 
             "tvtvtvtv_var", "tvwtvw_var"]
    vavg2 = ["uw_cov_tot", "vw_cov_tot", "tw_cov_tot", "u_var", "u_var_rot",
             "v_var", "v_var_rot", "w_var", "theta_var", "qw_cov_tot", "q_var", 
             "thetav_var", "tvw_cov_tot"]   
    # Begin calculating errors
    # Start with 1st order
    for v, var, avg in zip(vRFM1, vvar1, vavg1):
        print(f"Computing errors for: {v}")
        # use values of C and p to extrapolate calculation of MSE/var{x}
        # renormalize with variances in vvar1, Lengthscales in vRFM1
        X1_L = X1 #/ L[v].isel(z=range(nzabl))
        MSE[v] = s[var].isel(z=range(nzabl)) * (C[v] * (X1_L**-p[v]))
        # take sqrt to get RMSE
        RMSE = np.sqrt(MSE[v])
        # divide by <x> to get epsilon (relative random error)
        err[v] = RMSE / abs(s[avg].isel(z=range(nzabl)))
    # 2nd order
    for v, var, avg in zip(vRFM2, vvar2, vavg2):
        print(f"Computing errors for: {v}")
        # use values of C and p to extrapolate calculation of MSE/var{x}
        # renormalize with variances in vvar1, Lengthscales in vRFM2
        X2_L = X2 #/ L[v].isel(z=range(nzabl))
        MSE[v] = var4[var].isel(z=range(nzabl)) * (C[v] * (X2_L**-p[v]))
        # take sqrt to get RMSE
        RMSE = np.sqrt(MSE[v])
        # divide by <x> to get epsilon (relative random error)
        err[v] = RMSE / abs(s[avg].isel(z=range(nzabl)))
    # calculate errors in wind speed and direction using error propagation
    # grab individual RMSEs
    sig_u = np.sqrt(MSE["u"])
    sig_v = np.sqrt(MSE["v"])
    # calculate wspd error and store
    err["wspd"] = np.sqrt( (sig_u**2. * s.u_mean.isel(z=range(nzabl))**2. +\
                            sig_v**2. * s.v_mean.isel(z=range(nzabl))**2.)/\
                           (s.u_mean.isel(z=range(nzabl))**2. +\
                            s.v_mean.isel(z=range(nzabl))**2.) ) /\
                  s.uh.isel(z=range(nzabl))
    # calculate wdir error and store
    # first get wdir in radians
    wdir = s.wdir * np.pi/180.
    err["wdir"] = np.sqrt( (sig_u**2. * s.u_mean.isel(z=range(nzabl))**2. +\
                            sig_v**2. * s.v_mean.isel(z=range(nzabl))**2.)/\
                           ((s.u_mean.isel(z=range(nzabl))**2. +\
                            s.v_mean.isel(z=range(nzabl))**2.)**2.) ) /\
                  wdir.isel(z=range(nzabl))
    # calculate errors in ustar^2 for coordinate-agnostic horiz Reynolds stress
    # ustar2 = (<u'w'>^2 + <v'w'>^2) ^ 1/2
    sig_uw = np.sqrt(MSE["uw"])
    sig_vw = np.sqrt(MSE["vw"])
    err["ustar2"] = np.sqrt( (sig_uw**2. * s.uw_cov_tot.isel(z=range(nzabl))**2. +\
                              sig_vw**2. * s.vw_cov_tot.isel(z=range(nzabl))**2.)/\
                              (s.ustar2.isel(z=range(nzabl))**2.)) / \
                    s.ustar2.isel(z=range(nzabl))
    #
    # Save as netcdf file
    #
    fsave_err = f"{dnc}err.nc"
    if os.path.exists(fsave_err):
        os.system(f"rm {fsave_err}")
    print(f"Saving file: {fsave_err}")
    with ProgressBar():
        err.to_netcdf(fsave_err, mode="w")     

    # calculate error from Lumley and Panofsky for given sample times
    # err_LP = sqrt[(2*int_lengh*ens_variance)/(ens_mean^2*sample_length)]
    print("Calculate LP relative random errors...")
    # empty Dataset
    err_LP = xr.Dataset(data_vars=None, coords=dict(z=C.z), attrs=C.attrs)
    # loop through first-order moments
    for v, var, avg in zip(vRFM1, vvar1, vavg1):
        err_LP[v] = np.sqrt((2. * L[v] * s[var].isel(z=range(nzabl)))/\
                            (X1 * s[avg].isel(z=range(nzabl))**2.))
    # loop through second-order moments
    for v, var, avg in zip(vRFM2, vvar2, vavg2):
        err_LP[v] = np.sqrt((2. * L[v] * var4[var].isel(z=range(nzabl)))/\
                            (X2 * s[avg].isel(z=range(nzabl))**2.))
    # again
    # calculate errors in wind speed and direction using error propagation
    # grab individual RMSEs
    sig_u = err_LP["u"] * s.u_mean.isel(z=range(nzabl))
    sig_v = err_LP["v"] * s.v_mean.isel(z=range(nzabl))
    # calculate wspd error and store
    err_LP["wspd"] = np.sqrt( (sig_u**2. * s.u_mean.isel(z=range(nzabl))**2. +\
                               sig_v**2. * s.v_mean.isel(z=range(nzabl))**2.)/\
                              (s.u_mean.isel(z=range(nzabl))**2. +\
                               s.v_mean.isel(z=range(nzabl))**2.) ) /\
                     s.uh.isel(z=range(nzabl))
    # calculate wdir error and store
    # first get wdir in radians
    wdir = s.wdir * np.pi/180.
    err_LP["wdir"] = np.sqrt( (sig_u**2. * s.u_mean.isel(z=range(nzabl))**2. +\
                               sig_v**2. * s.v_mean.isel(z=range(nzabl))**2.)/\
                              ((s.u_mean.isel(z=range(nzabl))**2. +\
                                s.v_mean.isel(z=range(nzabl))**2.)**2.) ) /\
                     wdir.isel(z=range(nzabl))
    # calculate errors in ustar^2 for coordinate-agnostic horiz Reynolds stress
    # ustar2 = (<u'w'>^2 + <v'w'>^2) ^ 1/2
    sig_uw = err_LP["uw"] * s.uw_cov_tot.isel(z=range(nzabl))
    sig_vw = err_LP["vw"] * s.vw_cov_tot.isel(z=range(nzabl))
    err_LP["ustar2"] = np.sqrt( (sig_uw**2. * s.uw_cov_tot.isel(z=range(nzabl))**2. +\
                                 sig_vw**2. * s.vw_cov_tot.isel(z=range(nzabl))**2.)/\
                                (s.ustar2.isel(z=range(nzabl))**2.)) / \
                       s.ustar2.isel(z=range(nzabl))

    # save err_LP as netcdf
    fsave_LP = f"{dnc}err_LP.nc"
    if os.path.exists(fsave_LP):
        os.system(f"rm {fsave_LP}")
    print(f"Saving file: {fsave_LP}")
    with ProgressBar():
        err_LP.to_netcdf(fsave_LP, mode="w")
    
    return
# --------------------------------
def recalc_err_array(Tnew, s, C, p, L):
    """Calculate the relative random errors given the RFM coefficients
    C and p for a range of desired averaging times T.
    This will only process the following 1st order moments:
    u, v, theta, q, thetav
    -Input-
    Tnew: array of averaging time of 1st order moments
    var4: 4th order variances from calc_4_order
    s: mean statistics (mainly for 2nd order var/cov)
    C: coefficients from fit_RFM
    p: coefficients from fit_RFM
    L: integral lengthscales
    -Output-
    err: xr.Dataset
    """
    # grab abl indices
    nzabl = s.nzabl
    # create xarray Datasets for MSE and err
    MSE = xr.Dataset(data_vars=None, 
                     coords=dict(z=C.z, Tsample=Tnew), 
                     attrs=C.attrs)
    err = xr.Dataset(data_vars=None, 
                     coords=dict(z=C.z, Tsample=Tnew), 
                     attrs=C.attrs)
    # 1st order - vars from stat file
    vRFM1 = ["u", "v", "theta", "q", "thetav"]
    vvar1 = ["u_var", "v_var", "theta_var", "q_var", "thetav_var"]
    vavg1 = ["u_mean", "v_mean", "theta_mean", "q_mean", "thetav_mean"]
    # Begin calculating errors
    # Start with 1st order
    for v, var, avg in zip(vRFM1, vvar1, vavg1):
        print(f"Computing errors for: {v}")
        # create temp MSE DataArray
        MSE[v] = xr.DataArray(data=np.zeros((nzabl, len(Tnew)), 
                                            dtype=np.float64),
                              coords=dict(err.coords))
        # loop over Tnew to calculate MSE(z,Tsample)
        for jt, iT in enumerate(Tnew):
            # use Taylor hypothesis to convert time to space
            X1 = s.uh.isel(z=range(nzabl)) * iT
            # use values of C and p to extrapolate calculation of MSE/var{x}
            # renormalize with variances in vvar1, Lengthscales in vRFM1
            X1_L = X1 #/ L[v].isel(z=range(nzabl))
            MSE[v][:,jt] = s[var].isel(z=range(nzabl)) * (C[v] * (X1_L**-p[v]))
        # take sqrt to get RMSE
        RMSE = np.sqrt(MSE[v])
        # divide by <x> to get epsilon (relative random error)
        err[v] = RMSE / abs(s[avg].isel(z=range(nzabl)))

    # calculate errors in wind speed and direction using error propagation
    # grab individual RMSEs
    sig_u = np.sqrt(MSE["u"])
    sig_v = np.sqrt(MSE["v"])
    # calculate wspd error and store
    err["wspd"] = np.sqrt( (sig_u**2. * s.u_mean.isel(z=range(nzabl))**2. +\
                            sig_v**2. * s.v_mean.isel(z=range(nzabl))**2.)/\
                           (s.u_mean.isel(z=range(nzabl))**2. +\
                            s.v_mean.isel(z=range(nzabl))**2.) ) /\
                  s.uh.isel(z=range(nzabl))
    # calculate wdir error and store
    # first get wdir in radians
    wdir = s.wdir * np.pi/180.
    err["wdir"] = np.sqrt( (sig_u**2. * s.u_mean.isel(z=range(nzabl))**2. +\
                            sig_v**2. * s.v_mean.isel(z=range(nzabl))**2.)/\
                           ((s.u_mean.isel(z=range(nzabl))**2. +\
                            s.v_mean.isel(z=range(nzabl))**2.)**2.) ) /\
                  wdir.isel(z=range(nzabl))
    # finished, now return
    return err

# --------------------------------
# main loop
if __name__ == "__main__":
    # home directory for data
    # dnc = "/home/bgreene/simulations/RFM/u01_tw24_qw10_256/"
    # dnc = "/home/bgreene/simulations/RFM/u09_tw24_qw10_256/"
    # dnc = "/home/bgreene/simulations/RFM/u15_tw03_qw01_256/"
    # dnc = "/home/bgreene/simulations/RFM/u15_tw10_qw04_256/"
    # dnc = "/home/bgreene/simulations/RFM/u15_tw24_qw10_256/"
    
    # arguments for simulation directory to process
    parser = ArgumentParser()
    parser.add_argument("-d", required=True, action="store", dest="dsbl", nargs=1,
                        help="Simulation base directory")
    parser.add_argument("-s", required=True, action="store", dest="sim", nargs=1,
                        help="Simulation name")
    args = parser.parse_args()

    # construct simulation directory and ncdir
    dnc = os.path.join(args.dsbl[0], args.sim[0]) + os.sep

    # sim timesteps to consider
    t0 = 450000
    t1 = 540000
    dt = 1000
    fstats = "mean_stats_xyt_5-6h.nc"
    timesteps = np.arange(t0, t1+1, dt, dtype=np.int32)
    # determine filenames
    fall = [f"{dnc}all_{tt:07d}_rot.nc" for tt in timesteps]
    # load stats file
    s = load_stats(dnc+fstats)

    # calculate and load 4th order variances
    # calc_4_order(dnc, fall, s)
    # load again
    var4 = xr.load_dataset(dnc+"variances_4_order.nc")
    R = xr.load_dataset(dnc+"autocorr.nc")

    # filtering routines
    # define filter widths
    nfilt = 50 # yaml file later
    delta_x = np.logspace(np.log10(s.dx), np.log10(s.Lx), 
                          num=nfilt, base=10.0, dtype=np.float64)
    # construct filter transfer function
    # filt = construct_filter(delta_x, s.nx, s.Lx)
    # call RFM
    # calc_RFM_var(dnc, fall, s, filt, delta_x)
    # load back this data
    RFM_var = xr.load_dataset(dnc+"RFM.nc")

    # compute integral lengthscales using calc_lengthscale
    # print("Calculating lengthscales")
    # calc_lengthscale(dnc, R)
    # load
    L = xr.load_dataset(dnc+"lengthscale.nc")
    # quickly need to correct these integral scales
    for key in L.keys():
        # loop over z
        for jz in range(L.z.size):
            if L[key][jz] == 0.:
                L[key][jz] = R[key].isel(z=jz, x=range(2)).integrate("x")

    # next step: calculate/fit RFM coefficients to normalized variances
    # feed function var4 and RFM_var, stats file
    dx_fit_1 = [800, 3000] # use RFM_test.ipynb to explore this
    dx_fit_2 = [800, 3000] # appears to be valid for both 1 and 2
    fit_RFM(dnc, RFM_var, var4, L, s, dx_fit_1, dx_fit_2)
    # load C, p
    C = xr.load_dataset(dnc+"fit_C.nc")
    p = xr.load_dataset(dnc+"fit_p.nc")

    # calculate errors
    T1 = 5.    # seconds
    T2 = 1800. # seconds, = 30 min
    calc_error(dnc, T1, T2, var4, s, C, p, L)
