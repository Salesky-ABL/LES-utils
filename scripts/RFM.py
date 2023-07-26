#!/home/bgreene/anaconda3/bin/python
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
import yaml
import xrft
import numpy as np
import xarray as xr
from scipy.fft import fft, ifft, fft2, ifft2, fftfreq, fftshift
from scipy.optimize import curve_fit
from dask.diagnostics import ProgressBar
from LESutils import load_full
from spec import acf1d, calc_lengthscale
# --------------------------------
def calc_inst_var(d1, d2, SGS=None):
    """Calculate 'instantaneous' variance/covariance between 2 parameters
    plus optional SGS term by calculating perturbations (x,y) of d1, d2 then
    multiplying resulting fields
    :param DataArray d1: parameter 1 to use
    :param DataArray d2: parameter 2 to use
    :param DataArray SGS: optional subgrid term to include
    return single DataArray in (time,x,y,z) 
    """
    d1p = d1 - d1.mean(dim=("x","y"))
    d2p = d2 - d2.mean(dim=("x","y"))
    d1d2_cov = d1p * d2p
    if SGS is not None:
        return d1d2_cov + SGS
    else:
        return d1d2_cov
# --------------------------------
def calc_4_order(dnc, df, use_q=True):
    """Calculate 4th-order variances (u'w'u'w', u'u'u'u', etc.) for use
    in fitting RFM curves as well as computing LP errors later.
    Also calculate 1d autocorrelation function for _all_ parameters here. 
    Save netcdf file in dnc.
    :param str dnc: absolute path to netcdf directory for saving new file
    :param Dataset df: 4d (time,x,y,z) xarray Dataset for calculating
    :param bool use_q: flag for including humidity variables in calculations
    """
    # begin by computing "instantaneous" variances and covariances, store in df
    print("Calculating 'instantaneous' variances and covariances")
    # use the calc_inst_var function
    # u'w'
    df["uw"] = calc_inst_var(df.u, df.w, df.txz)
    # v'w'
    df["vw"] = calc_inst_var(df.v, df.w, df.tyz)
    # t'w'
    df["tw"] = calc_inst_var(df.theta, df.w, df.q3)
    if use_q:
        df["qw"] = calc_inst_var(df.q, df.w, df.wq_sgs)
    # need both rotated and unrotated to get wspd, wdir stats later
    # u'u'
    df["uu"] = calc_inst_var(df.u, df.u)
    # u'u' rot
    df["uur"] = calc_inst_var(df.u_rot, df.u_rot)
    # v'v'
    df["vv"] = calc_inst_var(df.v, df.v)
    # v'v' rot
    df["vvr"] = calc_inst_var(df.v_rot, df.v_rot)
    # w'w'
    df["ww"] = calc_inst_var(df.w, df.w)
    # t't'
    df["tt"] = calc_inst_var(df.theta, df.theta)
    if use_q:
        df["qq"] = calc_inst_var(df.q, df.q)
    #
    # Using instantaneous vars/covars, calculate inst 4th order variances
    #
    print("Calculating 4th order variances")
    # u'w'u'w'
    df["uwuw"] = calc_inst_var(df.uw, df.uw)
    # v'w'v'w'
    df["vwvw"] = calc_inst_var(df.vw, df.vw)
    # t'w't'w'
    df["twtw"] = calc_inst_var(df.tw, df.tw)
    # u'u'u'u'
    df["uuuu"] = calc_inst_var(df.uu, df.uu)
    # u'u'u'u' rot
    df["uuuur"] = calc_inst_var(df.uur, df.uur)
    # v'v'v'v'
    df["vvvv"] = calc_inst_var(df.vv, df.vv)
    # v'v'v'v' rot
    df["vvvvr"] = calc_inst_var(df.vvr, df.vvr)
    # w'w'w'w'
    df["wwww"] = calc_inst_var(df.ww, df.ww)
    # t't't't'
    df["tttt"] = calc_inst_var(df.tt, df.tt)
    # q'q'q'q', q'w'q'w'
    if use_q:
        df["qwqw"] = calc_inst_var(df.qw, df.qw)
        df["qqqq"] = calc_inst_var(df.qq, df.qq)
    # define new Dataset to hold all the mean 4th order variances
    var4 = xr.Dataset(data_vars=None, coords=dict(z=df.z), attrs=df.attrs)
    # u'w'u'w' = var{u'w'} = (u'w' - <u'w'>_xyt)**2
    var4["uwuw_var"] = df.uwuw.mean(dim=("time","x","y"))
    # v'w'v'w'
    var4["vwvw_var"] = df.vwvw.mean(dim=("time","x","y"))
    # t'w't'w'
    var4["twtw_var"] = df.twtw.mean(dim=("time","x","y"))
    # u'u'u'u'
    var4["uuuu_var"] = df.uuuu.mean(dim=("time","x","y"))
    # u'u'u'u' rot
    var4["uuuur_var"] = df.uuuur.mean(dim=("time","x","y"))
    # v'v'v'v'
    var4["vvvv_var"] = df.vvvv.mean(dim=("time","x","y"))
    # v'v'v'v' rot
    var4["vvvvr_var"] = df.vvvvr.mean(dim=("time","x","y"))
    # w'w'w'w'
    var4["wwww_var"] = df.wwww.mean(dim=("time","x","y"))
    # t't't't'
    var4["tttt_var"] = df.tttt.mean(dim=("time","x","y"))
    if use_q:
        var4["qwqw_var"] = df.qwqw.mean(dim=("time","x","y"))
        var4["qqqq_var"] = df.qqqq.mean(dim=("time","x","y"))
    # save file
    fsaveV = f"{dnc}variances_4_order.nc"
    if os.path.exists(fsaveV):
        os.system(f"rm {fsaveV}")
    print(f"Saving file: {fsaveV}")
    with ProgressBar():
        var4.to_netcdf(fsaveV, mode="w")

    #
    # calculate autocorrelations for everything
    #
    print("Begin calculating autocorrelations...")
    # initialize empty Dataset for autocorrs
    R = xr.Dataset(data_vars=None, attrs=df.attrs)
    # define list of parameters for looping
    vacf = ["u", "u_rot", "v", "v_rot", "w", "theta",
            "uw", "vw", "tw",
            "uu", "uur", "vv", "vvr", "ww", "tt",
            "uwuw", "vwvw", "twtw",
            "uuuu", "uuuur", "vvvv", "vvvvr", "wwww", "tttt"]
    if use_q:
        vacf += ["q", "qw", "qq", "qwqw", "qqqq"]
    # begin loop
    for v in vacf:
        # use acf1d and compute
        R[v] = acf1d(df[v], detrend="constant", poslags=True)
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
    :param np.array delta_x: array of filter widths for use with RFM
    :param int nx: number of points in x-direction
    :param float Lx: size of domain in x-direction
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
    y- and time-averaged variances.
    :param DataArray f: 4d data to filter and calculate variances
    :param array filt: filter transfer functions as shape(nfilt, nx)
    :param array delta_x: array of filter widths associated with filt
    """
    # grab some dimensions
    nfilt = len(delta_x)
    nx, ny, nz = f.x.size, f.y.size, f.z.size
    # define empty DataArray for returning later
    var_f_all = xr.DataArray(data=np.zeros((nfilt, nz), dtype=np.float64),
                             coords=dict(delta_x=delta_x, z=f.z))
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
    # begin time loop
    for jt in range(f.time.size):
        for i in range(nfilt):
            # multiply f_fft by filter function for filter width i
            f_fft_filt[:,:,:,i] = f_fft.isel(time=jt) * filt.isel(delta_x=i)
        # after looping over filter widths, calculate inverse fft on x-axis
        f_ifft_filt = xrft.ifft(f_fft_filt, dim="freq_x", true_phase=True, 
                                true_amplitude=True,
                                lag=f_fft.freq_x.direct_lag)
        var_f_all += f_ifft_filt.var(dim="x").mean(dim="y")
    # divide by nt to get time-averaged sigma_f_all
    # return
    return var_f_all / float(f.time.size)
# --------------------------------
def calc_RFM_var(dnc, df, use_q, filt, delta_x):
    """Calculate variances of parameters at each filter width delta_x.
    Save netcdf file in dnc.
    :param str dnc: absolute path to netcdf directory for saving new file
    :param Dataset df: 4d (time,x,y,z) xarray Dataset for calculating
    :param bool use_q: flag for including humidity variables in calculations
    :param array filt: filter transfer functions as shape(nfilt, nx)
    :param array delta_x: array of filter widths associated with filt    
    """
    # begin by computing "instantaneous" variances and covariances, store in df
    print("Calculating 'instantaneous' variances and covariances")
    # use the calc_inst_var function
    # u'w'
    df["uw"] = calc_inst_var(df.u, df.w, df.txz)
    # v'w'
    df["vw"] = calc_inst_var(df.v, df.w, df.tyz)
    # t'w'
    df["tw"] = calc_inst_var(df.theta, df.w, df.q3)
    # u'u'
    df["uu"] = calc_inst_var(df.u, df.u)
    # u'u' rot
    df["uur"] = calc_inst_var(df.u_rot, df.u_rot)
    # v'v'
    df["vv"] = calc_inst_var(df.v, df.v)
    # v'v' rot
    df["vvr"] = calc_inst_var(df.v_rot, df.v_rot)
    # w'w'
    df["ww"] = calc_inst_var(df.w, df.w)
    # t't'
    df["tt"] = calc_inst_var(df.theta, df.theta)
    if use_q:
        df["qq"] = calc_inst_var(df.q, df.q)
        df["qw"] = calc_inst_var(df.q, df.w, df.wq_sgs)
    # initialize empty Dataset for autocorrs
    RFMvar = xr.Dataset(data_vars=None, attrs=df.attrs)
    # define list of parameters for looping
    vfilt = ["u", "u_rot", "v", "v_rot", "w", "theta",
             "uw", "vw", "tw",
             "uu", "uur", "vv", "vvr", "ww", "tt"]
    if use_q:
        vfilt += ["q", "qw", "qq"]
    # begin loop
    for v in vfilt:
        # use relaxed_filter and compute
        print(f"Computing RFM for: {v}")
        RFMvar[v] = relaxed_filter(df[v], filt, delta_x)
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
def fit_RFM(dnc, RFM_var, var4, s, dx_fit_1, dx_fit_2):
    """Determine the coefficients C and p in the RFM power law
    based on the 2nd/4th-order variances and the RFM variances.
    Save netcdf files for C and p individually.
    :param str dnc: directory for saving netcdf file
    :param Dataset RFM_var: output from the RFM_var routine
    :param Dataset var4: 4th order variances from calc_4_order
    :param Dataset s: mean statistics (mainly for 2nd order var/cov)
    :param list dx_fit_1: pair of values for min and max filter widths\
        to fit power law on 1st order moments
    :param list dx_fit_2: same as dx_fit_1 but for second order moments
    """
    # NOTE: only computing within the ABL! Grab indices here
    nzabl = s.nzabl
    zabl = s.z.isel(z=range(nzabl))
    # create empty datasets C and p
    C = xr.Dataset(data_vars=None, coords=dict(z=zabl), attrs=RFM_var.attrs)
    p = xr.Dataset(data_vars=None, coords=dict(z=zabl), attrs=RFM_var.attrs)
    # define lists of desired parameters and their corresponding variances
    # 1st order - vars from stat file
    vRFM1 = ["u", "u_rot", "v", "v_rot", "theta"]
    vvar1 = ["u_var, u_var_rot", "v_var", "v_var_rot", "theta_var"]
    # 2nd order - vars from var4 file
    vRFM2 = ["uw", "vw", "tw", "uu", "uur", "vv", "vvr", "ww", "tt"]
    vvar2 = ["uwuw_var", "vwvw_var", "twtw_var", "uuuu_var", "uuuur_var",
             "vvvv_var", "vvvvr_var", "wwww_var", "tttt_var"]
    # check for q in RFM_var and add to all lists above
    if "q_mean" in list(s.keys()):
        # 1
        vRFM1.append("q")
        vvar1.append("q_var")
        # 2
        vRFM2 += ["qw", "qq"]
        vvar2 += ["qwqw_var", "qqqq_var"]
    # grab delta_x ranges based on dx_fit_1 and dx_fit_2
    ix1 = np.where((RFM_var.delta_x >= dx_fit_1[0]) &\
                   (RFM_var.delta_x <= dx_fit_1[1]))[0]
    ix2 = np.where((RFM_var.delta_x >= dx_fit_2[0]) &\
                   (RFM_var.delta_x <= dx_fit_2[1]))[0]
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
            # grab ranges of data for fitting
            # x: delta_x
            xfit = RFM_var.delta_x.isel(delta_x=ix1)
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
            # grab ranges of data for fitting
            # x: delta_x
            xfit = RFM_var.delta_x.isel(delta_x=ix2)
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
    :param str dnc: directory for saving netcdf file
    :param float T1: averaging time of 1st order moments
    :param float T2: averaging time of 2nd order moments
    :param Dataset var4: 4th order variances from calc_4_order
    :param Dataset s: mean statistics (mainly for 2nd order var/cov)
    :param Dataset C: coefficients from fit_RFM
    :param Dataset p: coefficients from fit_RFM
    :param Dataset L: integral lengthscales to calc LP errors
    """
    # grab abl indices
    nzabl = s.nzabl
    zabl = s.z.isel(z=range(nzabl))
    # use Taylor hypothesis to convert time to space
    X1 = s.uh.isel(z=range(nzabl)) * T1
    X2 = s.uh.isel(z=range(nzabl)) * T2
    # create xarray Datasets for MSE and err
    MSE = xr.Dataset(data_vars=None, coords=dict(z=C.z), attrs=C.attrs)
    err = xr.Dataset(data_vars=None, coords=dict(z=C.z), attrs=C.attrs)
    # define lists of desired parameters and their corresponding variances
    # 1st order - vars from stat file
    vRFM1 = ["u", "u_rot", "v", "v_rot", "theta"]
    vvar1 = ["u_var, u_var_rot", "v_var", "v_var_rot", "theta_var"]
    vavg1 = ["u_mean", "u_mean_rot", "v_mean", "v_mean_rot", "theta_mean"]
    # 2nd order - vars from var4 file
    vRFM2 = ["uw", "vw", "tw", "uu", "uur", "vv", "vvr", "ww", "tt"]
    vvar2 = ["uwuw_var", "vwvw_var", "twtw_var", "uuuu_var", "uuuur_var",
             "vvvv_var", "vvvvr_var", "wwww_var", "tttt_var"]
    vavg2 = ["uw_cov_tot", "vw_cov_tot", "tw_cov_tot", "u_var", "u_var_rot",
             "v_var", "v_var_rot", "w_var", "theta_var"]   
    # check for q in RFM_var and add to all lists above
    if "q_mean" in list(s.keys()):
        # 1
        vRFM1.append("q")
        vvar1.append("q_var")
        vavg1.append("q_mean")
        # 2
        vRFM2 += ["qw", "qq"]
        vvar2 += ["qwqw_var", "qqqq_var"]
        vavg2 += ["qw_cov_tot", "q_var"]
    # Begin calculating errors
    # Start with 1st order
    for v, var, avg in zip(vRFM1, vvar1, vavg1):
        print(f"Computing errors for: {v}")
        # use values of C and p to extrapolate calculation of MSE/var{x}
        # renormalize with variances in vvar1
        MSE[v] = s[var].isel(z=range(nzabl)) * (C[v] * (X1**-p[v]))
        # take sqrt to get RMSE
        RMSE = np.sqrt(MSE[v])
        # divide by <x> to get epsilon (relative random error)
        err[v] = RMSE / abs(s[avg].isel(z=range(nzabl)))
    # 2nd order
    for v, var, avg in zip(vRFM2, vvar2, vavg2):
        print(f"Computing errors for: {v}")
        # use values of C and p to extrapolate calculation of MSE/var{x}
        # renormalize with variances in vvar1
        MSE[v] = var4[var].isel(z=range(nzabl)) * (C[v] * (X2**-p[v]))
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
    sig_uw = np.sqrt(MSE["uw_cov_tot"])
    sig_vw = np.sqrt(MSE["vw_cov_tot"])
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
                            (X1 * s[avg].isel(z=range(nzabl))**2.))
    # save err_LP as netcdf
    fsave_LP = f"{dnc}err_LP.nc"
    if os.path.exists(fsave_LP):
        os.system(f"rm {fsave_LP}")
    print(f"Saving file: {fsave_LP}")
    with ProgressBar():
        err_LP.to_netcdf(fsave_LP, mode="w")
    
    return

# --------------------------------
# main loop
if __name__ == "__main__":
    dnc = "/home/bgreene/simulations/u15_tw10_qw04_dry3/output/netcdf/"
    dd, s = load_full(dnc, 360000, 432000, 1000, 0.05, False, 
                      "mean_stats_xyt_5-6h_rot.nc", True)
    
    # calculate and load 4th order variances
    calc_4_order(dnc, dd, True)
    # load again
    var4 = xr.load_dataset(dnc+"variances_4_order.nc")
    R = xr.load_dataset(dnc+"autocorr.nc")

    # filtering routines
    # define filter widths
    nfilt = 50 # yaml file later
    delta_x = np.logspace(np.log10(dd.dx), np.log10(dd.Lx), 
                          num=nfilt, base=10.0, dtype=np.float64)
    # construct filter transfer function
    filt = construct_filter(delta_x, dd.nx, dd.Lx)
    # call RFM
    calc_RFM_var(dnc, dd, True, filt, delta_x)
    # load back this data
    RFM_var = xr.load_dataset(dnc+"RFM.nc")

    # next step: calculate/fit RFM coefficients to normalized variances
    # feed function var4 and RFM_var, stats file
    dx_fit_1 = [2000, 5000] # use RFM_test.ipynb to explore this
    dx_fit_2 = [2000, 5000] # appears to be valid for both 1 and 2
    fit_RFM(dnc, RFM_var, var4, s, dx_fit_1, dx_fit_2)
    # load C, p
    C = xr.load_dataset(dnc+"fit_C.nc")
    p = xr.load_dataset(dnc+"fit_p.nc")

    # compute integral lengthscales using calc_lengthscale
    calc_lengthscale(dnc, R)
    # load
    L = xr.load_dataset(dnc+"lengthscale.nc")

    # calculate errors
    T1 = 5.    # seconds
    T2 = 1800. # seconds, = 30 min
    calc_error(dnc, T1, T2, var4, s, C, p, L)