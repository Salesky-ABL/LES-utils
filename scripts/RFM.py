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
from spec import acf1d
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
        vacf += ["q", "qw", "qwqw", "qqqq"]
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
    if use_q:
        df["qw"] = calc_inst_var(df.q, df.w, df.wq_sgs)
    # only going to care about rotated variances in u, v
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
    # initialize empty Dataset for autocorrs
    RFMvar = xr.Dataset(data_vars=None, attrs=df.attrs)
    # define list of parameters for looping
    vfilt = ["u", "u_rot", "v", "v_rot", "w", "theta",
             "uw", "vw", "tw",
             "uu", "uur", "vv", "vvr", "ww", "tt"]
    if use_q:
        vfilt += ["q", "qw"]
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
# main loop
if __name__ == "__main__":
    dnc = "/home/bgreene/simulations/u15_tw10_qw04_dry3/output/netcdf/"
    dd, s = load_full(dnc, 360000, 432000, 1000, 0.05, False, 
                      "mean_stats_xyt_5-6h.nc", True)
    
    # calculate and/or load 4th order variances
    fvar4 = dnc+"variances_4_order.nc"
    if not os.path.exists(fvar4):
        calc_4_order(dnc, dd, True)
    var4 = xr.load_dataset(fvar4)
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