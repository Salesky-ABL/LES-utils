#!/home/bgreene/anaconda3/bin/python
# --------------------------------
# Name: LESutils.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 1 Febuary 2023
# Purpose: collection of functions for use in
# processing raw output by Salesky LES code as well as
# calculating commonly-used statistics
# --------------------------------
import os
import yaml
import xrft
import xarray as xr
import numpy as np
from numba import njit
from dask.diagnostics import ProgressBar
from scipy.interpolate import RegularGridInterpolator
# --------------------------------
# Begin Defining Functions
# --------------------------------
def read_f90_bin(path, nx, ny, nz, precision):
    """Read raw binary files output by LES fortran code

    :param str path: absolute path to binary file
    :param int nx: number of points in streamwise dimension
    :param int ny: number of points in spanwise dimension
    :param int nz: number of points in wall-normal dimension
    :param int precision: precision of floating-point values (must be 4 or 8)
    :return: dat
    """
    print(f"Reading file: {path}")
    f = open(path, "rb")
    if (precision == 4):
        dat = np.fromfile(f, dtype=np.float32, count=nx*ny*nz)
    elif (precision == 8):
        dat = np.fromfile(f, dtype=np.float64, count=nx*ny*nz)
    else:
        raise ValueError("Precision must be 4 or 8")
    dat = np.reshape(dat, (nx,ny,nz), order="F")
    f.close()

    return dat
# ---------------------------------------------
def out2netcdf(dout, timestep, del_raw=False, **params):
    """Read binary output files from LES code and combine into one netcdf
    file per timestep using xarray for future reading and easier analysis.
    Looks for the following: 
    u, v, w, theta, q, txz, tyz, q3, wq_sgs, dissip.
    -Input-
    dout: string, absolute path to directory with raw binary output
    timestep: integer timestep of files to process
    del_raw: boolean flag to automatically delete raw .out files\
        to save space, default=False
    params: dictionary with output from params.yaml file
    -Output-
    single netcdf file with convention 'dout/netcdf/all_<timestep>.nc'
    """
    # grab relevent parameters
    nx, ny, nz = params["nx"], params["ny"], params["nz"]
    Lx, Ly, Lz = params["Lx"], params["Ly"], params["Lz"]
    dz = Lz/nz
    u_scale = params["u_scale"]
    theta_scale = params["T_scale"]
    q_scale = params["q_scale"]
    # dimensions
    x, y = np.linspace(0., Lx, nx), np.linspace(0, Ly, ny)
    # u- and w-nodes are staggered
    # zw = 0:Lz:nz
    # zu = dz/2:Lz-dz/2:nz
    # interpolate w, txz, tyz, q3 to u grid
    zw = np.linspace(0., Lz, nz)
    zu = np.linspace(dz/2., Lz+dz/2., nz)
    # --------------------------------
    # Read binary and save new files
    # --------------------------------
    # load and apply scales
    f1 = f"{dout}u_{timestep:07d}.out"
    u_in = read_f90_bin(f1,nx,ny,nz,8) * u_scale
    f2 = f"{dout}v_{timestep:07d}.out"
    v_in = read_f90_bin(f2,nx,ny,nz,8) * u_scale
    f3 = f"{dout}w_{timestep:07d}.out"
    w_in = read_f90_bin(f3,nx,ny,nz,8) * u_scale
    f4 = f"{dout}theta_{timestep:07d}.out"
    theta_in = read_f90_bin(f4,nx,ny,nz,8) * theta_scale
    f5 = f"{dout}txz_{timestep:07d}.out"
    txz_in = read_f90_bin(f5,nx,ny,nz,8) * u_scale * u_scale
    f6 = f"{dout}tyz_{timestep:07d}.out"
    tyz_in = read_f90_bin(f6,nx,ny,nz,8) * u_scale * u_scale
    f7 = f"{dout}q3_{timestep:07d}.out"
    q3_in = read_f90_bin(f7,nx,ny,nz,8) * u_scale * theta_scale
    f8 = f"{dout}q_{timestep:07d}.out"
    q_in = read_f90_bin(f8,nx,ny,nz,8) * q_scale
    f9 = f"{dout}wq_sgs_{timestep:07d}.out"
    wq_sgs_in = read_f90_bin(f9,nx,ny,nz,8) * u_scale * q_scale    
    fd = f"{dout}dissip_{timestep:07d}.out"
    diss_in = read_f90_bin(fd,nx,ny,nz,8) * u_scale * u_scale * u_scale / Lz
    # list of all out files for cleanup later
    fout_all = [f1, f2, f3, f4, f5, f6, f7, f8, f9, fd]
    # interpolate w, txz, tyz, q3, wq_sgs, dissip to u grid
    # create DataArrays
    w_da = xr.DataArray(w_in, dims=("x", "y", "z"), 
                        coords=dict(x=x, y=y, z=zw))
    txz_da = xr.DataArray(txz_in, dims=("x", "y", "z"), 
                          coords=dict(x=x, y=y, z=zw))
    tyz_da = xr.DataArray(tyz_in, dims=("x", "y", "z"), 
                          coords=dict(x=x, y=y, z=zw))
    q3_da = xr.DataArray(q3_in, dims=("x", "y", "z"), 
                         coords=dict(x=x, y=y, z=zw))
    diss_da = xr.DataArray(diss_in, dims=("x", "y", "z"), 
                           coords=dict(x=x, y=y, z=zw))
    wq_sgs_da = xr.DataArray(wq_sgs_in, dims=("x", "y", "z"), 
                             coords=dict(x=x, y=y, z=zw))
    # perform interpolation
    w_interp = w_da.interp(z=zu, method="linear", 
                           kwargs={"fill_value": "extrapolate"})
    txz_interp = txz_da.interp(z=zu, method="linear", 
                               kwargs={"fill_value": "extrapolate"})
    tyz_interp = tyz_da.interp(z=zu, method="linear", 
                               kwargs={"fill_value": "extrapolate"})
    q3_interp = q3_da.interp(z=zu, method="linear", 
                             kwargs={"fill_value": "extrapolate"})
    diss_interp = diss_da.interp(z=zu, method="linear", 
                                 kwargs={"fill_value": "extrapolate"})
    wq_sgs_interp = wq_sgs_da.interp(z=zu, method="linear", 
                                     kwargs={"fill_value": "extrapolate"})
    # construct dictionary of data to save -- u-node variables only!
    data_save = {
                 "u": (["x","y","z"], u_in),
                 "v": (["x","y","z"], v_in),
                 "theta": (["x","y","z"], theta_in),
                 "q": (["x","y","z"], q_in)
                }

    # construct dataset from these variables
    ds = xr.Dataset(data_save, coords=dict(x=x, y=y, z=zu), attrs=params)
    # now assign interpolated arrays that were on w-nodes
    ds["w"] = w_interp
    ds["txz"] = txz_interp
    ds["tyz"] = tyz_interp
    ds["q3"] = q3_interp
    ds["wq_sgs"] = wq_sgs_interp
    ds["dissip"] = diss_interp
    # hardcode dictionary of units to use by default
    if units is None:
        units = {
            "u": "m/s",
            "v": "m/s",
            "w": "m/s",
            "theta": "K",
            "q": "g/kg",
            "txz": "m^2/s^2",
            "tyz": "m^2/s^2",
            "q3": "K m/s",
            "wq_sgs": "m/s g/kg",
            "dissip": "m^2/s^3",
            "x": "m",
            "y": "m",
            "z": "m"
        }

    # loop and assign attributes
    for var in list(data_save.keys())+["x", "y", "z"]:
        ds[var].attrs["units"] = units[var]
    # save to netcdf file and continue
    fsave = f"{dout}netcdf/all_{timestep:07d}.nc"
    print(f"Saving file: {fsave.split(os.sep)[-1]}")
    ds.to_netcdf(fsave)

    # delete files from this timestep if desired
    if del_raw:
        print("Cleaning up raw files...")
        for ff in fout_all:
            os.system(f"rm {ff}")

    return
# ---------------------------------------------
def process_raw_sim(dout, nhr, del_raw, overwrite=False, 
                    cstats=True, rotate=False):
    """Use information from dout/param.yaml file to dynamically process raw 
    output files with out2netcdf function for a desired time period of files.
    Optional additional processing: calc_stats(), nc_rot().
    -Input-
    dout: string, absolute path to output files
    nhr: float, number of physical hours to process. Will use information from\
        param file to dynamically select files.
    del_raw: boolean, flag to pass to out2netcdf for cleaning up raw files
    overwrite: boolean, flag to overwrite output file in case it already exists.
    cstats: boolean, call calc_stats on the range determined for out2netcdf.
    rotate: boolean, call nc_rot on the same files as out2netcdf.
    -Output-
    single netcdf file for each timestep in the range nhr.
    """
    # import yaml file
    with open(dout+"params.yaml") as fp:
        params = yaml.safe_load(fp)
    # add simulation label to params
    params["simlabel"] = params["path"].split(os.sep)[-1]
    # determine range of files from info in params
    tf = params["jt_final"]
    nf = int(nhr / params["dt"]) // params["nwrite"]
    t0 = tf - nf
    timesteps = np.arange(t0, tf+1, params["nwrite"], dtype=np.int32)
    print(f"Processing {nhr} hours = {nf} timesteps from t0={t0} to tf={tf}")
    # check if netcdf directory exists
    dnc = f"{dout}netcdf/"
    if not os.path.exists(dnc):
        os.system(f"mkdir {dnc}")

    # loop over timesteps and call out2netcdf
    # store filenames created
    f_all = []
    for tt in timesteps:
        # check if all_{tt}.nc exists already
        ff = f"{dnc}all_{tt:07d}.nc"
        f_all.append(ff)
        if not os.path.exists(ff):
            print(f"Processing timestep: {tt}")
            out2netcdf(dout, tt, del_raw=del_raw, **params)
        elif (os.path.exists(ff) & overwrite):
            print(f"{ff} already exists!")
            # need to check if the raw out files exist -- just look at u
            if os.path.exists(f"{dout}u_{tt:07d}.out"):
                print(f"overwrite=True and raw files exist, continuing...")
                out2netcdf(dout, tt, del_raw=del_raw, **params)
            else:
                print("overwrite=True but raw files deleted, skip timestep.")
        else:
            print(f"{ff} already exists! Skipping to next timestep.")

    # run calc_stats()
    if cstats:
        print(f"Begin calculating stats for final {nhr} hours...")
        calc_stats(f_use=f_all, **params)

    return
# ---------------------------------------------
def calc_stats(f_use=None, nhr=None, **params):
    """Read multiple output netcdf files created by out2netcdf() to calculate
    averages in x, y, t and save as new netcdf file. Directly input the names
    of files to read, or state how many hours to process and dynamically
    determine file names from information in params kwargs.
    -Input-
    f_use: list of filename strings to read directly if not None
    nhr: float, number of hours worth of files to read if not None
    params: dictionary with relevant simulation information for processing
    -Output-
    statistics file with format 'mean_stats_xyt_<t0>-<tf>h.nc'
    """
    # make sure params is not empty
    if len(params.keys()) < 1:
        print("No parameters provided. Returning without proceeding.")
        return
    # construct dnc
    dnc = f"{params['path']}output/netcdf/"

    # option 1: f_use is not None
    if f_use is not None:
        fall = f_use
    elif nhr is not None:
        # determine range of files from info in params
        tf = params["jt_final"]
        nf = int(nhr / params["dt"]) // params["nwrite"]
        t0 = tf - nf
        timesteps = np.arange(t0, tf+1, params["nwrite"], dtype=np.int32)
        # determine files to read from timesteps
        fall = [f"{dnc}all_{tt:07d}.nc" for tt in timesteps]
    # get length of file list
    nf = len(fall)
    # calculate array of times represented by each file
    times = np.array([i*params["dt"]*params["nwrite"] for i in range(nf)])
    # --------------------------------
    # Load files and clean up
    # --------------------------------
    print("Reading files...")
    dd = xr.open_mfdataset(fall, combine="nested", concat_dim="time")
    dd.coords["time"] = times
    dd.time.attrs["units"] = "s"
    # --------------------------------
    # Calculate statistics
    # --------------------------------
    print("Beginning calculations")
    # create empty dataset that will hold everything
    dd_stat = xr.Dataset(data_vars=None, coords=dict(z=dd.z), attrs=dd.attrs)
    # list of base variables
    base = ["u", "v", "w", "theta", "q", "dissip"]
    # use for looping over vars in case dissip not used
    base1 = ["u", "v", "w", "theta", "q", "thetav"]
    # calculate means
    for s in base:
        dd_stat[f"{s}_mean"] = dd[s].mean(dim=("time", "x", "y"))
    # calculate covars
    # u'w'
    dd_stat["uw_cov_res"] = xr.cov(dd.u, dd.w, dim=("time", "x", "y"))
    dd_stat["uw_cov_tot"] = dd_stat.uw_cov_res + dd.txz.mean(dim=("time","x","y"))
    # v'w'
    dd_stat["vw_cov_res"] = xr.cov(dd.v, dd.w, dim=("time", "x", "y"))
    dd_stat["vw_cov_tot"] = dd_stat.vw_cov_res + dd.tyz.mean(dim=("time","x","y"))
    # theta'w'
    dd_stat["tw_cov_res"] = xr.cov(dd.theta, dd.w, dim=("time", "x", "y"))
    dd_stat["tw_cov_tot"] = dd_stat.tw_cov_res + dd.q3.mean(dim=("time","x","y"))
    # q'w'
    dd_stat["qw_cov_res"] = xr.cov(dd.q, dd.w, dim=("time","x","y"))
    dd_stat["qw_cov_tot"] = dd_stat.qw_cov_res + dd.wq_sgs.mean(dim=("time","x","y"))
    # calculate thetav
    dd["thetav"] = dd.theta * (1. + 0.61*dd.q/1000.)
    dd_stat["thetav_mean"] = dd.thetav.mean(dim=("time","x","y"))
    # tvw_cov_tot from tw_cov_tot and qw_cov_tot
    dd_stat["tvw_cov_tot"] = dd_stat.tw_cov_tot +\
        0.61 * dd_stat.thetav_mean[0] * dd_stat.qw_cov_tot/1000.
    # calculate vars
    for s in base1:
        # detrend by subtracting planar averages at each timestep
        vp = dd[s] - dd[s].mean(dim=("x","y"))
        dd_stat[f"{s}_var"] = vp.var(dim=("time","x","y"))
    # rotate u_mean and v_mean so <v> = 0
    angle = np.arctan2(dd_stat.v_mean, dd_stat.u_mean)
    dd_stat["u_mean_rot"] = dd_stat.u_mean*np.cos(angle) + dd_stat.v_mean*np.sin(angle)
    dd_stat["v_mean_rot"] =-dd_stat.u_mean*np.sin(angle) + dd_stat.v_mean*np.cos(angle)
    # rotate instantaneous u and v for variances 
    # (not sure if necessary by commutative property but might as well)
    angle_inst = np.arctan2(dd.v.mean(dim=("x","y")), dd.u.mean(dim=("x","y")))
    u_rot = dd.u*np.cos(angle_inst) + dd.v*np.sin(angle_inst)
    v_rot =-dd.u*np.sin(angle_inst) + dd.v*np.cos(angle_inst)
    # recalculate u_var_rot, v_var_rot
    uvar_rot = u_rot - u_rot.mean(dim=("x","y"))
    dd_stat["u_var_rot"] = uvar_rot.var(dim=("time","x","y"))
    vvar_rot = v_rot - v_rot.mean(dim=("x","y"))
    dd_stat["v_var_rot"] = vvar_rot.var(dim=("time","x","y"))
    # calculate <theta'q'>
    td = dd.theta - dd.theta.mean(dim=("x","y"))
    qd = dd.q - dd.q.mean(dim=("x","y"))
    dd_stat["tq_cov_res"] = xr.cov(td, qd, dim=("time","x","y"))
    # --------------------------------
    # Add attributes
    # --------------------------------
    # grid spacing
    dd_stat.attrs["delta"] = (dd.dx * dd.dy * dd.dz) ** (1./3.)
    # calculate number of hours in average based on timesteps array
    dd_stat.attrs["tavg"] = times[-1] / 3600.
    # --------------------------------
    # Save output file
    # --------------------------------
    # determine tavg string to use in save file, e.g., "5-6h"
    # TODO: determine handling for varying dt e.g. after interpolation
    t_final_s = (params["jt_final"] - params["jt_total_init"]) * params["dt"]
    t_final_h = t_final_s / 3600.
    t_start_h = t_final_h - (nf * params["nwrite"] * params["dt"] / 3600.)
    # check if t_start_h and t_final_h are round numbers
    if (t_start_h % 1.0 == 0.0):
        ts_s = str(int(t_start_h)) # use integer
    else:
        ts_s = f"{t_start_h:.1f}"  # convert to 1 decimal place float
    if (t_final_h % 1.0 == 0.0):
        tf_s = str(int(t_final_h)) # use integer
    else:
        tf_s = f"{t_final_h:.1f}"  # convert to 1 decimal place float
    fsave = f"{dnc}mean_stats_xyt_{ts_s}-{tf_s}h.nc"
    print(f"Saving file: {fsave}")
    with ProgressBar():
        dd_stat.to_netcdf(fsave, mode="w")
    print("Finished!")
    return
# ---------------------------------------------
def calc_stats_long(dnc, t0, t1, dt, delta_t, use_dissip, use_q, tavg, rotate):
    """Read multiple output netcdf files created by sim2netcdf() to calculate
    averages in x, y, t and save as new netcdf file. Identical in scope as
    calc_stats(), but sacrifices xarray syntax simplicity for the sake of
    memory management.

    :param str dnc: absolute path to directory for loading netCDF files
    :param int t0: first timestep to process
    :param int t1: last timestep to process
    :param int dt: number of timesteps between files to load
    :param float delta_t: dimensional timestep in simulation (seconds)
    :param bool use_dissip: flag for loading dissipation rate files (SBL only)
    :param bool use_q: flag for loading specific humidity files (les_brg only)
    :param str tavg: label denoting length of temporal averaging (e.g. 1h)
    :param bool rotate: flag for using rotated fields
    """
    # directories and configuration
    timesteps = np.arange(t0, t1+1, dt, dtype=np.int32)
     # determine files to read from timesteps
    if rotate:
        fall = [f"{dnc}all_{tt:07d}_rot.nc" for tt in timesteps]
    else:
        fall = [f"{dnc}all_{tt:07d}.nc" for tt in timesteps]
    nf = len(fall)
    # calculate array of times represented by each file
    times = np.array([i*delta_t*dt for i in range(nf)])
    # --------------------------------
    # Load one file for attrs
    # --------------------------------
    d1 = xr.load_dataset(fall[0])
    # --------------------------------
    # Calculate statistics
    # --------------------------------
    print("Beginning calculations")
    # create empty dataset that will hold everything
    # will average in time after big loop
    dd_long = xr.Dataset()
    # list of base variables
    base = ["u", "v", "w", "theta"]
    base1 = ["u", "v", "w", "theta"] # use for looping over vars in case dissip not used
    # check for dissip
    if use_dissip:
        base.append("dissip")
    if use_q:
        base.append("q")
        base1.append("q")
    # initialize empty arrays in dd_long for values in base, base1
    for s in base:
        dd_long[f"{s}_mean"] = xr.DataArray(data=np.zeros((d1.nz), dtype=np.float64),
                                            coords=dict(z=d1.z))
    dd_long["thetav_mean"] = xr.DataArray(data=np.zeros((d1.nz), dtype=np.float64),
                                          coords=dict(z=d1.z))
    dd_long["u_mean_rot"] = xr.DataArray(data=np.zeros((d1.nz), dtype=np.float64),
                                         coords=dict(z=d1.z))
    dd_long["v_mean_rot"] = xr.DataArray(data=np.zeros((d1.nz), dtype=np.float64),
                                         coords=dict(z=d1.z))
    # covars
    # u'w'
    dd_long["uw_cov_res"] = xr.DataArray(data=np.zeros((d1.nz), dtype=np.float64),
                                         coords=dict(z=d1.z))
    dd_long["uw_cov_sgs"] = xr.DataArray(data=np.zeros((d1.nz), dtype=np.float64),
                                         coords=dict(z=d1.z))
    # v'w'
    dd_long["vw_cov_res"] = xr.DataArray(data=np.zeros((d1.nz), dtype=np.float64),
                                         coords=dict(z=d1.z))
    dd_long["vw_cov_sgs"] = xr.DataArray(data=np.zeros((d1.nz), dtype=np.float64),
                                         coords=dict(z=d1.z))
    # theta'w'
    dd_long["tw_cov_res"] = xr.DataArray(data=np.zeros((d1.nz), dtype=np.float64),
                                         coords=dict(z=d1.z))
    dd_long["tw_cov_sgs"] = xr.DataArray(data=np.zeros((d1.nz), dtype=np.float64),
                                         coords=dict(z=d1.z))
    dd_long["tq_cov_res"] = xr.DataArray(data=np.zeros((d1.nz), dtype=np.float64),
                                         coords=dict(z=d1.z))
    if use_q:
        # q'w'
        dd_long["qw_cov_res"] = xr.DataArray(data=np.zeros((d1.nz), dtype=np.float64),
                                             coords=dict(z=d1.z))
        dd_long["qw_cov_sgs"] = xr.DataArray(data=np.zeros((d1.nz), dtype=np.float64),
                                             coords=dict(z=d1.z))

        dd_long["thetav_mean"] = xr.DataArray(data=np.zeros((d1.nz), dtype=np.float64),
                                              coords=dict(z=d1.z))
    # vars
    for s in base1:
        dd_long[f"{s}_var"] = xr.DataArray(data=np.zeros((d1.nz), dtype=np.float64),
                                           coords=dict(z=d1.z))
    dd_long["u_var_rot"] = xr.DataArray(data=np.zeros((d1.nz), dtype=np.float64),
                                        coords=dict(z=d1.z))
    dd_long["v_var_rot"] = xr.DataArray(data=np.zeros((d1.nz), dtype=np.float64),
                                        coords=dict(z=d1.z))
    #
    # big loop over time
    #
    for jt, ff in enumerate(fall):
        # only open one file at a time
        print(f"Calculating stats for file {jt+1}/{nf}")
        d = xr.load_dataset(ff)
        # calculate means
        for s in base:
            dd_long[f"{s}_mean"] += d[s].mean(dim=("x","y"))
        # calculate covars
        # u'w'
        dd_long["uw_cov_res"] += xr.cov(d.u, d.w, dim=("x","y"))
        dd_long["uw_cov_sgs"] += d.txz.mean(dim=("x","y"))
        # v'w'
        dd_long["vw_cov_res"] += xr.cov(d.v, d.w, dim=("x","y"))
        dd_long["vw_cov_sgs"] += d.tyz.mean(dim=("x","y"))
        # theta'w'
        dd_long["tw_cov_res"] += xr.cov(d.theta, d.w, dim=("x","y"))
        dd_long["tw_cov_sgs"] += d.q3.mean(dim=("x","y"))
        if use_q:
            # q'w'
            dd_long["qw_cov_res"] += xr.cov(d.q, d.w, dim=("x","y"))
            dd_long["qw_cov_sgs"] += d.wq_sgs.mean(dim=("x","y"))
            # calculate thetav
            tv = d.theta * (1. + 0.61*d.q/1000.)
            dd_long["thetav_mean"] += tv.mean(dim=("x","y"))
            # also take the chance to calculate <theta'q'>
            dd_long["tq_cov_res"] += xr.cov(d.theta, d.q, dim=("x","y"))
        # calculate vars
        for s in base1:
            vp = d[s] - d[s].mean(dim=("x","y"))
            dd_long[f"{s}_var"] += vp.var(dim=("x","y"))
        # rotate u_mean and v_mean so <v> = 0
        angle = np.arctan2(d.v.mean(dim=("x","y")), d.u.mean(dim=("x","y")))
        u_rot = d.u*np.cos(angle) + d.v*np.sin(angle)
        v_rot =-d.u*np.sin(angle) + d.v*np.cos(angle)
        # calculate and store mean, var
        dd_long["u_mean_rot"] += u_rot.mean(dim=("x","y"))
        dd_long["v_mean_rot"] += v_rot.mean(dim=("x","y"))
        upr = u_rot - u_rot.mean(dim=("x","y"))
        dd_long["u_var_rot"] += upr.var(dim=("x","y"))
        vpr = v_rot - v_rot.mean(dim=("x","y"))
        dd_long["v_var_rot"] += vpr.var(dim=("x","y"))
        
    # --------------------------------
    # outside of big time loop
    # --------------------------------
    # average over number of timesteps (nf)
    print("Averaging in time...")
    # define empty Dataset for saving
    dd_save = xr.Dataset()
    # loop over all variables in dd_long and divide by nf
    for s in list(dd_long.keys()):
        dd_save[s] = dd_long[s] / float(nf)
    print("Calculating and storing additional parameters...")
    # combine resolved and sgs to get tot profiles
    dd_save["uw_cov_tot"] = dd_save.uw_cov_res + dd_save.uw_cov_sgs
    dd_save["vw_cov_tot"] = dd_save.vw_cov_res + dd_save.vw_cov_sgs
    dd_save["tw_cov_tot"] = dd_save.tw_cov_res + dd_save.tw_cov_sgs
    if use_q:
        dd_save["qw_cov_tot"] = dd_save.qw_cov_res + dd_save.qw_cov_sgs
        # tvw_cov_tot from tw_cov_tot and qw_cov_tot
        dd_save["tvw_cov_tot"] = dd_save.tw_cov_tot +\
                0.61 * dd_save.thetav_mean[0] * dd_save.qw_cov_tot/1000.
    # --------------------------------
    # Add attributes
    # --------------------------------
    # copy from dd
    dd_save.attrs = d1.attrs
    dd_save.attrs["delta"] = (d1.dx * d1.dy * d1.dz) ** (1./3.)
    # calculate number of hours in average based on timesteps array
    dd_save.attrs["tavg"] = delta_t * (t1 - t0) / 3600.
    # --------------------------------
    # Save output file
    # --------------------------------
    if rotate:
        fsave = f"{dnc}mean_stats_xyt_{tavg}_rot.nc"
    else:
        fsave = f"{dnc}mean_stats_xyt_{tavg}.nc"
    print(f"Saving file: {fsave}")
    with ProgressBar():
        dd_save.to_netcdf(fsave, mode="w")
    print("Finished!")
    return
# ---------------------------------------------
def load_stats(fstats, SBL=False, display=False):
    """Reading function for average statistics files created from calc_stats()
    Load netcdf files using xarray and calculate numerous relevant parameters
    :param str fstats: absolute path to netcdf file for reading
    :param bool SBL: denote whether sim data is SBL or not to calculate\
        appropriate ABL depth etc., default=False
    :param bool display: print statistics from files, default=False
    :return dd: xarray dataset
    """
    print(f"Reading file: {fstats}")
    dd = xr.load_dataset(fstats)
    # calculate ustar and h
    dd["ustar"] = ((dd.uw_cov_tot**2.) + (dd.vw_cov_tot**2.)) ** 0.25
    dd["ustar2"] = dd.ustar ** 2.
    if SBL:
        dd["h"] = dd.z.where(dd.ustar2 <= 0.05*dd.ustar2[0], drop=True)[0] / 0.95
    else:
        # CBL
        dd["h"] = dd.z.isel(z=dd.tw_cov_tot.argmin())
    # save number of points within abl (z <= h)
    dd.attrs["nzabl"] = dd.z.where(dd.z <= dd.h, drop=True).size
    # grab ustar0 and calc tstar0 for normalizing in plotting
    dd["ustar0"] = dd.ustar.isel(z=0)
    dd["tstar0"] = -dd.tw_cov_tot.isel(z=0)/dd.ustar0
    # local thetastar
    dd["tstar"] = -dd.tw_cov_tot / dd.ustar
    # qstar
    if "qw_cov_tot" in list(dd.keys()):
        dd["qstar0"] = -dd.qw_cov_tot.isel(z=0) / dd.ustar0
    # calculate TKE
    dd["e"] = 0.5 * (dd.u_var + dd.v_var + dd.w_var)
    # calculate Obukhov length L
    dd["L"] = -(dd.ustar0**3) * dd.theta_mean.isel(z=0) / (0.4 * 9.81 * dd.tw_cov_tot.isel(z=0))
    # calculate uh and wdir
    dd["uh"] = np.sqrt(dd.u_mean**2. + dd.v_mean**2.)
    dd["wdir"] = np.arctan2(-dd.u_mean, -dd.v_mean) * 180./np.pi
    jz_neg = np.where(dd.wdir < 0.)[0]
    dd["wdir"][jz_neg] += 360.
    # calculate mean lapse rate between lowest grid point and z=h
    delta_T = dd.theta_mean.sel(z=dd.h, method="nearest") - dd.theta_mean[0]
    delta_z = dd.z.sel(z=dd.h, method="nearest") - dd.z[0]
    dd["dT_dz"] = delta_T / delta_z
    # calculate eddy turnover time TL
    if SBL:
        # use ustar
        dd["TL"] = dd.h / dd.ustar0
    else:
        # calculate wstar and use for TL calc
        # use humidity if exists
        if "tvw_cov_tot" in list(dd.keys()):
            dd["wstar"] = ((9.81/dd.thetav_mean.sel(z=dd.h/2, method="nearest"))\
                            * dd.tvw_cov_tot[0] * dd.h) ** (1./3.)
            # use this wstart to calc Qstar, otherwise skip
            dd["Qstar0"] = dd.qw_cov_tot.isel(z=0) / dd.wstar
        else:
            dd["wstar"] = ((9.81/dd.theta_mean.sel(z=dd.h/2, method="nearest"))\
                            * dd.tw_cov_tot[0] * dd.h) ** (1./3.)
        # now calc TL using wstar in CBL
        dd["TL"] = dd.h / dd.wstar
        # use wstar to define Tstar in CBL
        dd["Tstar0"] = dd.tw_cov_tot.isel(z=0) / dd.wstar
        
    # determine how many TL exist over range of files averaged
    # convert tavg string to number by cutting off the single letter at the end
    # dd["nTL"] = dd.tavg * 3600. / dd.TL

    # calculate MOST dimensionless functions phim, phih
    kz = 0.4 * dd.z # kappa * z
    dd["phim"] = (kz/dd.ustar) * np.sqrt(dd.u_mean.differentiate("z", 2)**2.+\
                                         dd.v_mean.differentiate("z", 2)**2.)
    dd["phih"] = (kz/dd.tstar) * dd.theta_mean.differentiate("z", 2)
    # MOST stability parameter z/L
    dd["zL"] = dd.z / dd.L
    if SBL:
        # calculate TKE-based sbl depth
        dd["he"] = dd.z.where(dd.e <= 0.05*dd.e[0], drop=True)[0]
        # calculate h/L as global stability parameter
        dd["hL"] = dd.h / dd.L
        # create string for labels from hL
        dd.attrs["label3"] = f"$h/L = {{{dd.hL.values:3.2f}}}$"
        # calculate Richardson numbers
        # sqrt((du_dz**2) + (dv_dz**2))
        dd["du_dz"] = np.sqrt(dd.u_mean.differentiate("z", 2)**2. +\
                              dd.v_mean.differentiate("z", 2)**2.)
        # Rig = N^2 / S^2
        dd["N2"] = dd.theta_mean.differentiate("z", 2) * 9.81 / dd.theta_mean.isel(z=0)
        # flag negative values of N^2
        dd.N2[dd.N2 < 0.] = np.nan
        dd["Rig"] = dd.N2 / dd.du_dz / dd.du_dz
        # Rif = beta * w'theta' / (u'w' du/dz + v'w' dv/dz)
        dd["Rif"] = (9.81/dd.theta_mean.isel(z=0)) * dd.tw_cov_tot /\
                    (dd.uw_cov_tot*dd.u_mean.differentiate("z", 2) +\
                     dd.vw_cov_tot*dd.v_mean.differentiate("z", 2))
        # # bulk Richardson number Rib based on values at top of sbl and sfc
        # dz = dd.z[dd.nzsbl] - dd.z[0]
        # dTdz = (dd.theta_mean[dd.nzsbl] - dd.theta_mean[0]) / dz
        # dUdz = (dd.u_mean[dd.nzsbl] - dd.u_mean[0]) / dz
        # dVdz = (dd.v_mean[dd.nzsbl] - dd.v_mean[0]) / dz
        # dd.attrs["Rib"] = (dTdz * 9.81 / dd.theta_mean[0]) / (dUdz**2. + dVdz**2.)
        # calc Ozmidov scale real quick
        dd["Lo"] = np.sqrt(-dd.dissip_mean / (dd.N2 ** (3./2.)))
        # calculate Kolmogorov microscale: eta = (nu**3 / dissip) ** 0.25
        dd["eta"] = ((1.14e-5)**3. / (-dd.dissip_mean)) ** 0.25
        # calculate MOST dimensionless dissipation rate: kappa*z*epsilon/ustar^3
        dd["phie"] = -1*dd.dissip_mean*kz / (dd.ustar**3.)
        # calculate gradient scales from Sorbjan 2017, Greene et al. 2022
        l0 = 19.22 # m
        l1 = 1./(dd.Rig**(3./2.)).where(dd.z <= dd.h, drop=True)
        kz = 0.4 * dd.z.where(dd.z <= dd.h, drop=True)
        dd["Ls"] = kz / (1 + (kz/l0) + (kz/l1))
        dd["Us"] = dd.Ls * np.sqrt(dd.N2)
        dd["Ts"] = dd.Ls * dd.theta_mean.differentiate("z", 2)
        # calculate local Obukhov length Lambda
        dd["LL"] = -(dd.ustar**3.) * dd.theta_mean / (0.4 * 9.81 * dd.tw_cov_tot)
        # calculate level of LLJ: zj
        dd["zj"] = dd.z.isel(z=dd.uh.argmax())
    # print table statistics
    if display:
        print(f"---{dd.stability}---")
        print(f"u*: {dd.ustar0.values:4.3f} m/s")
        print(f"theta*: {dd.tstar0.values:5.4f} K")
        print(f"Q*: {1000*dd.tw_cov_tot.isel(z=0).values:4.3f} K m/s")
        print(f"h: {dd.h.values:4.3f} m")
        print(f"L: {dd.L.values:4.3f} m")
        print(f"h/L: {(dd.h/dd.L).values:4.3f}")
        # print(f"Rib: {dd.Rib.values:4.3f}")
        print(f"zj/h: {(dd.z.isel(z=dd.uh.argmax())/dd.h).values:4.3f}")
        print(f"dT/dz: {1000*dd.dT_dz.values:4.1f} K/km")
        print(f"TL: {dd.TL.values:4.1f} s")
        print(f"nTL: {dd.nTL.values:4.1f}")

    return dd
# ---------------------------------------------
def load_full(dnc, t0, t1, dt, delta_t, SBL=False, stats=None, rotate=False):
    """Reading function for multiple instantaneous volumetric netcdf files
    Load netcdf files using xarray
    :param str dnc: abs path directory for location of netcdf files
    :param int t0: first timestep to process
    :param int t1: last timestep to process
    :param int dt: number of timesteps between files to load
    :param float delta_t: dimensional timestep in simulation (seconds)
    :param bool SBL: calculate SBL-specific parameters. default=False
    :param str stats: name of statistics file. default=None
    :param bool rotate: use rotated netcdf files. default=False

    :return dd: xarray dataset of 4d volumes
    :return s: xarray dataset of statistics file
    """
    # load individual files into one dataset
    timesteps = np.arange(t0, t1+1, dt, dtype=np.int32)
    # determine files to read from timesteps
    if rotate:
        fall = [f"{dnc}all_{tt:07d}_rot.nc" for tt in timesteps]
    else:
        fall = [f"{dnc}all_{tt:07d}.nc" for tt in timesteps]
    nf = len(fall)
    # calculate array of times represented by each file
    times = np.array([i*delta_t*dt for i in range(nf)])
    # read files
    print("Loading files...")
    dd = xr.open_mfdataset(fall, combine="nested", concat_dim="time")
    dd.coords["time"] = times
    dd.time.attrs["units"] = "s"
    # calculate theta_v if q in varlist
    if "q" in list(dd.keys()):
        dd["thetav"] = dd.theta * (1. + 0.61*dd.q/1000.)
    # stats file
    if stats is not None:
        # load stats file
        s = load_stats(dnc+stats, SBL=SBL)
        # calculate rotated u, v based on xy mean at each timestep
        uxy = dd.u.mean(dim=("x","y"))
        vxy = dd.v.mean(dim=("x","y"))
        angle = np.arctan2(vxy, uxy)
        dd["u_rot"] = dd.u*np.cos(angle) + dd.v*np.sin(angle)
        dd["v_rot"] =-dd.u*np.sin(angle) + dd.v*np.cos(angle)
        # return both dd and s
        return dd, s
    # just return dd if no stats
    return dd
# ---------------------------------------------
def timeseries2netcdf(dout, dnc, scales, use_q, delta_t, nz, Lz, 
                      nz_mpi, nhr, tf, simlabel, del_raw=False):
    """Load raw timeseries data at each level and combine into single
    netcdf file with dimensions (time, z)
    :param str dout: absolute path to directory with LES output binary files
    :param str dnc: absolute path to directory for saving output netCDF files
    :param tuple<Quantity> scales: dimensional scales from LES code\
        (uscale, Tscale, qscale)
    :param bool use_q: flag to use specific humidity q
    :param float delta_t: dimensional timestep in simulation (seconds)
    :param int nz: resolution of simulation in vertical
    :param float Lz: height of domain in m
    :param int nz_mpi: number of vertical levels per MPI process
    :param float nhr: number of hours to load, counting backwards from end
    :param int tf: final timestep in file
    :param str simlabel: unique identifier for batch of files
    :param bool del_raw: automatically delete raw .out files from LES code\
        to save space, default=False
    """
    # grab relevent parameters
    u_scale = scales[0]
    theta_scale = scales[1]
    if use_q:
        q_scale = scales[2]
    dz = Lz/nz
    # define z array
    # u- and w-nodes are staggered
    # zw = 0:Lz:nz
    # zu = dz/2:Lz-dz/2:nz-1
    # interpolate w, txz, tyz, q3 to u grid
    zw = np.linspace(0., Lz, nz)
    zu = np.linspace(dz/2., Lz+dz/2., nz)
    # determine number of hours to process from tavg
    nt = int(nhr*3600./delta_t)
    istart = tf - nt
    # define array of time in seconds
    time = np.linspace(0., nhr*3600.-delta_t, nt, dtype=np.float64)
    print(f"Loading {nt} timesteps = {nhr} hr for simulation {simlabel}")
    # define DataArrays for u, v, w, theta, txz, tyz, q3
    # shape(nt,nz)
    # u, v, theta, q, thetav
    u_ts, v_ts, theta_ts, q_ts, thetav_ts =\
    (xr.DataArray(np.zeros((nt, nz), dtype=np.float64),
                  dims=("t", "z"), coords=dict(t=time, z=zu)) for _ in range(5))
    # w, txz, tyz, q3, wq_sgs
    w_ts, txz_ts, tyz_ts, q3_ts, wq_sgs_ts =\
    (xr.DataArray(np.zeros((nt, nz), dtype=np.float64),
                  dims=("t", "z"), coords=dict(t=time, z=zw)) for _ in range(5))
    fout_all = []
    # compute number of MPI levels
    nmpi = nz // nz_mpi
    # now loop through each file (one for each mpi process)
    # there are nz_mpi columns in each file!
    # initialize jz counter index for first nz_mpi columns
    jz = np.arange(nz_mpi, dtype=np.int32)
    # initialze range counter for which columns to use
    cols = range(1, nz_mpi+1)
    for jmpi in range(nmpi):
        print(f"Loading timeseries data, jz={jmpi}")
        fu = f"{dout}u_timeseries_c{jmpi:03d}.out"
        u_ts[:,jz] = np.loadtxt(fu, skiprows=istart, usecols=cols)
        fv = f"{dout}v_timeseries_c{jmpi:03d}.out"
        v_ts[:,jz] = np.loadtxt(fv, skiprows=istart, usecols=cols)
        fw = f"{dout}w_timeseries_c{jmpi:03d}.out"
        w_ts[:,jz] = np.loadtxt(fw, skiprows=istart, usecols=cols)
        ftheta = f"{dout}t_timeseries_c{jmpi:03d}.out"
        theta_ts[:,jz] = np.loadtxt(ftheta, skiprows=istart, usecols=cols)
        ftxz = f"{dout}txz_timeseries_c{jmpi:03d}.out"
        txz_ts[:,jz] = np.loadtxt(ftxz, skiprows=istart, usecols=cols)
        ftyz = f"{dout}tyz_timeseries_c{jmpi:03d}.out"
        tyz_ts[:,jz] = np.loadtxt(ftyz, skiprows=istart, usecols=cols)
        fq3 = f"{dout}q3_timeseries_c{jmpi:03d}.out"
        q3_ts[:,jz] = np.loadtxt(fq3, skiprows=istart, usecols=cols)
        fout_all += [fu, fv, fw, ftheta, ftxz, ftyz, fq3]
        # load q
        if use_q:
            fq = f"{dout}q_timeseries_c{jmpi:03d}.out"
            q_ts[:,jz] = np.loadtxt(fq, skiprows=istart, usecols=cols)
            ftv = f"{dout}tv_timeseries_c{jmpi:03d}.out"
            thetav_ts[:,jz] = np.loadtxt(ftv, skiprows=istart, usecols=cols)
            fqw = f"{dout}wq_sgs_timeseries_c{jmpi:03d}.out"
            wq_sgs_ts[:,jz] = np.loadtxt(fqw, skiprows=istart, usecols=cols)
            fout_all += [fq, ftv, fqw]
        # increment jz
        jz += nz_mpi
    # apply scales
    u_ts *= u_scale
    v_ts *= u_scale
    w_ts *= u_scale
    theta_ts *= theta_scale
    txz_ts *= (u_scale * u_scale)
    tyz_ts *= (u_scale * u_scale)
    q3_ts *= (u_scale * theta_scale)
    if use_q:
        q_ts *= q_scale
        thetav_ts *= theta_scale
        wq_sgs_ts *= (u_scale * q_scale)
    # define dictionary of attributes
    attrs = {"label": simlabel, "dt": delta_t, "nt": nt, "nz": nz, "total_time": nhr}
    # combine DataArrays into Dataset and save as netcdf
    # initialize empty Dataset
    ts_all = xr.Dataset(data_vars=None, coords=dict(t=time, z=zu), attrs=attrs)
    # now store
    ts_all["u"] = u_ts
    ts_all["v"] = v_ts
    ts_all["w"] = w_ts.interp(z=zu, method="linear", 
                              kwargs={"fill_value": "extrapolate"})
    ts_all["theta"] = theta_ts
    ts_all["txz"] = txz_ts.interp(z=zu, method="linear", 
                                  kwargs={"fill_value": "extrapolate"})
    ts_all["tyz"] = tyz_ts.interp(z=zu, method="linear", 
                                  kwargs={"fill_value": "extrapolate"})
    ts_all["q3"] = q3_ts.interp(z=zu, method="linear", 
                                kwargs={"fill_value": "extrapolate"})
    if use_q:
        ts_all["q"] = q_ts
        ts_all["thetav"] = thetav_ts
        ts_all["qw_sgs"] = wq_sgs_ts.interp(z=zu, method="linear",
                                            kwargs={"fill_value": "extrapolate"})
    # save to netcdf
    fsave_ts = f"{dnc}timeseries_all_{nhr}h.nc"
    with ProgressBar():
        ts_all.to_netcdf(fsave_ts, mode="w")
    
    # optionally delete files
    if del_raw:
        print("Cleaning up raw files...")
        for ff in fout_all:
            os.system(f"rm {ff}")
        
    print(f"Finished saving timeseries for simulation {simlabel}")

    return
# ---------------------------------------------
def load_timeseries(dnc, detrend=True, tavg="1.0h"):
    """Reading function for timeseries files created from timseries2netcdf()
    Load netcdf files using xarray and calculate numerous relevant parameters
    :param str dnc: path to netcdf directory for simulation
    :param bool detrend: detrend timeseries for calculating variances, default=True
    :param str tavg: select which timeseries file to use in hours, default="1h"
    :return d: xarray dataset
    """
    # load timeseries data
    fts = f"timeseries_all_{tavg}.nc"
    d = xr.open_dataset(dnc+fts)
    # calculate means
    varlist = ["u", "v", "w", "theta"]
    # check for humidity
    if "q" in list(d.keys()):
        varlist.append("q")
        varlist.append("thetav")
        use_q = True
    else:
        use_q = False
    for v in varlist:
        d[f"{v}_mean"] = d[v].mean("t") # average in time
    # rotate coords so <v> = 0
    angle = np.arctan2(d.v_mean, d.u_mean)
    d["u_mean_rot"] = d.u_mean*np.cos(angle) + d.v_mean*np.sin(angle)
    d["v_mean_rot"] =-d.u_mean*np.sin(angle) + d.v_mean*np.cos(angle)
    # rotate instantaneous u and v
    d["u_rot"] = d.u*np.cos(angle) + d.v*np.sin(angle)
    d["v_rot"] =-d.u*np.sin(angle) + d.v*np.cos(angle)
    # calculate "inst" vars
    if detrend:
        ud = xrft.detrend(d.u, dim="t", detrend_type="linear")
        udr = xrft.detrend(d.u_rot, dim="t", detrend_type="linear")
        vd = xrft.detrend(d.v, dim="t", detrend_type="linear")
        vdr = xrft.detrend(d.v_rot, dim="t", detrend_type="linear")
        wd = xrft.detrend(d.w, dim="t", detrend_type="linear")
        td = xrft.detrend(d.theta, dim="t", detrend_type="linear")
        # store these detrended variables
        d["ud"] = ud
        d["udr"] = udr
        d["vd"] = vd
        d["vdr"] = vdr
        d["wd"] = wd
        d["td"] = td
        # now calculate vars
        d["uu"] = ud * ud
        d["uur"] = udr * udr
        d["vv"] = vd * vd
        d["vvr"] = vdr * vdr
        d["ww"] = wd * wd
        d["tt"] = td * td
        # calculate "inst" covars
        d["uw"] = (ud * wd) + d.txz
        d["vw"] = (vd * wd) + d.tyz
        d["tw"] = (td * wd) + d.q3
        # do same with q variables
        if use_q:
            qd = xrft.detrend(d.q, dim="t", detrend_type="linear")
            d["qd"] = qd
            d["qq"] = qd * qd
            d["qw"] = (qd * qd) + d.qw_sgs
            d["tvw"] = d.tw + 0.61*d.td*d.qw/1000.
            d["tq"] = td * qd
    else:
        d["uu"] = (d.u - d.u_mean) * (d.u - d.u_mean)
        d["uur"] = (d.u_rot - d.u_mean_rot) * (d.u_rot - d.u_mean_rot)
        d["vv"] = (d.v - d.v_mean) * (d.v - d.v_mean)
        d["vvr"] = (d.v_rot - d.v_mean_rot) * (d.v_rot - d.v_mean_rot)
        d["ww"] = (d.w - d.w_mean) * (d.w - d.w_mean)
        d["tt"] = (d.theta - d.theta_mean) * (d.theta - d.theta_mean)
        # calculate "inst" covars
        d["uw"] = (d.u - d.u_mean) * (d.w - d.w_mean) + d.txz
        d["vw"] = (d.v - d.v_mean) * (d.w - d.w_mean) + d.tyz
        d["tw"] = (d.theta - d.theta_mean) * (d.w - d.w_mean) + d.q3
        # do same for q
        if use_q:
            d["qq"] = (d.q - d.q_mean) * (d.q - d.q_mean)
            d["qw"] = (d.q - d.q_mean) * (d.w - d.w_mean) + d.qw_sgs
            d["tvw"] = d.tw + 0.61*(d.theta - d.theta_mean)*d.qw/1000.
            d["tq"] = (d.q - d.q_mean) * (d.theta - d.theta_mean)
    
    return d
# ---------------------------------------------
@njit
def interp_uas(dat, z_LES, z_UAS):
    """Interpolate LES virtual tower timeseries data in the vertical to match
    ascent rate of emulated UAS to create timeseries of ascent data
    :param float dat: 2d array of the field to interpolate, shape(nt, nz)
    :param float z_LES: 1d array of LES grid point heights
    :param float z_UAS: 1d array of new UAS z from ascent & sampling rates
    Outputs 2d array of interpolated field, shape(nt, len(z_UAS))
    """
    nt = np.shape(dat)[0]
    nz = len(z_UAS)
    dat_interp = np.zeros((nt, nz), dtype=np.float64)
    for i in range(nt):
        dat_interp[i,:] = np.interp(z_UAS, z_LES, dat[i,:])
    return dat_interp
# ---------------------------------------------
def UASprofile(ts, zmax=2000., err=None, ascent_rate=3.0, time_average=3.0, time_start=0.0):
    """Emulate a vertical profile from a rotary-wing UAS sampling through a
    simulated ABL with chosen constant ascent rate and time averaging
    :param xr.Dataset ts: timeseries data from virtual tower created by
        timeseries2netcdf()
    :param float zmax: maximum height within domain to consider in m, 
        default=2000.
    :param xr.Dataset err: profile of errors to accompany emulated
        measurements, default=None
    :param float ascent_rate: ascent rate of UAS in m/s, default=3.0
    :param float time_average: averaging time bins in s, default=3.0
    :param float time_start: when to initialize ascending profile
        in s, default=0.0
    Returns new xarray Dataset with emulated profile
    """
    # First, calculate array of theoretical altitudes based on the base time
    # vector and ascent_rate while keeping account for time_start
    zuas = ascent_rate * ts.t.values
    # find the index in ts.time that corresponds to time_start
    istart = int(time_start / ts.dt)
    # set zuas[:istart] = 0 and then subtract everything after that
    zuas[:istart] = 0
    zuas[istart:] -= zuas[istart]
    # now only grab indices where 1 m <= zuas <= zmax
    iuse = np.where((zuas >= 1.) & (zuas <= zmax))[0]
    zuas = zuas[iuse]
    # calculate dz_uas from ascent_rate and time_average
    dz_uas = ascent_rate * time_average
    # loop over keys and interpolate
    interp_keys = ["u", "v", "theta"]
    d_interp = {} # define empty dictionary for looping
    for key in interp_keys:
        print(f"Interpolating {key}...")
        d_interp[key] = interp_uas(ts[key].isel(t=iuse).values,
                                   ts.z.values, zuas)

    # grab data from interpolated arrays to create simulated raw UAS profiles
    # define xarray dataset to eventually store all
    uas_raw = xr.Dataset(data_vars=None, coords=dict(z=zuas))
    # begin looping through keys
    for key in interp_keys:
        # define empty list
        duas = []
        # loop over altitudes/times
        for i in range(len(iuse)):
            duas.append(d_interp[key][i,i])
        # assign to uas_raw
        uas_raw[key] = xr.DataArray(data=np.array(duas), coords=dict(z=zuas))
    
    # emulate post-processing and average over altitude bins
    # can do this super easily with xarray groupby_bins
    # want bins to be at the midpoint between dz_uas grid
    zbin = np.arange(dz_uas/2, zmax-dz_uas/2, dz_uas)
    # group by altitude bins and calculate mean in one line
    uas_mean = uas_raw.groupby_bins("z", zbin).mean("z", skipna=True)
    # fix z coordinates: swap z_bins out for dz_uas grid
    znew = np.arange(dz_uas, zmax-dz_uas, dz_uas)
    # create new coordinate "z" from znew that is based on z_bins, then swap and drop
    uas_mean = uas_mean.assign_coords({"z": ("z_bins", znew)}).swap_dims({"z_bins": "z"})
    # # only save data for z <= h
    # h = err.z.max()
    # uas_mean = uas_mean.where(uas_mean.z <= h, drop=True)
    # calculate wspd, wdir from uas_mean profile
    uas_mean["wspd"] = (uas_mean.u**2. + uas_mean.v**2.) ** 0.5
    wdir = np.arctan2(-uas_mean.u, -uas_mean.v) * 180./np.pi
    wdir[wdir < 0.] += 360.
    uas_mean["wdir"] = wdir
    #
    # interpolate errors for everything in uas_mean
    #
    if err is not None:
        uas_mean["wspd_err"] = err.uh.interp(z=uas_mean.z)
        uas_mean["wdir_err"] = err.alpha.interp(z=uas_mean.z)
        uas_mean["theta_err"] = err.theta.interp(z=uas_mean.z)

    return uas_mean
# ---------------------------------------------
def ec_tow(ts, h, time_average=1800.0, time_start=0.0):
    """Emulate a tower extending throughout ABL with EC system at each vertical
    gridpoint and calculate variances and covariances
    :param xr.Dataset ts: dataset with virtual tower data to construct UAS prof
    :param float h: ABL depth in m
    :param float time_average: time range in s to avg timeseries; default=1800
    :param float time_start: when to initialize averaging; default=0.0
    :param bool quicklook: flag to make quicklook of raw vs averaged profiles
    Outputs new xarray Dataset with emulated vars and covars
    """
    # check if time_average is an array or single value and convert to iterable
    if np.shape(time_average) == ():
        time_average = np.array([time_average])
    else:
        time_average = np.array(time_average)
    # initialize empty dataset to hold everything
    ec_ = xr.Dataset(data_vars=None, coords=dict(z=ts.z, Tsample_ec=time_average))
    # loop through variable names to initialize empty DataArrays
    for v in ["uw_cov_tot","vw_cov_tot", "tw_cov_tot", "ustar2", 
              "u_var", "v_var", "w_var", "theta_var",
              "u_var_rot", "v_var_rot", "e"]:
        ec_[v] = xr.DataArray(data=np.zeros((len(ts.z), len(time_average)), 
                                            dtype=np.float64),
                              coords=dict(z=ts.z, Tsample_ec=time_average))
    # loop over time_average to calculate ec stats
    for jt, iT in enumerate(time_average):
        # first find the index in df.t that corresponds to time_start
        istart = int(time_start / ts.dt)
        # determine how many indices to use from time_average
        nuse = int(iT / ts.dt)
        # create array of indices to use
        iuse = np.linspace(istart, istart+nuse-1, nuse, dtype=np.int32)
        # begin calculating statistics
        # use the detrended stats from load_timeseries
        # u'w'
        ec_["uw_cov_tot"][:,jt] = ts.uw.isel(t=iuse).mean("t")
        # v'w'
        ec_["vw_cov_tot"][:,jt] = ts.vw.isel(t=iuse).mean("t")
        # theta'w'
        ec_["tw_cov_tot"][:,jt] = ts.tw.isel(t=iuse).mean("t")
        # ustar^2 = sqrt(u'w'^2 + v'w'^2)
        ec_["ustar2"][:,jt] = ((ec_.uw_cov_tot[:,jt]**2.) + (ec_.vw_cov_tot[:,jt]**2.)) ** 0.5
        # variances
        ec_["u_var"][:,jt] = ts.uu.isel(t=iuse).mean("t")
        ec_["u_var_rot"][:,jt] = ts.uur.isel(t=iuse).mean("t")
        ec_["v_var"][:,jt] = ts.vv.isel(t=iuse).mean("t")
        ec_["v_var_rot"][:,jt] = ts.vvr.isel(t=iuse).mean("t")
        ec_["w_var"][:,jt] = ts.ww.isel(t=iuse).mean("t")
        ec_["theta_var"][:,jt] = ts.tt.isel(t=iuse).mean("t")
        # calculate TKE
        ec_["e"][:,jt] = 0.5 * (ec_.u_var.isel(Tsample_ec=jt) +\
                                ec_.v_var.isel(Tsample_ec=jt) +\
                                ec_.w_var.isel(Tsample_ec=jt))
    
    # only return ec where z <= h
    return ec_.where(ec_.z <= h, drop=True)
# ---------------------------------------------
def coord_rotate_3D(nx, ny, nz, Lx, Ly, x, y, ca, sa, dat):
    """Rotate 3D field into new coordinate system aligned with mean
    wind at each height, interpolated into new grid.
    :param int nx: number of points in x-dimension
    :param int ny: number of points in x-dimension
    :param int nz: number of points in x-dimension
    :param float Lx: size of domain in x-dimension
    :param float Ly: size of domain in y-dimension
    :param list<Quantity> x: array of original x-axis values
    :param list<Quantity> y: array of original y-axis values
    :param ndarray ca: cosine of mean wind direction
    :param ndarray sa: sine of mean wind direction
    :param ndarray dat: 3-dimensional field to be rotated
    return dat_r: rotated field
    """
    # create big 3d array of repeating dat variable
    # to use as reference points when interpolating
    # interpolate spectrally first though
    nf = 2
    nx2, ny2 = nf*nx, nf*ny
    dat_spec = interpolate_spec_2d(dat, Lx, Ly, nf).to_numpy()
    # initialize empty array and fill with dat_spec
    datbig = np.zeros((4*nx2, 4*ny2, nz), dtype=np.float64)
    for jx in range(4):
        for jy in range(4):
            datbig[jx*nx2:(jx+1)*nx2,jy*ny2:(jy+1)*ny2,:] = dat_spec
    # create lab frame coordinates to match big variable array
    xbig = np.linspace(-2*Lx, 2*Lx, 4*nx2)
    ybig = np.linspace(-2*Ly, 2*Ly, 4*ny2)
    # create empty array to store interpolated values
    datinterp = np.zeros((nx, ny, nz), dtype=np.float64)
    # create meshgrid of lab coordinates
    xx, yy = np.meshgrid(x, y, indexing="xy")
    # begin loop over z
    for jz in range(nz):
        # calculate meshgrid of rotated reference frame at this height
        xxp = xx*ca[jz] - yy*sa[jz]
        yyp = xx*sa[jz] + yy*ca[jz]
        # create interpolator object for 2d slice of variable at jz
        interp = RegularGridInterpolator((xbig, ybig), datbig[:,:,jz])
        # create array of new coords to use
        points = np.array([xxp.ravel(), yyp.ravel()]).T
        # interpolate and store in variable array
        datinterp[:,:,jz] = interp(points, method="nearest").reshape(nx, ny).T

    return datinterp
# ---------------------------------------------
@njit
def calc_increments(nx, ny, nz, dat1, dat2):
    # initialize counter array for averaging later
    rcount = np.zeros(nx, dtype=np.float64)
    # initialize increment array
    delta2 = np.zeros((nx, ny, nz), dtype=np.float64)
    # begin loop over starting x
    for jx in range(nx):
        # loop over x lag
        for jr in range(nx):
            # test to see if starting position + lag are within domain
            if jx+jr < nx:
                # increment count
                rcount[jr] += 1.0
                # calculate increment
                d1 = dat1[jx+jr,:,:] - dat1[jx,:,:]
                d2 = dat2[jx+jr,:,:] - dat2[jx,:,:]
                incr = d1 * d2
                # store in delta2
                delta2[jr,:,:] += incr
    # normalize by rcount to average over lags
    Dout = np.zeros((nx, ny, nz), dtype=np.float64)
    for jr in range(nx):
        Dout[jr,:,:] = delta2[jr,:,:] / rcount[jr]
    # return Dout(lag, y, z)
    return Dout
# ---------------------------------------------
def xr_rotate(df):
    """Use coord_rotate_3D to rotate 3D Dataset and convert to a new
    Dataset with rotated coordinate system (x,y,z)
    :param Dataset df: 3-dimensional Dataset to rotate
    returns df_rot 3D Dataset
    """
    # some dimensions
    nx, ny, nz = df.nx, df.ny, df.nz
    Lx, Ly = df.Lx, df.Ly
    x, y, z = df.x, df.y, df.z
    # would like to use the following vars (basically ignore u_rot, v_rot)
    vuse = ["u","v","w","theta","q","txz","tyz","q3","wq_sgs","dissip"]
    # check against what is actually in df
    vhave = [key for key in vuse if key in list(df.keys())]
    # initialize empty Dataset to store rotated data
    df_rot = xr.Dataset(data_vars=None, coords=dict(x=x, y=y, z=z), attrs=df.attrs)
    # calculate wind angles
    u = df.u
    v = df.v
    # mean u, v
    u_bar = u.mean(dim=("x","y")).compute()
    v_bar = v.mean(dim=("x","y")).compute()
    # calc angle
    angle = np.arctan2(v_bar, u_bar)
    # angle cos and sin
    ca = np.cos(angle).values
    sa = np.sin(angle).values
    # loop over variables
    for var in vhave:
        # convert 3d dataarray to numpy
        dat = df[var]
        # rotate and interpolate
        vrot = coord_rotate_3D(nx, ny, nz, Lx, Ly, x, y, ca, sa, dat)
        # add vrot to df_jt as DataArray
        # has same x,y,z dimensions as original data!
        df_rot[var] = xr.DataArray(data=vrot, coords=dict(x=x, y=y, z=z))
    # now recalculate u_rot, v_rot
    u_bar_r = df_rot["u"].mean(dim=("x","y")).compute()
    v_bar_r = df_rot["v"].mean(dim=("x","y")).compute()
    angle_r = np.arctan2(v_bar_r, u_bar_r).compute()
    # calculate using xarray implicit
    df_rot["u_rot"] = df_rot.u*np.cos(angle_r) + df_rot.v*np.sin(angle_r)
    df_rot["v_rot"] =-df_rot.u*np.sin(angle_r) + df_rot.v*np.cos(angle_r)
    # add another attribute for rotate
    df_rot.attrs["rotate"] = "True"
    # return
    return df_rot
# ---------------------------------------------
def nc_rotate(dnc, t0, t1, dt):
    """Purpose: load netcdf all_TTTTTTT.nc files and rotate coordinates
    into mean streamwise flow using rot_interp, then save netcdf file with
    new coordinates.
    :param str dnc: absolute path to netcdf directory
    :param int t0: starting timestep
    :param int t1: ending timestep
    :param int dt: number of files between timesteps
    """
    # determine list of file timesteps from input
    timesteps = np.arange(t0, t1+1, dt, dtype=np.int32)
    # determine files to read from timesteps
    fall = [f"{dnc}all_{tt:07d}.nc" for tt in timesteps]
    # BEGIN TIME LOOP
    for jf, ff in enumerate(fall):
        # load file
        d = xr.load_dataset(ff)
        # use xr_rotate to rotate this Dataset
        Dsave = xr_rotate(d)
        # save to netcdf file and continue
        fsave = f"{dnc}all_{timesteps[jf]:07d}_rot.nc"
        print(f"Saving file: {fsave.split(os.sep)[-1]}")
        Dsave.to_netcdf(fsave, mode="w")
    
    print("Finished saving all files!")
    return
# ---------------------------------------------
def interpolate_spec_2d(d, Lx, Ly, nf):
    """Interpolate a 3d dataarray d spectrally in x,y wavenumbers
    input d: original data
    input Lx: length of domain in x-dimension
    input Ly: length of domain in x-dimension
    input nf: factor by which to increase number of points in both x & y dim
    output dint: interpolated DataArray
    """
    # compute new arrays of x, y for storing into dataarray later
    nx, ny = d.x.size, d.y.size
    nx2, ny2 = nf*nx, nf*ny
    x2, y2 = np.linspace(0, Lx, nx2), np.linspace(0, Ly, ny2)
    # take FFT of d
    f_d = xrft.fft(d, dim=("x","y"), true_phase=True, 
                   true_amplitude=True).compute()
    # pad by factor nf
    npadx = (nx//2)*(nf-1)
    npady = (ny//2)*(nf-1)
    f_d_pad = xrft.padding.pad(f_d, freq_x=npadx, freq_y=npady)
    # take IFFT of padded array
    d_int = xrft.ifft(f_d_pad, dim=("freq_x","freq_y"), 
                      true_phase=True, true_amplitude=True,
                      lag=(f_d.freq_x.direct_lag, f_d.freq_y.direct_lag)).real
    # redefine x and y in d_int then return
    d_int["x"] = x2
    d_int["y"] = y2
    return d_int