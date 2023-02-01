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
import sys
import xrft
import xarray as xr
import numpy as np
from numpy.fft import fft, ifft
from scipy.signal import detrend
from scipy.optimize import curve_fit
from dask.diagnostics import ProgressBar
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
def sim2netcdf(dout, dnc, resolution, dimensions, scales, t0, t1, dt, 
               use_dissip, simlabel, units=None):
    """Read binary output files from LES code and combine into one netcdf
    file per timestep using xarray for future reading and easier analysis

    :param str dout: absolute path to directory with LES output binary files
    :param str dnc: absolute path to directory for saving output netCDF files
    :param tuple<int> resolution: simulation resolution (nx, ny, nz)
    :param tuple<Quantity> dimensions: simulation dimensions (Lx, Ly, Lz)
    :param tuple<Quantity> scales: dimensional scales from LES code\
        (uscale, Tscale)
    :param int t0: first timestep to process
    :param int t1: last timestep to process
    :param int dt: number of timesteps between files to load
    :param bool use_dissip: flag for loading dissipation rate files (SBL only)
    :param str simlabel: unique identifier for batch of files
    :param dict units: dictionary of units corresponding to each loaded\
        variable. Default values hard-coded in this function
    """
    # check if dnc exists
    if not os.path.exists(dnc):
        os.mkdir(dnc)
    # grab relevent parameters
    nx, ny, nz = resolution
    Lx, Ly, Lz = dimensions
    dx, dy, dz = Lx/nx, Ly/ny, Lz/nz
    u_scale = scales[0]
    theta_scale = scales[1]
    # define timestep array
    timesteps = np.arange(t0, t1+1, dt, dtype=np.int32)
    nt = len(timesteps)
    # dimensions
    x, y = np.linspace(0., Lx, nx), np.linspace(0, Ly, ny)
    # u- and w-nodes are staggered
    # zw = 0:Lz:nz
    # zu = dz/2:Lz-dz/2:nz-1
    # interpolate w, txz, tyz, q3 to u grid
    zw = np.linspace(0., Lz, nz)
    zu = np.linspace(dz/2., Lz+dz/2., nz)
    # --------------------------------
    # Loop over timesteps to load and save new files
    # --------------------------------
    for i in range(nt):
        # load files - DONT FORGET SCALES!
        f1 = f"{dout}u_{timesteps[i]:07d}.out"
        u_in = read_f90_bin(f1,nx,ny,nz,8) * u_scale
        f2 = f"{dout}v_{timesteps[i]:07d}.out"
        v_in = read_f90_bin(f2,nx,ny,nz,8) * u_scale
        f3 = f"{dout}w_{timesteps[i]:07d}.out"
        w_in = read_f90_bin(f3,nx,ny,nz,8) * u_scale
        f4 = f"{dout}theta_{timesteps[i]:07d}.out"
        theta_in = read_f90_bin(f4,nx,ny,nz,8) * theta_scale
        f5 = f"{dout}txz_{timesteps[i]:07d}.out"
        txz_in = read_f90_bin(f5,nx,ny,nz,8) * u_scale * u_scale
        f6 = f"{dout}tyz_{timesteps[i]:07d}.out"
        tyz_in = read_f90_bin(f6,nx,ny,nz,8) * u_scale * u_scale
        f7 = f"{dout}q3_{timesteps[i]:07d}.out"
        q3_in = read_f90_bin(f7,nx,ny,nz,8) * u_scale * theta_scale
        # interpolate w, txz, tyz, q3 to u grid
        # create DataArrays
        w_da = xr.DataArray(w_in, dims=("x", "y", "z"), coords=dict(x=x, y=y, z=zw))
        txz_da = xr.DataArray(txz_in, dims=("x", "y", "z"), coords=dict(x=x, y=y, z=zw))
        tyz_da = xr.DataArray(tyz_in, dims=("x", "y", "z"), coords=dict(x=x, y=y, z=zw))
        q3_da = xr.DataArray(q3_in, dims=("x", "y", "z"), coords=dict(x=x, y=y, z=zw))
        # perform interpolation
        w_interp = w_da.interp(z=zu, method="linear", 
                               kwargs={"fill_value": "extrapolate"})
        txz_interp = txz_da.interp(z=zu, method="linear", 
                                   kwargs={"fill_value": "extrapolate"})
        tyz_interp = tyz_da.interp(z=zu, method="linear", 
                                   kwargs={"fill_value": "extrapolate"})
        q3_interp = q3_da.interp(z=zu, method="linear", 
                                 kwargs={"fill_value": "extrapolate"})
        # construct dictionary of data to save -- u-node variables only!
        data_save = {
                        "u": (["x","y","z"], u_in),
                        "v": (["x","y","z"], v_in),
                        "theta": (["x","y","z"], theta_in),
                    }
        # check fo using dissipation files
        if use_dissip:
            # read binary file
            f8 = f"{dout}dissip_{timesteps[i]:07d}.out"
            diss_in = read_f90_bin(f8,nx,ny,nz,8) * u_scale * u_scale * u_scale / Lz
            # interpolate to u-nodes
            diss_da = xr.DataArray(diss_in, dims=("x", "y", "z"), 
                                   coords=dict(x=x, y=y, z=zw))
            diss_interp = diss_da.interp(z=zu, method="linear", 
                                         kwargs={"fill_value": "extrapolate"})
        # construct dataset from these variables
        ds = xr.Dataset(
            data_save,
            coords={
                "x": x,
                "y": y,
                "z": zu
            },
            attrs={
                "nx": nx,
                "ny": ny,
                "nz": nz,
                "Lx": Lx,
                "Ly": Ly,
                "Lz": Lz,
                "dx": dx,
                "dy": dy,
                "dz": dz,
                "label": simlabel
            })
        # now assign interpolated arrays that were on w-nodes
        ds["w"] = w_interp
        ds["txz"] = txz_interp
        ds["tyz"] = tyz_interp
        ds["q3"] = q3_interp
        if use_dissip:
            ds["dissip"] = diss_interp
        # hardcode dictionary of units to use by default
        if units is None:
            units = {
                "u": "m/s",
                "v": "m/s",
                "w": "m/s",
                "theta": "K",
                "txz": "m^2/s^2",
                "tyz": "m^2/s^2",
                "q3": "K m/s",
                "dissip": "m^2/s^3",
                "x": "m",
                "y": "m",
                "z": "m"
            }

        # loop and assign attributes
        for var in list(data_save.keys())+["x", "y", "z"]:
            ds[var].attrs["units"] = units[var]
        # save to netcdf file and continue
        fsave = f"{dnc}all_{timesteps[i]:07d}.nc"
        print(f"Saving file: {fsave.split(os.sep)[-1]}")
        ds.to_netcdf(fsave)

    print("Finished saving all files!")
    return
# ---------------------------------------------
def calc_stats(dnc, t0, t1, dt, delta_t, use_dissip, detrend_stats, tavg):
    """Read multiple output netcdf files created by sim2netcdf() to calculate
    averages in x, y, t and save as new netcdf file

    :param str dnc: absolute path to directory for loading netCDF files
    :param int t0: first timestep to process
    :param int t1: last timestep to process
    :param int dt: number of timesteps between files to load
    :param float delta_t: dimensional timestep in simulation (seconds)
    :param bool use_dissip: flag for loading dissipation rate files (SBL only)
    :param bool detrend_stats: flag for detrending fields in time when\
        calculating statistics
    :param str tavg: label denoting length of temporal averaging (e.g. 1h)
    """
    # directories and configuration
    timesteps = np.arange(t0, t1, dt, dtype=np.int32)
    # determine files to read from timesteps
    fall = [f"{dnc}all_{tt:07d}.nc" for tt in timesteps]
    nf = len(fall)
    # calculate array of times represented by each file
    times = np.array([i*delta_t*dt for i in range(nf)])
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
    dd_stat = xr.Dataset()
    # list of base variables
    base = ["u", "v", "w", "theta"]
    base1 = ["u", "v", "w", "theta"] # use for looping over vars in case dissip not used
    # check for dissip
    if use_dissip:
        base.append("dissip")
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
    # calculate vars
    for s in base1:
        if detrend_stats:
            vv = np.var(detrend(dd[s], axis=0, type="linear"), axis=(0,1,2))
            dd_stat[f"{s}_var"] = xr.DataArray(vv, dims=("z"), coords=dict(z=dd.z))
        else:
            dd_stat[f"{s}_var"] = dd[s].var(dim=("time", "x", "y"))
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
    if detrend_stats:
        uvar_rot = np.var(detrend(u_rot, axis=0, type="linear"), axis=(0,1,2))
        dd_stat["u_var_rot"] = xr.DataArray(uvar_rot, dims=("z"), coords=dict(z=dd.z))
        vvar_rot = np.var(detrend(v_rot, axis=0, type="linear"), axis=(0,1,2))
        dd_stat["v_var_rot"] = xr.DataArray(vvar_rot, dims=("z"), coords=dict(z=dd.z))
    else:
        dd_stat["u_var_rot"] = u_rot.var(dim=("time", "x", "y"))
        dd_stat["v_var_rot"] = v_rot.var(dim=("time", "x", "y"))
    # --------------------------------
    # Add attributes
    # --------------------------------
    # copy from dd
    dd_stat.attrs = dd.attrs
    dd_stat.attrs["delta"] = (dd.dx * dd.dy * dd.dz) ** (1./3.)
    dd_stat.attrs["tavg"] = tavg
    # --------------------------------
    # Save output file
    # --------------------------------
    fsave = f"{dnc}mean_stats_xyt_{tavg}.nc"
    print(f"Saving file: {fsave}")
    with ProgressBar():
        dd_stat.to_netcdf(fsave, mode="w")
    print("Finished!")
    return
# ---------------------------------------------
def load_stats(fstats, SBL=True, display=False):
    """Reading function for average statistics files created from calc_stats()
    Load netcdf files using xarray and calculate numerous relevant parameters
    :param str fstats: absolute path to netcdf file for reading
    :param bool SBL: denote whether sim data is SBL or not to calculate\
        appropriate ABL depth etc.
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
    # grab ustar0 and calc tstar0 for normalizing in plotting
    dd["ustar0"] = dd.ustar.isel(z=0)
    dd["tstar0"] = -dd.tw_cov_tot.isel(z=0)/dd.ustar0
    # local thetastar
    dd["tstar"] = -dd.tw_cov_tot / dd.ustar
    # calculate TKE
    dd["e"] = 0.5 * (dd.u_var + dd.v_var + dd.w_var)
    # calculate Obukhov length L
    dd["L"] = -(dd.ustar0**3) * dd.theta_mean.isel(z=0) / (0.4 * 9.81 * dd.tw_cov_tot.isel(z=0))
    # calculate uh and wdir
    dd["uh"] = np.sqrt(dd.u_mean**2. + dd.v_mean**2.)
    dd["wdir"] = np.arctan2(-dd.u_mean, -dd.v_mean) * 180./np.pi
    dd["wdir"] = dd.wdir.where(dd.wdir < 0.) + 360.
    # calculate mean lapse rate between lowest grid point and z=h
    delta_T = dd.theta_mean.sel(z=dd.h, method="nearest") - dd.theta_mean[0]
    delta_z = dd.z.sel(z=dd.h, method="nearest") - dd.z[0]
    dd["dT_dz"] = delta_T / delta_z
    # calculate eddy turnover time TL
    dd["TL"] = dd.h / dd.ustar0
    dd["nTL"] = 3600. / dd.TL
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
        # save number of points within sbl
        dd.attrs["nzsbl"] = dd.z.where(dd.z <= dd.h, drop=True).size
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
def load_full(dnc, t0, t1, dt, delta_t, SBL=False, stats=None):
    """Reading function for multiple instantaneous volumetric netcdf files
    Load netcdf files using xarray
    :param str dnc: abs path directory for location of netcdf files
    :param int t0: first timestep to process
    :param int t1: last timestep to process
    :param int dt: number of timesteps between files to load
    :param float delta_t: dimensional timestep in simulation (seconds)
    :param bool SBL: calculate SBL-specific parameters. defaulte=False
    :param str stats: name of statistics file. default=None

    :return dd: xarray dataset of 4d volumes
    :return s: xarray dataset of statistics file
    """
    # load final hour of individual files into one dataset
    # note this is specific for SBL simulations
    timesteps = np.arange(t0, t1+1, dt, dtype=np.int32)
    # determine files to read from timesteps
    fall = [f"{dnc}all_{tt:07d}.nc" for tt in timesteps]
    nf = len(fall)
    # calculate array of times represented by each file
    times = np.array([i*delta_t*dt for i in range(nf)])
    # read files
    print("Loading files...")
    dd = xr.open_mfdataset(fall, combine="nested", concat_dim="time")
    dd.coords["time"] = times
    dd.time.attrs["units"] = "s"
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
    # just return dd if no SBL
    return dd
# ---------------------------------------------