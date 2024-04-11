#!/home/bgreene/anaconda3/envs/LES/bin/python
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
from glob import glob
from dask.diagnostics import ProgressBar
from scipy.interpolate import RegularGridInterpolator
from scipy.fft import fft2, ifft2, fftshift, ifftshift
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
    # u
    f01 = f"{dout}u_{timestep:07d}.out"
    u_in = read_f90_bin(f01,nx,ny,nz,8) * u_scale
    # v
    f02 = f"{dout}v_{timestep:07d}.out"
    v_in = read_f90_bin(f02,nx,ny,nz,8) * u_scale
    # w
    f03 = f"{dout}w_{timestep:07d}.out"
    w_in = read_f90_bin(f03,nx,ny,nz,8) * u_scale
    # theta
    f04 = f"{dout}theta_{timestep:07d}.out"
    theta_in = read_f90_bin(f04,nx,ny,nz,8) * theta_scale
    # q
    f05 = f"{dout}q_{timestep:07d}.out"
    q_in = read_f90_bin(f05,nx,ny,nz,8) * q_scale
    # p
    f06 = f"{dout}p_{timestep:07d}.out"
    p_in = read_f90_bin(f06,nx,ny,nz,8) # unsure how to dim
    # dissipation
    f07 = f"{dout}dissip_{timestep:07d}.out"
    diss_in = read_f90_bin(f07,nx,ny,nz,8) * u_scale * u_scale * u_scale / Lz
    # 6 tau_ij terms
    f08 = f"{dout}txx_{timestep:07d}.out"
    txx_in = read_f90_bin(f08,nx,ny,nz,8) * u_scale * u_scale
    f09 = f"{dout}txy_{timestep:07d}.out"
    txy_in = read_f90_bin(f09,nx,ny,nz,8) * u_scale * u_scale
    f10 = f"{dout}txz_{timestep:07d}.out"
    txz_in = read_f90_bin(f10,nx,ny,nz,8) * u_scale * u_scale
    f11 = f"{dout}tyy_{timestep:07d}.out"
    tyy_in = read_f90_bin(f11,nx,ny,nz,8) * u_scale * u_scale
    f12 = f"{dout}tyz_{timestep:07d}.out"
    tyz_in = read_f90_bin(f12,nx,ny,nz,8) * u_scale * u_scale
    f13 = f"{dout}tzz_{timestep:07d}.out"
    tzz_in = read_f90_bin(f13,nx,ny,nz,8) * u_scale * u_scale
    # tw_sgs
    f14 = f"{dout}sgs_t3_{timestep:07d}.out"
    tw_sgs_in = read_f90_bin(f14,nx,ny,nz,8) * u_scale * theta_scale
    # qw_sgs
    f15 = f"{dout}sgs_q3_{timestep:07d}.out"
    qw_sgs_in = read_f90_bin(f15,nx,ny,nz,8) * u_scale * q_scale  
    # e_sgs
    f16 = f"{dout}e_sgs_{timestep:07d}.out"
    e_sgs_in = read_f90_bin(f16,nx,ny,nz,8) * u_scale * u_scale

    # list of all out files for cleanup later
    fout_all = [f01, f02, f03, f04, f05, f06, f07, f08, f09,
                f10, f11, f12, f13, f14, f15, f16]
    # interpolate w, sgs terms (incl. e), and dissip to u grid
    # create DataArrays
    w_da = xr.DataArray(w_in, dims=("x", "y", "z"), 
                        coords=dict(x=x, y=y, z=zw))
    txx_da = xr.DataArray(txx_in, dims=("x", "y", "z"), 
                          coords=dict(x=x, y=y, z=zw))
    txy_da = xr.DataArray(txy_in, dims=("x", "y", "z"), 
                          coords=dict(x=x, y=y, z=zw))
    txz_da = xr.DataArray(txz_in, dims=("x", "y", "z"), 
                          coords=dict(x=x, y=y, z=zw))
    tyy_da = xr.DataArray(tyy_in, dims=("x", "y", "z"), 
                          coords=dict(x=x, y=y, z=zw))
    tyz_da = xr.DataArray(tyz_in, dims=("x", "y", "z"), 
                          coords=dict(x=x, y=y, z=zw))
    tzz_da = xr.DataArray(tzz_in, dims=("x", "y", "z"), 
                          coords=dict(x=x, y=y, z=zw))
    tw_sgs_da = xr.DataArray(tw_sgs_in, dims=("x", "y", "z"), 
                             coords=dict(x=x, y=y, z=zw))
    qw_sgs_da = xr.DataArray(qw_sgs_in, dims=("x", "y", "z"), 
                             coords=dict(x=x, y=y, z=zw))
    diss_da = xr.DataArray(diss_in, dims=("x", "y", "z"), 
                           coords=dict(x=x, y=y, z=zw))
    e_sgs_da = xr.DataArray(e_sgs_in, dims=("x", "y", "z"), 
                            coords=dict(x=x, y=y, z=zw))
    # perform interpolation
    w_interp = w_da.interp(z=zu, method="linear", 
                           kwargs={"fill_value": "extrapolate"})
    txx_interp = txx_da.interp(z=zu, method="linear", 
                               kwargs={"fill_value": "extrapolate"})
    txy_interp = txy_da.interp(z=zu, method="linear", 
                               kwargs={"fill_value": "extrapolate"})
    txz_interp = txz_da.interp(z=zu, method="linear", 
                               kwargs={"fill_value": "extrapolate"})
    tyy_interp = tyy_da.interp(z=zu, method="linear", 
                               kwargs={"fill_value": "extrapolate"})
    tyz_interp = tyz_da.interp(z=zu, method="linear", 
                               kwargs={"fill_value": "extrapolate"})
    tzz_interp = tzz_da.interp(z=zu, method="linear", 
                               kwargs={"fill_value": "extrapolate"})
    tw_sgs_interp = tw_sgs_da.interp(z=zu, method="linear", 
                                     kwargs={"fill_value": "extrapolate"})
    qw_sgs_interp = qw_sgs_da.interp(z=zu, method="linear", 
                                     kwargs={"fill_value": "extrapolate"})
    diss_interp = diss_da.interp(z=zu, method="linear", 
                                 kwargs={"fill_value": "extrapolate"})
    e_sgs_interp = e_sgs_da.interp(z=zu, method="linear", 
                                   kwargs={"fill_value": "extrapolate"})
    # construct dictionary of data to save -- u-node variables only!
    data_save = {
                 "u": (["x","y","z"], u_in),
                 "v": (["x","y","z"], v_in),
                 "theta": (["x","y","z"], theta_in),
                 "q": (["x","y","z"], q_in),
                 "p": (["x","y","z"], p_in)
                }

    # construct dataset from these variables
    ds = xr.Dataset(data_save, coords=dict(x=x, y=y, z=zu), attrs=params)
    # now assign interpolated arrays that were on w-nodes
    ds["w"] = w_interp
    ds["txx"] = txx_interp
    ds["txy"] = txy_interp
    ds["txz"] = txz_interp
    ds["tyy"] = tyy_interp
    ds["tyz"] = tyz_interp
    ds["tzz"] = tzz_interp
    ds["tw_sgs"] = tw_sgs_interp
    ds["qw_sgs"] = qw_sgs_interp
    ds["dissip"] = diss_interp
    ds["e_sgs"] = e_sgs_interp
    # hardcode dictionary of units to use by default
    units = {
            "u": "m/s",
            "v": "m/s",
            "w": "m/s",
            "theta": "K",
            "q": "g/kg",
            "p": "n/a",
            "txx": "m^2/s^2",
            "txy": "m^2/s^2",
            "txz": "m^2/s^2",
            "tyy": "m^2/s^2",
            "tyz": "m^2/s^2",
            "tzz": "m^2/s^2",
            "tw_sgs": "K m/s",
            "qw_sgs": "m/s g/kg",
            "dissip": "m^2/s^3",
            "e_sgs": "m^2/s^2",
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
def process_raw_sim(dout, nhr, nhr_s, del_raw, organize, nrun, 
                    overwrite=False, cstats=True, rotate=False, 
                    ts2nc=False, del_remaining=False):
    """Use information from dout/param.yaml file to dynamically process raw 
    output files with out2netcdf function for a desired time period of files.
    Optional additional processing: calc_stats(), nc_rot().
    -Input-
    dout: string, absolute path to output files
    nhr: float, number of physical hours to process. Will use information from\
        param file to dynamically select files.
    nhr_s: string, label for outputting stats file matching nhr (e.g., 5-6h)
    del_raw: boolean, flag to pass to out2netcdf for cleaning up raw files
    organize: boolean, flag to call organize_output on batch job output
    nrun: integer, number of independent job batches to pass to organize
    overwrite: boolean, flag to overwrite output file in case it already exists.
    cstats: boolean, call calc_stats on the range determined for out2netcdf.
    rotate: boolean, call nc_rot on the same files as out2netcdf.
    del_remaining: boolean, delete all remaining raw files (not timeseries).
    -Output-
    single netcdf file for each timestep in the range nhr.
    if cstats, also return filename created
    """
    # import yaml file
    with open(dout+"params.yaml") as fp:
        params = yaml.safe_load(fp)
    # add simulation label to params
    params["simlabel"] = params["path"].split(os.sep)[-2]
    # add nhr_s to params
    params["nhr_s"] = nhr_s
    # determine range of files from info in params
    tf = params["jt_final"]
    nt = int(nhr * 3600. / params["dt"])
    t0 = tf - nt
    timesteps = np.arange(t0, tf+1, params["nwrite"], dtype=np.int32)
    nf = len(timesteps)
    print(f"Processing {nhr} hours = {nf} timesteps from t0={t0} to tf={tf}")
    # check if netcdf directory exists
    dnc = f"{dout}netcdf/"
    if not os.path.exists(dnc):
        os.system(f"mkdir {dnc}")
    # add dnc to params
    params["dnc"] = dnc
    # first need to organize sim output by calling organize_output()
    if organize:
        organize_output(dout, nrun)

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
        params["return_fstats"] = True
        fstats = calc_stats(f_use=f_all, **params)
    else:
        fstats = None

    # run nc_rotate()
    if rotate:
        print("Begin rotating fields and saving output...")
        nc_rotate(dnc, t0, tf, params["nwrite"])

    # run timeseries2netcdf()
    if ts2nc:
        print("Begin processing timeseries data...")
        timeseries2netcdf(dout, del_npz=del_raw, **params)

    # delete all remaining raw files to save space
    if del_remaining:
        print("Delete remaining raw files to save storage space...")
        # only files left should be .out
        # timeseries files should have been converted to .npz already
        os.system(f"rm {dout}*.out")

    print(f"Finished processing simulation {params['simlabel']}!")
    return fstats
# ---------------------------------------------
def organize_output(dout, nrun):
    """For simulations that are split into batches, loop over
    runs and combine files into single directory. Volume files
    have cumulative timesteps, but timeseries do not. Use
    params.yaml files for reference.
    -Input-
    dout: string, absolute path to output files
    nrun: integer, number of runXX folders to loop over
    -Output-
    none
    """
    # current workflow: nrun-1 folders, with run==nrun in dout
    # lets create new directory to make looping easier
    os.system(f"mkdir -p {dout}run{nrun:02d}/")
    os.system(f"mv {dout}*.* {dout}run{nrun:02d}/")
    # create a directory in dout to hold everything as we loop
    dall = f"{dout}runall/"
    os.system(f"mkdir -p {dall}")
    vall_v = ["u", "v", "w", "theta", "q", "p", 
              "txx", "txy", "txz", "tyy", "tyz", "tzz",
              "sgs_t3", "sgs_q3", "e_sgs", "dissip"]
    vall_t = ["u", "v", "w", "t", "q", "p", "tv",
              "txx", "txy", "txz", "tyy", "tyz", "tzz",
              "sgs_t3", "sgs_q3", "e_sgs", "dissip"]
    # dictionary to keep track of which runs have timeseries, and how long
    tslog = {}
    # begin looping over runs
    for r in range(1, nrun+1):
        print(f"Begin run: {r}")
        drun = f"{dout}run{r:02d}/"
        # load yaml file
        with open(drun+"params.yaml") as f:
            params = yaml.safe_load(f)
        # move files
        for v in vall_v:
            print(f"Moving volume files: {v}")
            fmove = glob(f"{drun}{v}_*.out")
            # move volume files, ignore timeseries files
            fvol = [ff for ff in fmove if "timeseries" not in ff]
            for ff in fvol:
                os.system(f"mv {ff} {dall}")
        # move timeseries files
        # first, check if there are any in this folder
        if len(glob(f"{drun}*timeseries*.out")) > 0:
            for v in vall_t:
                print(f"Moving timeseries files...")
                # combine all ts file levels into one
                # grab dimensions from params
                nz = params["nz"]
                nz_mpi = params["nz_mpi"]
                nmpi = nz // nz_mpi
                # now loop through each file (one for each mpi process)
                # there are nz_mpi columns in each file!
                # initialize jz counter index for first nz_mpi columns
                jz = np.arange(1, nz_mpi+1, dtype=np.int32)
                # initialze range counter for which columns to use
                cols = range(1, nz_mpi+1)
                # create numpy array to hold all levels
                # get more from params
                jt_start = params["jt_total_init"]
                jt_final = params["jt_final"]
                nt = params["nsteps"]
                # add nt to tslog
                tslog[f"{r:02d}"] = nt
                # initialize
                tsv = np.zeros((nt, nz+1), dtype=np.float64)
                # first column is timestamp
                tsv[:,0] = np.arange(jt_start, jt_final, dtype=np.float64)
                # add 
                for jmpi in range(nmpi):
                    fts = f"{drun}{v}_timeseries_c{jmpi:03d}.out"
                    # load files and store in tsv
                    tsv[:,jz] = np.loadtxt(fts, usecols=cols)
                    # increment jz!
                    jz += nz_mpi
                # save tsv as new .npz file in dall
                fts_new = f"{dall}{v}_timeseries_run{r:02d}.npz"
                print(f"Saving file: {fts_new}")
                np.savez(fts_new, tsv)
    # outside all loops
    # combine timeseries files into one per variable
    print("Cleanup timeseries files...")
    nt_all = sum(tslog.values())
    # loop over variables
    for v in vall_t:
        # initialize big array to hold all runs
        # nz should still be the valid value
        tsfull = np.zeros((nt_all, nz+1), dtype=np.float64)
        # initialize counter
        tcount = 0
        # loop over runs
        for r, nt in tslog.items():
            # load timeseries npz file
            fts_new = f"{dall}{v}_timeseries_run{r}.npz"
            # store in tsfull
            tsfull[tcount:tcount+nt,:] = np.load(fts_new)["arr_0"]
            # increment tcount
            tcount += nt
            # delete fts_new 
            os.system(f"rm {fts_new}")
        # save tsfull as new file
        fts_all = f"{dall}{v}_timeseries.npz"
        print(f"Saving file: {fts_all}")
        np.savez(fts_all, tsfull)
    # move everything in dall up a dir into output/
    print("Moving files into output/ and cleaning up remaining directories")
    os.system(f"mv {dall}* {dout}")
    # move params.yaml files and delete directories
    for r in range(1, nrun+1):
        drun = f"{dout}run{r:02d}/"
        if r == nrun:
            pnew = "params.yaml"
        else:
            pnew = f"params{r:02d}.yaml"
        os.system(f"mv {drun}params.yaml {dout}{pnew}")
        print(f"Removing directory: {drun}")
        os.system(f"rm -r {drun}")
    
    print("Finished cleaning up simulation output!")
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
    statistics file with format 'mean_stats_xyt_<nhr_s>.nc'
    """
    # make sure params is not empty
    if len(params.keys()) < 1:
        print("No parameters provided. Returning without proceeding.")
        return
    # construct dnc
    # dnc = f"{params['path']}output/netcdf/"
    dnc = params["dnc"]
    # option 1: f_use is not None
    if f_use is not None:
        fall = f_use
    elif nhr is not None:
        # determine range of files from info in params
        tf = params["jt_final"]
        nt = int(nhr * 3600. / params["dt"])
        t0 = tf - nt
        timesteps = np.arange(t0, tf+1, params["nwrite"], dtype=np.int32)
        # determine files to read from timesteps
        fall = [f"{dnc}all_{tt:07d}.nc" for tt in timesteps]
    # get length of file list
    nf = len(fall)
    # list of base variables
    base = ["u", "v", "w", "theta", "q", "dissip", "p", "thetav"]
    # use for looping over vars in case dissip not used
    base1 = ["u", "v", "w", "theta", "q", "thetav"]
    # list of remaining variables to average over
    base2 = ["txx", "txy", "txz", "tyy", "tyz", "tzz", 
             "tw_sgs", "qw_sgs", "e_sgs"]
    # create empty dataset that will hold everything
    dd_stat = xr.Dataset(data_vars=None)
    # --------------------------------
    # Begin looping over files and load one by one
    # --------------------------------
    for jf, ff in enumerate(fall):
        # load file
        print(f"Loading file: {ff}")
        dd = xr.load_dataset(ff)
        # --------------------------------
        # Calculate statistics
        # --------------------------------
        # calculate thetav
        dd["thetav"] = dd.theta * (1. + 0.61*dd.q/1000.)
        # calculate means
        for s in base+base2:
            # if this is first file, create new data variable
            if jf == 0:
                dd_stat[f"{s}_mean"] = dd[s].mean(dim=("x","y"))
            # otherwise, add to existing array for averaging in time later
            else:
                dd_stat[f"{s}_mean"] += dd[s].mean(dim=("x","y"))
        # calculate covars
        if jf == 0:
            # u'w'
            dd_stat["uw_cov_res"] = xr.cov(dd.u, dd.w, dim=("x","y"))
            # v'w'
            dd_stat["vw_cov_res"] = xr.cov(dd.v, dd.w, dim=("x","y"))
            # theta'w'
            dd_stat["tw_cov_res"] = xr.cov(dd.theta, dd.w, dim=("x","y"))
            # q'w'
            dd_stat["qw_cov_res"] = xr.cov(dd.q, dd.w, dim=("x","y"))
        else:
            # u'w'
            dd_stat["uw_cov_res"] += xr.cov(dd.u, dd.w, dim=("x","y"))
            # v'w'
            dd_stat["vw_cov_res"] += xr.cov(dd.v, dd.w, dim=("x","y"))
            # theta'w'
            dd_stat["tw_cov_res"] += xr.cov(dd.theta, dd.w, dim=("x","y"))
            # q'w'
            dd_stat["qw_cov_res"] += xr.cov(dd.q, dd.w, dim=("x","y"))
        # calculate vars
        for s in base1:
            if jf == 0:
                dd_stat[f"{s}_var"] = dd[s].var(dim=("x","y"))
            else:
                dd_stat[f"{s}_var"] += dd[s].var(dim=("x","y"))
        # rotate instantaneous u and v for variances
        # (not sure if necessary by commutative property but might as well)
        angle_inst = np.arctan2(dd.v.mean(dim=("x","y")), dd.u.mean(dim=("x","y")))
        u_rot = dd.u*np.cos(angle_inst) + dd.v*np.sin(angle_inst)
        v_rot =-dd.u*np.sin(angle_inst) + dd.v*np.cos(angle_inst)
        if jf == 0:
            dd_stat["u_var_rot"] = u_rot.var(dim=("x","y"))
            dd_stat["v_var_rot"] = v_rot.var(dim=("x","y"))
        else:
            dd_stat["u_var_rot"] += u_rot.var(dim=("x","y"))
            dd_stat["v_var_rot"] += v_rot.var(dim=("x","y"))
        # calculate <theta'q'>
        if jf == 0:
            dd_stat["tq_cov_res"] = xr.cov(dd.theta, dd.q, dim=("x","y"))
        else:
            dd_stat["tq_cov_res"] += xr.cov(dd.theta, dd.q, dim=("x","y"))
    # --------------------------------
    # OUTSIDE LOOP
    # --------------------------------
    # divide all variables by nf to average in time
    dd_stat /= float(nf)
    # now can compute total fluxes with sgs components
    dd_stat["uw_cov_tot"] = dd_stat.uw_cov_res + dd_stat.txz_mean
    dd_stat["vw_cov_tot"] = dd_stat.vw_cov_res + dd_stat.tyz_mean
    dd_stat["tw_cov_tot"] = dd_stat.tw_cov_res + dd_stat.tw_sgs_mean
    dd_stat["qw_cov_tot"] = dd_stat.qw_cov_res + dd_stat.qw_sgs_mean
    dd_stat["tvw_cov_tot"] = dd_stat.tw_cov_tot +\
                0.61 * dd_stat.thetav_mean[0] * dd_stat.qw_cov_tot/1000.
    # rotate u_mean and v_mean so <v> = 0
    angle = np.arctan2(dd_stat.v_mean, dd_stat.u_mean)
    dd_stat["u_mean_rot"] = dd_stat.u_mean*np.cos(angle) + dd_stat.v_mean*np.sin(angle)
    dd_stat["v_mean_rot"] =-dd_stat.u_mean*np.sin(angle) + dd_stat.v_mean*np.cos(angle)
    # --------------------------------
    # Add attributes
    # --------------------------------
    dd_stat.attrs = dd.attrs
    # grid spacing
    dd_stat.attrs["delta"] = (dd.dx * dd.dy * dd.dz) ** (1./3.)
    # --------------------------------
    # Save output file
    # --------------------------------
    fsave = f"{dnc}mean_stats_xyt_{params['nhr_s']}.nc"
    print(f"Saving file: {fsave}")
    with ProgressBar():
        dd_stat.to_netcdf(fsave, mode="w")
    print("Finished!")
    # check if "return_fstats" is in params to return filename
    if "return_fstats" in params.keys():
        if params["return_fstats"]:
            return fsave.split(os.sep)[-1]

    return
# ---------------------------------------------
def load_stats(fstats):
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
    # calculate ustar
    dd["ustar"] = ((dd.uw_cov_tot**2.) + (dd.vw_cov_tot**2.)) ** 0.25
    dd["ustar2"] = dd.ustar ** 2.
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
    # calc ABL depth h
    if dd.lbc == 0:
        # dd["h"] = dd.z.where(dd.ustar2 <= 0.05*dd.ustar2[0], drop=True)[0] / 0.95
        # use 1% threshold instead of 5%
        # dd["h"] = dd.z.where(dd.ustar2 <= 0.01*dd.ustar2[0], drop=True)[0] / 0.99
        # use FWHM of LLJ compared with Ug for h
        dG = dd.uh.max() - 8.
        dd["h"] = dd.z.where(dd.uh >= 8 + 0.5*dG, drop=True)[-1]
    else:
        # CBL
        dd["h"] = dd.z.isel(z=dd.tw_cov_tot.argmin())
    # save number of points within abl (z <= h)
    dd.attrs["nzabl"] = dd.z.where(dd.z <= dd.h, drop=True).size

    # calculate mean lapse rate between lowest grid point and z=h
    delta_T = dd.theta_mean.sel(z=dd.h, method="nearest") - dd.theta_mean[0]
    delta_z = dd.z.sel(z=dd.h, method="nearest") - dd.z[0]
    dd["dT_dz"] = delta_T / delta_z
    # calculate eddy turnover time TL
    if dd.lbc == 0:
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
    if dd.lbc == 0:
        # calculate TKE-based sbl depth
        # dd["he"] = dd.z.where(dd.e <= 0.05*dd.e[0], drop=True)[0]
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

    return dd
# ---------------------------------------------
def load_full(dnc, t0, t1, dt, delta_t, stats=None, rotate=False):
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
        s = load_stats(dnc+stats)
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
def timeseries2netcdf(dout, dnc, del_npz=False, **params):
    """Load raw timeseries .npz files and process into single netcdf file with 
    dimensions (time, z). To be used *after* organize_output().
    -Input-
    dout: string, directory where timeseries npz files are stored
    dnc: new directory for saving combined netcdf file
    del_npz: bool, delete raw npz files after combining to save space
    params: dictionary with relevant sim information for processing
    """
    # grab relevent parameters
    print(f"Processing simulation {params['simlabel']}")
    u_scale = params["u_scale"]
    theta_scale = params["T_scale"]
    q_scale = params["q_scale"]
    Lz = params["Lz"]
    nz = params["nz"]
    dz = Lz/nz
    # define z array
    # u- and w-nodes are staggered
    # zw = 0:Lz:nz
    # zu = dz/2:Lz-dz/2:nz-1
    # interpolate w, txz, tyz, q3 to u grid
    zw = np.linspace(0., Lz, nz)
    zu = np.linspace(dz/2., Lz+dz/2., nz)
    # at this point, all timeseries file levels combined into one
    # npz file per variable, so load each one
    # u
    f01 = f"{dout}u_timeseries.npz"
    u_ts = np.load(f01)["arr_0"][:,1:] * u_scale
    # v
    f02 = f"{dout}v_timeseries.npz"
    v_ts = np.load(f02)["arr_0"][:,1:] * u_scale
    # w
    f03 = f"{dout}w_timeseries.npz"
    w_ts = np.load(f03)["arr_0"][:,1:] * u_scale
    # theta
    f04 = f"{dout}t_timeseries.npz"
    theta_ts = np.load(f04)["arr_0"][:,1:] * theta_scale
    # q
    f05 = f"{dout}q_timeseries.npz"
    q_ts = np.load(f05)["arr_0"][:,1:] * q_scale
    # thetav
    f06 = f"{dout}tv_timeseries.npz"
    thetav_ts = np.load(f06)["arr_0"][:,1:] * theta_scale
    # p
    f07 = f"{dout}p_timeseries.npz"
    p_ts = np.load(f07)["arr_0"][:,1:] # dont know appropriate scale yet
    # txx
    f08 = f"{dout}txx_timeseries.npz"
    txx_ts = np.load(f08)["arr_0"][:,1:] * u_scale * u_scale
    # txy
    f09 = f"{dout}txy_timeseries.npz"
    txy_ts = np.load(f09)["arr_0"][:,1:] * u_scale * u_scale
    # txz
    f10 = f"{dout}txz_timeseries.npz"
    txz_ts = np.load(f10)["arr_0"][:,1:] * u_scale * u_scale
    # tyy
    f11 = f"{dout}tyy_timeseries.npz"
    tyy_ts = np.load(f11)["arr_0"][:,1:] * u_scale * u_scale
    # tyz
    f12 = f"{dout}tyz_timeseries.npz"
    tyz_ts = np.load(f12)["arr_0"][:,1:] * u_scale * u_scale
    # tzz
    f13 = f"{dout}tzz_timeseries.npz"
    tzz_ts = np.load(f13)["arr_0"][:,1:] * u_scale * u_scale
    # sgs_t3
    f14 = f"{dout}sgs_t3_timeseries.npz"
    tw_sgs_ts = np.load(f14)["arr_0"][:,1:] * theta_scale * u_scale
    # sgs_q3
    f15 = f"{dout}sgs_q3_timeseries.npz"
    qw_sgs_ts = np.load(f15)["arr_0"][:,1:] * q_scale * u_scale
    # e_sgs
    f16 = f"{dout}e_sgs_timeseries.npz"
    e_sgs_ts = np.load(f16)["arr_0"][:,1:] * u_scale * u_scale
    # dissip
    f17 = f"{dout}dissip_timeseries.npz"
    dissip_ts = np.load(f17)["arr_0"][:,1:] * u_scale * u_scale * u_scale / Lz

    # determine time array from number of timesteps in these files
    # load all timestamps because we have control over output in sim now
    nt = u_ts.shape[0]
    time = np.arange(nt, dtype=np.float64) * params["dt"]
    nhr = int(nt * params["dt"] / 3600.)
    print(f"Timestsps in npz files: {nt} = final {nhr} hrs of sim")
    # update params to have this updated nhr value, delete nhr_s
    params["total_time"] = nhr
    if "nhr_s" in params.keys():
        del params["nhr_s"]

    # initialize Dataset for saving output
    ts_all = xr.Dataset(data_vars=None, coords=dict(t=time,z=zu), attrs=params)
    # add data on uvp nodes: u,v,theta,q,p
    ts_all["u"] = xr.DataArray(u_ts, dims=("t","z"), 
                               coords=dict(t=time, z=zu))
    ts_all["v"] = xr.DataArray(v_ts, dims=("t","z"), 
                               coords=dict(t=time, z=zu))
    ts_all["theta"] = xr.DataArray(theta_ts, dims=("t","z"), 
                                   coords=dict(t=time, z=zu))
    ts_all["thetav"] = xr.DataArray(thetav_ts, dims=("t","z"), 
                                    coords=dict(t=time, z=zu))
    ts_all["q"] = xr.DataArray(q_ts, dims=("t","z"), 
                               coords=dict(t=time, z=zu))
    ts_all["p"] = xr.DataArray(p_ts, dims=("t","z"), 
                               coords=dict(t=time, z=zu))
    # interpolate data on w nodes and add: w, t_ij, sgs_q3, sgs_t3, e_sgs, dissip
    ts_all["w"] =  xr.DataArray(w_ts, dims=("t","z"), coords=dict(t=time, z=zw)
                               ).interp(z=zu, method="linear", 
                                        kwargs={"fill_value": "extrapolate"})
    ts_all["txx"] =  xr.DataArray(txx_ts, dims=("t","z"), coords=dict(t=time, z=zw)
                                 ).interp(z=zu, method="linear", 
                                          kwargs={"fill_value": "extrapolate"})
    ts_all["txy"] =  xr.DataArray(txy_ts, dims=("t","z"), coords=dict(t=time, z=zw)
                                 ).interp(z=zu, method="linear", 
                                          kwargs={"fill_value": "extrapolate"})
    ts_all["txz"] =  xr.DataArray(txz_ts, dims=("t","z"), coords=dict(t=time, z=zw)
                                 ).interp(z=zu, method="linear", 
                                          kwargs={"fill_value": "extrapolate"})
    ts_all["tyy"] =  xr.DataArray(tyy_ts, dims=("t","z"), coords=dict(t=time, z=zw)
                                 ).interp(z=zu, method="linear", 
                                          kwargs={"fill_value": "extrapolate"})
    ts_all["tyz"] =  xr.DataArray(tyz_ts, dims=("t","z"), coords=dict(t=time, z=zw)
                                 ).interp(z=zu, method="linear", 
                                          kwargs={"fill_value": "extrapolate"})
    ts_all["tzz"] =  xr.DataArray(tzz_ts, dims=("t","z"), coords=dict(t=time, z=zw)
                                 ).interp(z=zu, method="linear", 
                                          kwargs={"fill_value": "extrapolate"})
    ts_all["tw_sgs"] =  xr.DataArray(tw_sgs_ts, dims=("t","z"), coords=dict(t=time, z=zw)
                                    ).interp(z=zu, method="linear", 
                                             kwargs={"fill_value": "extrapolate"})
    ts_all["qw_sgs"] =  xr.DataArray(qw_sgs_ts, dims=("t","z"), coords=dict(t=time, z=zw)
                                    ).interp(z=zu, method="linear", 
                                             kwargs={"fill_value": "extrapolate"})
    ts_all["e_sgs"] =  xr.DataArray(e_sgs_ts, dims=("t","z"), coords=dict(t=time, z=zw)
                                   ).interp(z=zu, method="linear", 
                                            kwargs={"fill_value": "extrapolate"})
    ts_all["dissip"] =  xr.DataArray(dissip_ts, dims=("t","z"), coords=dict(t=time, z=zw)
                                    ).interp(z=zu, method="linear", 
                                             kwargs={"fill_value": "extrapolate"})
    # save to netcdf
    fsave_ts = f"{dnc}timeseries_all_{nhr}h.nc"
    with ProgressBar():
        ts_all.to_netcdf(fsave_ts, mode="w")
    
    # optionally delete files
    if del_npz:
        print("Cleaning up raw files...")
        os.system(f"rm {dout}*.npz")
        
    print(f"Finished saving timeseries for simulation {params['simlabel']}")

    return
# ---------------------------------------------
def load_timeseries(dnc, detrend=True, tlab="1.0h", tuse=None):
    """Reading function for timeseries files created from timseries2netcdf()
    Load netcdf files using xarray and calculate numerous relevant parameters
    :param str dnc: path to netcdf directory for simulation
    :param bool detrend: detrend timeseries for calculating variances, default=True
    :param str tavg: select which timeseries file to use in hours, default="1.0h"
    :param float tuse: how many hours from the end to use, default=None
    :return d: xarray dataset
    """
    # load timeseries data
    fts = f"timeseries_all_{tlab}.nc"
    print(f"Reading file: {fts}")
    d = xr.open_dataset(dnc+fts)
    # calculate means
    varlist = ["u", "v", "w", "theta", "q", "thetav"]
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
        qd = xrft.detrend(d.q, dim="t", detrend_type="linear")
        # store these detrended variables
        d["ud"] = ud
        d["udr"] = udr
        d["vd"] = vd
        d["vdr"] = vdr
        d["wd"] = wd
        d["td"] = td
        d["qd"] = qd
        # now calculate vars
        d["uu"] = ud * ud
        d["uur"] = udr * udr
        d["vv"] = vd * vd
        d["vvr"] = vdr * vdr
        d["ww"] = wd * wd
        d["tt"] = td * td
        d["qq"] = qd * qd
        # calculate "inst" covars
        d["uw"] = (ud * wd) + d.txz
        d["vw"] = (vd * wd) + d.tyz
        d["tw"] = (td * wd) + d.tw_sgs
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
        d["qq"] = (d.q - d.q_mean) * (d.q - d.q_mean)
        # calculate "inst" covars
        d["uw"] = (d.u - d.u_mean) * (d.w - d.w_mean) + d.txz
        d["vw"] = (d.v - d.v_mean) * (d.w - d.w_mean) + d.tyz
        d["tw"] = (d.theta - d.theta_mean) * (d.w - d.w_mean) + d.tw_sgs
        d["qw"] = (d.q - d.q_mean) * (d.w - d.w_mean) + d.qw_sgs
        d["tvw"] = d.tw + 0.61*(d.theta - d.theta_mean)*d.qw/1000.
        d["tq"] = (d.q - d.q_mean) * (d.theta - d.theta_mean)
    
    # return desired portion of timeseries files
    if tuse is None:
        return d
    else:
        # calculate number of points per hour
        pph = d.t.size / d.total_time
        # now can get starting time by multiplying tuse by pph
        jtuse = -1 * int(tuse * pph)
        # return just this range
        return d.isel(t=range(jtuse, 0))

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
    vuse = ["u","v","w","theta","q","p",
            "txx","txy","txz","tyy","tyz","tzz",
            "tw_sgs","qw_sgs","e_sgs","dissip"]
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
# ---------------------------------------------
def read_checkpoint_binary(fcheck, nx, ny, nz, lbc=0):
    """Reads checkpoint_XXXXXXX.out file produced by LES code in inst_field
    subroutine of io.f90. Only works for lbc=0 as of now (i.e., expects
    T_sfc_init and q_sfc_init to be inside of checkpoint file). Returns list
    of numpy arrays.
    -Input-
    fcheck: path to checkpoint file
    nx: number of grid points in x-dimension (will be converted to ld)
    ny: number of grid points in y-dimension (used as-is)
    nz: number of grid points in z-dimension (will be converted to nz_tot=nz+1)
    lbc: lower boundary condition flag, default=0 (do not change)
    -Returns-
    check: list of numpy nd arrays
    """
    # compute relevant dimensions expected from output
    ld = int(2 * ((nx//2)+1))
    nz_tot = nz + 1
    # open binary file
    print(f"Reading file: {fcheck}")
    with open(fcheck, "rb") as ff:
        # jt_total, integer
        jt_total = np.fromfile(ff, dtype=np.int32, count=1)
        # u_tot(ld,ny,nz_tot)
        u_tot = np.fromfile(ff, dtype=np.float64, count=ld*ny*nz_tot)
        # v_tot(ld,ny,nz_tot)
        v_tot = np.fromfile(ff, dtype=np.float64, count=ld*ny*nz_tot)
        # w_tot(ld,ny,nz_tot)
        w_tot = np.fromfile(ff, dtype=np.float64, count=ld*ny*nz_tot)
        # theta_tot(ld,ny,nz_tot)
        theta_tot = np.fromfile(ff, dtype=np.float64, count=ld*ny*nz_tot)
        # q_tot(ld,ny,nz_tot)
        q_tot = np.fromfile(ff, dtype=np.float64, count=ld*ny*nz_tot)
        # RHSx_tot(ld,ny,nz_tot)
        RHSx_tot = np.fromfile(ff, dtype=np.float64, count=ld*ny*nz_tot)
        # RHSy_tot(ld,ny,nz_tot)
        RHSy_tot = np.fromfile(ff, dtype=np.float64, count=ld*ny*nz_tot)
        # RHSz_tot(ld,ny,nz_tot)
        RHSz_tot = np.fromfile(ff, dtype=np.float64, count=ld*ny*nz_tot)
        # RHS_T_tot(ld,ny,nz_tot)
        RHS_T_tot = np.fromfile(ff, dtype=np.float64, count=ld*ny*nz_tot)
        # sgs_t3(ld,ny,1)
        sgs_t3_tot = np.fromfile(ff, dtype=np.float64, count=ld*ny)
        # psi_m(nx,ny)
        psi_m_tot = np.fromfile(ff, dtype=np.float64, count=nx*ny)
        # Cs_opt2_tot(ld,ny,nz_tot)
        Cs_opt2_tot = np.fromfile(ff, dtype=np.float64, count=ld*ny*nz_tot)
        # F_LM_tot(ld,ny,nz_tot)
        F_LM_tot = np.fromfile(ff, dtype=np.float64, count=ld*ny*nz_tot)
        # F_MM_tot(ld,ny,nz_tot)
        F_MM_tot = np.fromfile(ff, dtype=np.float64, count=ld*ny*nz_tot)
        # F_QN_tot(ld,ny,nz_tot)
        F_QN_tot = np.fromfile(ff, dtype=np.float64, count=ld*ny*nz_tot)
        # F_NN_tot(ld,ny,nz_tot)
        F_NN_tot = np.fromfile(ff, dtype=np.float64, count=ld*ny*nz_tot)
        # T_s(nx,ny)
        T_s_tot = np.fromfile(ff, dtype=np.float64, count=nx*ny)
        # q_s(nx,ny)
        q_s_tot = np.fromfile(ff, dtype=np.float64, count=nx*ny)
        # RHS_q_tot(ld,ny,nz_tot)
        RHS_q_tot = np.fromfile(ff, dtype=np.float64, count=ld*ny*nz_tot)
        # sgs_q3(ld,ny,1)
        sgs_q3_tot = np.fromfile(ff, dtype=np.float64, count=ld*ny)

    # finished reading
    print("Reshaping arrays")
    # reshape each variable into 3d/2d, then only return nonzero values
    u = u_tot.reshape((ld,ny,nz_tot), order="F")[:nx,:,:nz]
    v = v_tot.reshape((ld,ny,nz_tot), order="F")[:nx,:,:nz]
    w = w_tot.reshape((ld,ny,nz_tot), order="F")[:nx,:,:nz]
    theta = theta_tot.reshape((ld,ny,nz_tot), order="F")[:nx,:,:nz]
    q = q_tot.reshape((ld,ny,nz_tot), order="F")[:nx,:,:nz]
    RHSx = RHSx_tot.reshape((ld,ny,nz_tot), order="F")[:nx,:,:nz]
    RHSy = RHSy_tot.reshape((ld,ny,nz_tot), order="F")[:nx,:,:nz]
    RHSz = RHSz_tot.reshape((ld,ny,nz_tot), order="F")[:nx,:,:nz]
    RHS_T = RHS_T_tot.reshape((ld,ny,nz_tot), order="F")[:nx,:,:nz]
    sgs_t3 = sgs_t3_tot.reshape((ld,ny), order="F")[:nx,:]
    psi_m = psi_m_tot.reshape((nx,ny), order="F")
    Cs_opt2 = Cs_opt2_tot.reshape((ld,ny,nz_tot), order="F")[:nx,:,:nz]
    F_LM = F_LM_tot.reshape((ld,ny,nz_tot), order="F")[:nx,:,:nz]
    F_MM = F_MM_tot.reshape((ld,ny,nz_tot), order="F")[:nx,:,:nz]
    F_QN = F_QN_tot.reshape((ld,ny,nz_tot), order="F")[:nx,:,:nz]
    F_NN = F_NN_tot.reshape((ld,ny,nz_tot), order="F")[:nx,:,:nz]
    T_s = T_s_tot.reshape((nx,ny), order="F")
    q_s = q_s_tot.reshape((nx,ny), order="F")
    RHS_q = RHS_q_tot.reshape((ld,ny,nz_tot), order="F")[:nx,:,:nz]
    sgs_q3 = sgs_q3_tot.reshape((ld,ny), order="F")[:nx,:]
    print("Finished reading and reshaping! Returning...")
    # return list of each of these variables
    return [jt_total, u, v, w, theta, q, RHSx, RHSy, RHSz, RHS_T, sgs_t3, 
            psi_m, Cs_opt2, F_LM, F_MM, F_QN, F_NN, T_s, q_s,RHS_q,sgs_q3]
# ---------------------------------------------
def interp_2d_spec_np(a, nx, ny, nz, nf):
    """Compute two-dimensional spectral interpolation on a numpy array 
    by performing the following steps:
    1) compute forward 2d fft and normalize by original (nx*ny)
    2) shift zero-frequency to center of array and zero-pad
    3) shift back to have zero-frequency in top left of array
    4) compute inverse 2d fft, do not normalize by new (nx_new*ny_new)
    5) return real values
    -Input-
    nx: number of grid points in x-dimension
    ny: number of grid points in y-dimension
    nz: number of grid points in z-dimension (can be zero)
    nf: factor by which to increase number of points in both x & y dim
    -Returns-
    interpolated array a_interp(nx*nf, ny*nf, nz)
    """
    # compute forward fft
    # normalize by nx*ny with forward
    f_a = fft2(a, axes=(0,1), norm="forward")
    # shift frequencies to prepare for zero-padding
    f_a_shift = fftshift(f_a, axes=(0,1))
    # compute number of zeros to pad to beginning and end of x,y
    npadx = (nx//2)*(nf-1)
    npady = (ny//2)*(nf-1)
    # check if only passed a 2d array to begin
    if nz == 0:
        f_a_shift_pad = np.pad(f_a_shift, pad_width=((npadx, npadx), 
                                                     (npady, npady)))
    else:
        # only pad in x,y
        f_a_shift_pad = np.pad(f_a_shift, pad_width=((npadx, npadx), 
                                                     (npady, npady), 
                                                     (0, 0)))
    # shift zero frequencies back to start
    f_a_pad = ifftshift(f_a_shift_pad, axes=(0,1))
    # ifft to return interpolated array
    a_interp = ifft2(f_a_pad, axes=(0,1), norm="forward")
    # return real values
    return np.real(a_interp)
# ---------------------------------------------
def interp_2d_np(a, nx, ny, nf, Lx, Ly, method="linear"):
    """Compute two-dimensional interpolation on a numpy array by in
    physical space using the scipy.interpolate.RegularGridInterpolater
    module
    -Input-
    nx: number of grid points in x-dimension
    ny: number of grid points in y-dimension
    nf: factor by which to increase number of points in both x & y dim
    Lx: size of domain in x-dimension
    Ly: size of domain in y-dimension
    method: interpolation method to pass to RegularGridInterpolater,
        default="linear"
    -Returns-
    interpolated array a_interp(nx*nf, ny*nf, nz)
    """
    # construct original x-y grid
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    # xg, yg = np.meshgrid(x, y, indexing="ij", sparse=True)
    # initialize interpolater
    interp = RegularGridInterpolator(points=((x,y)), values=a,
                                     method=method, fill_value=None)
    # construct new x-y grid
    xnew = np.linspace(0, Lx, nf*nx)
    ynew = np.linspace(0, Ly, nf*ny)
    Xg, Yg = np.meshgrid(xnew, ynew, indexing="ij")
    # perform interpolation and return
    return interp((Xg,Yg))
# ---------------------------------------------
def interp_checkpoint_2d(fcheck, fdir_save, nx, ny, nz, nf, Lx, Ly, spec=True):
    """Interpolate a simulation checkpoint file in the horizontal (x,y) using
    spectral or physical interpolation. Call read_checkpoint_binary and 
    interp_2d_spec_np/interp_2d_np (NOT interpolate_spec_2d!) to handle 
    pure np arrays.
    -Input-
    fcheck: path to checkpoint file
    fdir_save: directory to save new file (will get new filename from fcheck)
    nx: number of grid points in x-dimension
    ny: number of grid points in y-dimension
    nz: number of grid points in z-dimension
    nf: factor by which to increase number of points in both x & y dim
    Lx: size of domain in x-dimension
    Ly: size of domain in y-dimension
    spec: flag to use spectral or physical interpolation, default=True
    -Output-
    binary file with same structure as fcheck but with interpolated fields
    """
    # first, read the binary checkpoint file
    checkpoint = read_checkpoint_binary(fcheck, nx, ny, nz)
    # initialize new list to hold interpolated fields
    checkpoint_interp = []
    # compute dimensions for later
    nx_new = nx*nf
    ny_new = ny*nf
    ld = int(2 * ((nx_new//2)+1))
    nz_tot = nz + 1
    # loop over each field in checkpoint (not including jt_total) and interp
    for jx, x in enumerate(checkpoint):
        # dont do anything with jt_total, just store and move on
        if jx == 0:
            checkpoint_interp.append(x)
        else:
            # compute size of x to feed to interp_2d_spec_np
            nn = x.shape
            nx_x, ny_x = nn[0], nn[1]
            # check how many dimensions in x
            nd = x.ndim
            # if nd is 2, then nz_x should be 0; otherwise, can index
            if nd == 2:
                nz_x = 0
            else:
                nz_x = nn[2]
            # select interpolation method
            if spec:
                # call interp_2d_spec_np()
                print(f"Calling interp_2d_spec_np for field {jx}, shape={x.shape}")
                x_interp = interp_2d_spec_np(x, nx_x, ny_x, nz_x, nf)
            else:
                # call interp_2d_np()
                print(f"Calling interp_2d_np for field {jx}, shape={x.shape}")
                x_interp = interp_2d_np(x, nx_x, ny_x, nf, Lx, Ly, "cubic")
            print(f"Finished interpolating! New shape={x_interp.shape}")
            # store in larger arrays
            # TODO: convert list to dictionary with variable names
            # for now, will hard code with jx need which shape
            if jx in [1,2,3,4,5,6,7,8,9,12,13,14,15,16,19]:
                # these go in arrays size(ld,ny,nz_tot)
                x_new = np.zeros((ld,ny_new,nz_tot), dtype=np.float64)
                x_new[:nx_new,:,:nz] = x_interp
            elif jx in [10,20]:
                # these are 2d and go in arrays size(ld,ny)
                x_new = np.zeros((ld,ny_new), dtype=np.float64)
                x_new[:nx_new,:] = x_interp
            elif jx in [11,17,18]:
                # these get to stay size(nx,ny)
                x_new = x_interp
            else:
                print(f"How did you get here? jx={jx}")

            # store padded arrays
            checkpoint_interp.append(x_new)

    # finished interpolating all variables
    # want save interpolated file in fdir_save
    # parse fcheck to get new name: fdir_save/checkpoint_xxxxxxx_interp.out
    fsave = os.path.join(fdir_save, 
                         f"{fcheck.split('/')[-1].split('.')[0]}_interp.out")
    print(f"Saving file: {fsave}")
    with open(fsave, "wb") as ff:
        # loop over each variable and convert to binary
        for c in checkpoint_interp:
            ff.write(c.tobytes("F"))
    print("Finished writing interpolated file.")
    return
# ---------------------------------------------
def calc_TKE_budget(dnc, fall, s):
    """Calculate TKE budget terms by looping over files in fall.
    Save a single netcdf file in dnc.
    -Input-
    dnc: string, absolute path to netcdf directory for saving new file
    fall: list of strings of all the files to be loaded one-by-one
    s: xr.Dataset, stats file corresponding to simulation for attrs
    """
    # construct Dataset to save
    dsave = xr.Dataset(data_vars=None, attrs=s.attrs)
    # get number of files
    nf = len(fall)
    # start by computing components that do not require full 3d fields
    # P: shear production
    P = (-1 * s.u_mean.differentiate("z",2)*s.uw_cov_tot) +\
        (-1 * s.v_mean.differentiate("z",2)*s.vw_cov_tot)
    dsave["P"] = P
    # B: buoyancy production
    g = 9.81
    beta = g/s.thetav_mean[0]
    B = beta*s.tvw_cov_tot
    dsave["B"] = B
    # partition into production by theta versus q
    Bt = beta*s.tw_cov_tot
    Bq = 0.61*g*s.qw_cov_tot/1000.
    dsave["Bt"] = Bt
    dsave["Bq"] = Bq
    # D: dissipation
    dsave["D"] = s.dissip_mean

    # compute turbulent transport by looping over volume files
    # begin looping over each file and load one by one
    for jf, ff in enumerate(fall):
        # load file
        print(f"Loading file: {ff}")
        d = xr.load_dataset(ff)
        # compute velocity fluctuations
        up = d.u - d.u.mean(dim=("x","y"))
        vp = d.v - d.v.mean(dim=("x","y"))
        wp = d.w - d.w.mean(dim=("x","y"))
        # compute "inst" 3rd-order moments uj'uj'u3'
        uuw = up * up * wp
        vvw = vp * vp * wp
        www = wp * wp * wp
        # add and take mean to get <e'w'>
        ew = 0.5 * (uuw + vvw + www).mean(dim=("x","y"))
        # calc gradient to get T: turbulent transport
        T = -1 * ew.differentiate("z", 2).compute()
        # store depending on whether this is first iteration
        if jf == 0:
            dsave["T"] = T
        else:
            dsave["T"] += T
    # divide dsave by nf to get mean
    dsave["T"] /= float(nf)
    # finally, calculate residual term R
    dsave["R"] = -1 * (dsave.P + dsave.B + dsave.D + dsave.T)

    # save file
    fsave = f"{dnc}TKE_budget.nc"
    # delete old file for saving new one
    if os.path.exists(fsave):
        os.system(f"rm {fsave}")
    print(f"Saving file: {fsave}")
    with ProgressBar():
        dsave.to_netcdf(fsave, mode="w")
    
    print("Finished computing TKE budget!")
    return