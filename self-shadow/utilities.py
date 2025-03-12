#######################################
# imports
import numpy as np
import pandas as pd
import sys
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob
import subprocess
import time
#import emcee

from multiprocessing import Pool
from functools import partial

from tqdm import tqdm

from astropy.io import fits
from astropy import units as u
from astropy import constants as c

from importlib import reload

from scipy.interpolate import interp1d
from scipy.ndimage import rotate
from scipy.optimize import curve_fit
#######################################
#constants and couple useful dicts
datadir = "/home/dkacan/research/radmc3d-2.0-master/RIPPED/data/"

plabels = {"Fnu":r"$F_\nu$ (mJy)",
           "Fnu_micro":r"$F_\nu$ ($\mu$Jy)",
           "lam":"$\lambda$ ($\mu$m)",
           "wav":"$\lambda$ ($\mu$m)",
           "tau":r"Optical depth ($\tau$)",
           "r_au":"$r$ (au)",
           "z_au":"$z$ (au)",
           "peaktau":"Peak optical depth"}

ipm = pd.DataFrame(columns = ['print_name','fitlim','featurelim','fitorder','pcolor'])
ipm.loc['h2o'] = ["H$_2$O (3.0 $\\mu$m)" ,[(.6,2.8),(3.7,4)]       ,(.6,4)     ,2,'darkblue']
ipm.loc['co2'] = ["CO$_2$ (4.27 $\\mu$m)",[(4,4.2),(4.3,4.5)]      ,(4,4.5)    ,2,'darkred']
ipm.loc['co']  = ["CO (4.67 $\\mu$m)"    ,[(4.55,4.66),(4.69,4.72)],(4.55,4.72),2,'darkorange']

def wfits(im, fname,he=None):
    """wfits - write im to file fname, automatically overwriting any old file"""

    #if header is a dictionary, assumes that you want to make a header with those values
    # easiest way to do that is by saving first without header, reading it in again,
    #change header and than save correctly.
    if isinstance(he,dict):
        fits.writeto(fname,im,header=None,overwrite=True)
        _,basicheader = fits.getdata(fname,header=True)
        for key,value in he.items():
            basicheader[key] = value
        he = basicheader

    fits.writeto(fname,im,he,overwrite=True)

def deredden(wav, data, Av_obj, redden = False, unc_b = False):
    if Av_obj <= 3:
        A_to_Av = 3.55
        wave_law, ext_law = np.loadtxt(os.path.join(datadir,'Mathis_CCM89_ext.txt'),skiprows=8).T
    elif Av_obj < 8:
        A_to_Av = 7.75
        wave_law, ext_law = np.loadtxt(os.path.join(datadir,'extinc3_8.dat'),usecols=[0,1]).T
    else:
        A_to_Av = 7.75
        wave_law, ext_law = np.loadtxt(os.path.join(datadir,'extinc3_8.dat'),usecols=[0,2]).T
    
    if (len(data.shape) > 1)*unc_b:
        unc = data[1]
        flux = data[0]
    else:
        flux = data

    ext_law_interp = interp1d(wave_law, ext_law, fill_value = "extrapolate")(wav)
    A_lambda = ext_law_interp * (Av_obj/A_to_Av)
    fac = 10**(0.4*A_lambda)

    if redden:
        fac = 1/fac
    
    mask = (wav < 40)
    flux_out = flux.copy()
    flux_out[mask] = (flux_out[mask].T * fac[mask]).T
    
    if (len(data.shape) > 1) * unc_b:
        unc_out = flux_out * np.sqrt((unc/flux)**2)
        unc_out[mask] = unc[mask]
        return [flux_out,unc_out]
    else:
        return flux_out

@u.quantity_input
def mass_from_flux(freq:u.GHz,flux:u.mJy,d:u.pc):
    """
    Mass derived from continuum flux using the standard formula 
    and values for kappa.

    input:
    freq: frequency (with unit)
    flux: flux at frequency freq (with units)
    d:    distance ()
    """

    T = 20*u.K
    kappa = 10*freq/(1000*u.GHz)*u.cm**2*u.g**-1  

    def planck_f(freq, T):
        a = 2.0*c.h*(freq**3)
        b = c.h*freq/(c.k_B*T)
        intensity =  a/( (c.c**2 * (np.exp(b) - 1.0) ))
        return intensity
    
    mass = ((flux*(d)**2)/(kappa*planck_f(freq,T))).to(u.Msun)
    
    for i in range(len(np.atleast_1d(mass))):
        print(f'mass = {np.atleast_1d(mass)[i]:.1e}')
    return mass

def rebin_spec(wav,data,rebinfac = 10):
    """
    Input:
    wav:      wavelength of spectrum
    data:     spectrum in Jy/mJy
    rebinfac: factor to rebin the spectrum using a tophat function
              Note that this assumes a smooth R to work properly
    
    Returns:
    rebinned wav and data
    """
    try:
        wavout = np.nanmedian(wav[:-(len(wav)%rebinfac)].reshape((len(wav)//rebinfac,rebinfac)),axis = -1)
        dataout = np.nanmedian(data[:-(len(data)%rebinfac)].reshape((len(data)//rebinfac,rebinfac)),axis = -1)
    except:
        wavout = np.nanmedian(wav.reshape((int(len(wav)/rebinfac),rebinfac)),axis = -1)
        dataout = np.nanmedian(data.reshape((int(len(data)/rebinfac),rebinfac)),axis = -1)
    return wavout,dataout

def rebin2d(array, new_shape, avg = False):    
    def get_row_compressor(old_dimension, new_dimension, avg = False):
        dim_compressor = np.zeros((new_dimension, old_dimension))
        bin_size = float(old_dimension) / new_dimension
        next_bin_break = bin_size
        which_row = 0
        which_column = 0
        while which_row < dim_compressor.shape[0] and which_column < dim_compressor.shape[1]:
            if round(next_bin_break - which_column, 10) >= 1:
                dim_compressor[which_row, which_column] = 1
                which_column += 1
            elif next_bin_break == which_column:
                which_row += 1
                next_bin_break += bin_size
            else:
                partial_credit = next_bin_break - which_column
                dim_compressor[which_row, which_column] = partial_credit
                which_row += 1
                dim_compressor[which_row, which_column] = 1 - partial_credit
                which_column += 1
                next_bin_break += bin_size
        if avg:
            dim_compressor /= bin_size
        return dim_compressor
    
    def get_column_compressor(old_dimension, new_dimension, avg):
        return get_row_compressor(old_dimension, new_dimension, avg).transpose()

    # Note: new shape should be smaller in both dimensions than old shape
    if isinstance(new_shape,int):
        array = np.atleast_2d(array)
        new_shape = (1,new_shape)
        
    if len(new_shape) == 3:
        return np.array([np.mat(get_row_compressor(array.shape[1], new_shape[1],avg)) * \
               np.mat(array[i]) *  np.mat(get_column_compressor(array.shape[2], new_shape[2],avg))
        for i in np.arange(len(array))])

    elif len(new_shape) == 2:
        return np.array(np.mat(get_row_compressor(array.shape[0], new_shape[0],avg)) * \
               np.mat(array) *  np.mat(get_column_compressor(array.shape[1], new_shape[1],avg)))

def getdf(name,burnin = 100,start = False):
    df = pd.read_csv(name,delim_whitespace=True)
    if start:
        df = df[df.iter == 0]
    else:
        df = df[df.iter > burnin]
    df = df.sort_values("lnprob").drop(columns = ['T','walker','iter','lnprob'])
    shp = df.shape
    df = df.drop_duplicates(keep = 'last')
    print(f'now: {df.shape} was: {shp} acceptance: {df.shape[0]/shp[0]}')
    labels = df.columns
    df = df.values
    return df,labels

def continuumfitting(lam,sed,range_limits, xmin, xmax, order=5,extrap = False,origwav = True):
    """
    Input:
    lam: wavelength of spectrum
    sed: spectrum in mJy
    range_limits: list of continuum ranges to consider in continuum fit
    xmin: min wavelength of output spectrum
    xmax: max wavelength of output spectrum
    order: order polynomial to fit
    extrap: If True, will extrapolate the fit if xmin < input wavelengths
    origwav: If True, gives back everything in original resolution. 
             makes a high resolution interpolation otherwise
    
    Returns:
    input wavelength
    continuum
    input data
    optical depth
    """

    #make mask to select points for continuum fit
    mask = np.zeros_like(lam)
    for limit in np.atleast_1d(range_limits):
        mask += (lam < limit[1]) * (lam > limit[0])
    mask = mask > 0
    mask *= (lam > xmin) * (lam < xmax)

    #fit continuum
    Fcont = np.poly1d(np.polyfit(lam[mask], sed[mask], order))
    
    try:
        Fdata = interp1d(lam,sed, kind='cubic',fill_value="extrapolate")
    except ValueError:
        Fdata = interp1d(lam,sed, kind='linear',fill_value="extrapolate")
        
    if origwav:
        tt = lam[(lam>=xmin)*(lam<=xmax)]
    else:
        if extrap:
            tt = np.linspace(xmin, xmax, 5000)
        else:
            tt = np.linspace(lam[mask].min(), lam[mask].max(), 5000)
    
    #convert ices to optical depth
    tau_poly = -1*np.log(Fdata(tt)/Fcont(tt))
    return tt,Fcont(tt),Fdata(tt),tau_poly

def rmask(im,r,xc=None,yc=None):
    rs = rinpix(im,xc=xc,yc=yc)
    return rs<r

def rinpix(im,xc=None,yc=None):
    ny,nx = np.shape(im)
    if not xc:
        xc = (nx-1)/2.
    if not yc:
        yc = (ny-1)/2.
    y,x = np.mgrid[:ny,:nx]
    r = np.sqrt((y-yc)**2+(x-xc)**2)
    return r

def wfits(im, fname,he=None):
    """wfits - write im to file fname, automatically overwriting any old file"""

    #if header is a dictionary, assumes that you want to make a header with those values
    # easiest way to do that is by saving first without header, reading it in again,
    #change header and than save correctly.
    if isinstance(he,dict):
        fits.writeto(fname,im,header=None,overwrite=True)
        _,basicheader = fits.getdata(fname,header=True)
        for key,value in he.items():
            basicheader[key] = value
        he = basicheader

    fits.writeto(fname,im,he,overwrite=True)

def multiprocess(func, args, Nthreads):
    """Function to run a function via multiprocessing.

    Parameters
    ----------
    func : function
        Function for which multiprocessing is used.
    args : unspecified
        The arguments needed to run the function.
    Nthreads : integer
        Number of threads to be used for multiprocessing.

    Returns
    -------
    """

    with Pool(processes=Nthreads) as p:
        res = p.map(partial(func), args)
    return list(res)
