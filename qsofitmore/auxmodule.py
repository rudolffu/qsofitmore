#!/usr/bin/env python
from math import log
from os import stat
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat
from uncertainties.umath import *


def broken_pl_model(wave, a1, a2, b):
    w_blue = wave[wave<4661]
    w_red = wave[wave>=4661]
    if len(w_blue)>0 and len(w_red)>0:
        w_break = wave[wave<4661][-1]
        f_blue = b*(w_blue/3.0e3)**a1
        f_red = f_blue[-1]*(w_red/w_break)**a2
        f_all = np.concatenate((f_blue,f_red), axis=None)
    elif len(w_blue)>0 and len(w_red)==0:
        f_all = b*(wave/3.0e3)**a1
    return f_all

# Return LaTeX name for a line / complex name
def texlinename(name) -> str:
    if name == 'Ha':
        tname = r'H$\alpha$'
    elif name == 'Hb':
        tname = r'H$\beta$'
    elif name == 'Hr':
        tname = r'H$\gamma$'
    elif name == 'Hg':
        tname = r'H$\gamma$'
    elif name == 'Lya':
        tname = r'Ly$\alpha$'
    else:
        tname = name
    return tname


def designation(ra, dec, telescope=None) -> str:
    c = SkyCoord(ra=ra*u.degree, 
                 dec=dec*u.degree,frame='icrs')
    srahms = c.ra.to_string(unit=u.hourangle, sep='', precision=2, pad=True)
    sdecdms = c.dec.to_string(sep='', precision=1, alwayssign=True, pad=True)
    if telescope is not None:
        newname = 'J'+srahms+sdecdms+'_'+str(telescope)
    else:
        newname = 'J'+srahms+sdecdms
    return newname


def sciplotstyle():
    SMALL_SIZE = 16
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 20
    plt.rc('font', size=SMALL_SIZE, family='serif')    # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.serif'] = ['Times New Roman']


def Flux2L(flux, z):
    """Transfer flux to luminoity assuming a flat Universe"""
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    DL = cosmo.luminosity_distance(z).value*10**6*3.08*10**18  # unit cm
    L = flux*1.e-17*4.*np.pi*DL**2  # erg/s/A
    return L


def mbh_hb(fwhm, fwhm_err, L5100, L5100_err):
    if fwhm * fwhm_err * L5100 * L5100_err != 0:
        ufwhm = ufloat(fwhm, fwhm_err)
        uL5100 = 10 ** ufloat(L5100, L5100_err)
        logmbh = log10(ufwhm**2 * (uL5100*1e-44)**0.5)+0.91
    else:
        logmbh = ufloat(np.nan, np.nan)
    return logmbh.n, logmbh.s


mbh_hb_df = lambda x: mbh_hb(
    x['Hb_whole_br_fwhm'],
    x['Hb_whole_br_fwhm_err'],
    x['L5100'],
    x['L5100_err'])


def mbh_mgii(fwhm, fwhm_err, L3000, L3000_err):
    if fwhm * fwhm_err * L3000 * L3000_err != 0:
        ufwhm = ufloat(fwhm, fwhm_err)
        uL3000 = 10 ** ufloat(L3000, L3000_err)
        logmbh = log10(ufwhm**1.51 * (uL3000*1e-44)**0.5)+2.60
    else:
        logmbh = ufloat(np.nan, np.nan)
    return logmbh.n, logmbh.s


mbh_mgii_df = lambda x: mbh_mgii(
    x['MgII_whole_br_fwhm'],
    x['MgII_whole_br_fwhm_err'],
    x['L3000'],
    x['L3000_err'])


def mbh_civ(fwhm, fwhm_err, L1350, L1350_err):
    if fwhm * fwhm_err * L1350 * L1350_err != 0:
        ufwhm = ufloat(fwhm, fwhm_err)
        uL1350 = 10 ** ufloat(L1350, L1350_err)
        logmbh = log10(ufwhm**2 * (uL1350*1e-44)**0.53)+0.66
    else:
        logmbh = ufloat(np.nan, np.nan)
    return logmbh.n, logmbh.s


mbh_civ_df = lambda x: mbh_civ(
    x['CIV_whole_br_fwhm'],
    x['CIV_whole_br_fwhm_err'],
    x['L1350'],
    x['L1350_err'])


def check_wings(na_dict):
    wings = ['OIII4959w', 'OIII5007w']
    lines = ['OIII4959', 'OIII5007']
    wing_status = []
    for i in range(len(wings)):
        wing = wings[i]
        line = lines[i]
        if wing in na_dict.keys():
            if na_dict[wing]['ew'].size == 0:
                status_temp = False
            elif na_dict[wing]['ew']/na_dict[line]['ew']>2:
                status_temp = False
            else:
                status_temp = True
        else:
            status_temp = None
        wing_status.append(status_temp)
    return wing_status