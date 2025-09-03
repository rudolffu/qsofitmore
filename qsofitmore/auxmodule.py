#!/usr/bin/env python
from math import log
from os import stat
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat, umath


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


def flux_to_luminosity(flux, z):
    """
    Calculate luminosity using flux assuming a flat Universe
    
    Parameters
    ----------
    flux: float
        integrated flux in erg/s/cm^2, or monochromatic flux in erg/s/cm^2/A
    z: float
        redshift
    
    Returns
    -------
    L: float
        luminosity in erg/s, or monochromatic luminosity in erg/s/A
    """
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    D_L = cosmo.luminosity_distance(z).to('cm').value
    L = flux * 1e-17 * 4 * np.pi * D_L**2  
    # erg/s for integrated flux, erg/s/A for monochromatic flux
    return L


def mbh_ha_only(fwhm, LOGLHA, fwhm_err=0, LOGLHA_ERR=0):
    """
    Calculate black hole mass using Halpha line width and luminosity
        based on Greene & Ho 2005, ApJ, 630, 122
    
    Parameters:
    -----------
    fwhm : float
        H-alpha line FWHM in km/s
    LOGLHA : float
        Log10 of H-alpha luminosity
    fwhm_err : float, optional
        Uncertainty in FWHM (default: 0)
    LOGLHA_ERR : float, optional
        Uncertainty in log luminosity (default: 0)
    """
    if fwhm != 0 and LOGLHA != 0:
        if fwhm_err == 0 and LOGLHA_ERR == 0:
            # No uncertainties provided, return point estimate
            LHa = 10 ** LOGLHA
            mbh = 2e6 * (LHa/1e42)**0.55 * (fwhm/1000)**2.06
            logmbh_val = np.log10(mbh)
            return logmbh_val, 0.0
        else:
            # Calculate with uncertainties
            ufwhm = ufloat(fwhm, fwhm_err)
            uLHa = 10 ** ufloat(LOGLHA, LOGLHA_ERR)
            mbh = 2e6 * (uLHa/1e42)**0.55 * (ufwhm/1000)**2.06
            logmbh = umath.log10(mbh)
            return logmbh.n, logmbh.s
    else:
        if fwhm_err == 0 and LOGLHA_ERR == 0:
            return np.nan, 0.0
        else:
            logmbh = ufloat(np.nan, np.nan)
            return logmbh.n, logmbh.s


def mbh_hb(fwhm, L5100, fwhm_err=0, L5100_err=0):
    """
    Calculate black hole mass using H-beta line width and 5100A luminosity
    
    Parameters:
    -----------
    fwhm : float
        H-beta line FWHM in km/s
    L5100 : float
        Log10 of 5100A luminosity
    fwhm_err : float, optional
        Uncertainty in FWHM (default: 0)
    L5100_err : float, optional
        Uncertainty in log luminosity (default: 0)
    """
    if fwhm != 0 and L5100 != 0:
        if fwhm_err == 0 and L5100_err == 0:
            # No uncertainties provided, return point estimate
            L5100_linear = 10 ** L5100
            logmbh_val = np.log10(fwhm**2 * (L5100_linear*1e-44)**0.5) + 0.91
            return logmbh_val, 0.0
        else:
            # Calculate with uncertainties
            ufwhm = ufloat(fwhm, fwhm_err)
            uL5100 = 10 ** ufloat(L5100, L5100_err)
            logmbh = umath.log10(ufwhm**2 * (uL5100*1e-44)**0.5)+0.91
            return logmbh.n, logmbh.s
    else:
        if fwhm_err == 0 and L5100_err == 0:
            return np.nan, 0.0
        else:
            logmbh = ufloat(np.nan, np.nan)
            return logmbh.n, logmbh.s


mbh_hb_df = lambda x: mbh_hb(
    x['Hb_whole_br_fwhm'],
    x['L5100'],
    x['Hb_whole_br_fwhm_err'],
    x['L5100_err'])


def mbh_mgii(fwhm, L3000, fwhm_err=0, L3000_err=0):
    """
    Calculate black hole mass using MgII line width and 3000A luminosity
    
    Parameters:
    -----------
    fwhm : float
        MgII line FWHM in km/s
    L3000 : float
        Log10 of 3000A luminosity
    fwhm_err : float, optional
        Uncertainty in FWHM (default: 0)
    L3000_err : float, optional
        Uncertainty in log luminosity (default: 0)
    """
    if fwhm != 0 and L3000 != 0:
        if fwhm_err == 0 and L3000_err == 0:
            # No uncertainties provided, return point estimate
            L3000_linear = 10 ** L3000
            logmbh_val = np.log10(fwhm**1.51 * (L3000_linear*1e-44)**0.5) + 2.60
            return logmbh_val, 0.0
        else:
            # Calculate with uncertainties
            ufwhm = ufloat(fwhm, fwhm_err)
            uL3000 = 10 ** ufloat(L3000, L3000_err)
            logmbh = umath.log10(ufwhm**1.51 * (uL3000*1e-44)**0.5)+2.60
            return logmbh.n, logmbh.s
    else:
        if fwhm_err == 0 and L3000_err == 0:
            return np.nan, 0.0
        else:
            logmbh = ufloat(np.nan, np.nan)
            return logmbh.n, logmbh.s


mbh_mgii_df = lambda x: mbh_mgii(
    x['MgII_whole_br_fwhm'],
    x['L3000'],
    x['MgII_whole_br_fwhm_err'],
    x['L3000_err'])


def mbh_civ(fwhm, L1350, fwhm_err=0, L1350_err=0):
    """
    Calculate black hole mass using CIV line width and 1350A luminosity
    
    Parameters:
    -----------
    fwhm : float
        CIV line FWHM in km/s
    L1350 : float
        Log10 of 1350A luminosity
    fwhm_err : float, optional
        Uncertainty in FWHM (default: 0)
    L1350_err : float, optional
        Uncertainty in log luminosity (default: 0)
    """
    if fwhm != 0 and L1350 != 0:
        if fwhm_err == 0 and L1350_err == 0:
            # No uncertainties provided, return point estimate
            L1350_linear = 10 ** L1350
            logmbh_val = np.log10(fwhm**2 * (L1350_linear*1e-44)**0.53) + 0.66
            return logmbh_val, 0.0
        else:
            # Calculate with uncertainties
            ufwhm = ufloat(fwhm, fwhm_err)
            uL1350 = 10 ** ufloat(L1350, L1350_err)
            logmbh = umath.log10(ufwhm**2 * (uL1350*1e-44)**0.53)+0.66
            return logmbh.n, logmbh.s
    else:
        if fwhm_err == 0 and L1350_err == 0:
            return np.nan, 0.0
        else:
            logmbh = ufloat(np.nan, np.nan)
            return logmbh.n, logmbh.s


mbh_civ_df = lambda x: mbh_civ(
    x['CIV_whole_br_fwhm'],
    x['L1350'],
    x['CIV_whole_br_fwhm_err'],
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
            # elif na_dict[wing]['ew']/na_dict[line]['ew']>3:
            #     status_temp = False
            else:
                status_temp = True
        else:
            status_temp = None
        wing_status.append(status_temp)
    return wing_status