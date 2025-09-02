#!/usr/bin/env python
from astropy.io import fits
import numpy as np
import re
# import matplotlib.pyplot as plt
import os
# import importlib
from astropy import units as u
from astropy.coordinates import SkyCoord


def wang2019(wave, ebv, waveunit=u.AA, Rv=3.1):
    """wang2019(self, wave, ebv, waveunit=u.AA, Rv=3.1):
    Extinction function given by the law from Wang & Chen (2019).
    Parameters
    ----------
    wave : numpy.ndarray (1-d) 
        Wavelengths.
    ebv : float
        E(B-V) in magnitude.
    waveunit : astropy.units.Quantity
        Unit of the wavelengths.
    Rv : float
        Default: 3.1.
    Returns
    -------  
    Alam : numpy.ndarray (1-d)
        Extinction in magnitudes. 
    Paper information
    -------  
    bibcode: 2019ApJ...877..116W
    doi: 10.3847/1538-4357/ab1c61
    adsurl: https://ui.adsabs.harvard.edu/abs/2019ApJ...877..116W/abstract
    """

    Av = Rv * ebv
    uwave = wave * waveunit
    uwave = uwave.to(u.micron)
    Yf = 1/uwave.value - 1.82
    Alam_over_Av = 1.0 + 0.7499 * Yf - 0.1086 * Yf ** 2 - \
        0.08909 * Yf ** 3 + 0.02905 * Yf ** 4 + 0.01069 * \
        Yf ** 5 + 0.001707 * Yf ** 6 - 0.001002 * Yf ** 7
    if uwave.value.max() >= 1.0:
        mask = uwave.value >= 1.0
        Alam_over_Av[mask] = 0.3722 * uwave.value[mask] ** (-2.070)
    Alam = Alam_over_Av * Av
    return Alam

def f99law(wave, ebv, waveunit=u.AA, Rv=3.1):
    """
    Extinction function given by the law from Fitzpatrick (1999).
    Parameters
    ----------
    wave : numpy.ndarray (1-d) 
        Wavelengths.
    ebv : float
        E(B-V) in magnitude.
    waveunit : astropy.units.Quantity
        Unit of the wavelengths.
    Rv : float
        Default: 3.1.
    Returns
    -------  
    Alam : numpy.ndarray (1-d)
        Extinction in magnitudes.
    Paper information
    -------  
    bibcode: 1999PASP..111...63F
    doi: 10.1086/316293
    adsurl: https://ui.adsabs.harvard.edu/abs/1999PASP..111...63F/abstract
    """
    
    Av = Rv * ebv
    uwave = wave * waveunit
    x = 1.0 / uwave.to(u.micron).value  # Convert to inverse microns
    
    # Initialize extinction curve
    k = np.zeros_like(x)
    
    # IR - Optical: 0.3 <= x <= 1.1
    ir_opt = (x >= 0.3) & (x <= 1.1)
    if np.any(ir_opt):
        k[ir_opt] = 1.0 + 0.17699 * (x[ir_opt] - 1.82) - 0.50447 * (x[ir_opt] - 1.82)**2 - \
                    0.02427 * (x[ir_opt] - 1.82)**3 + 0.72085 * (x[ir_opt] - 1.82)**4 + \
                    0.01979 * (x[ir_opt] - 1.82)**5 - 0.77530 * (x[ir_opt] - 1.82)**6 + \
                    0.32999 * (x[ir_opt] - 1.82)**7
    
    # UV: 1.1 < x <= 3.3
    uv = (x > 1.1) & (x <= 3.3)
    if np.any(uv):
        y = x[uv] - 1.82
        k[uv] = 1.0 + 0.104 * y - 0.609 * y**2 + 0.701 * y**3 + 1.137 * y**4 - \
                1.718 * y**5 - 0.827 * y**6 + 1.647 * y**7 - 0.505 * y**8
    
    # Far-UV: 3.3 < x <= 8.0
    fuv = (x > 3.3) & (x <= 8.0)
    if np.any(fuv):
        y = x[fuv]
        k[fuv] = 1.752 - 0.316 * y - 0.104 / ((y - 4.67)**2 + 0.341)
    
    # Extrapolation to 8.0 < x <= 11.0 (down to 912 Angstroms)
    ext = x > 8.0
    if np.any(ext):
        y = x[ext]
        k[ext] = 1.752 - 0.316 * y - 0.104 / ((y - 4.67)**2 + 0.341)
    
    Alam = k * Av
    return Alam


def getebv(ra, dec, mapname='planck', map_dir=None):
    """
    Query the dust map with "dustmaps" to get the line-of-sight
    E(B-V) value for a given object.
    Parameters:
    ----------
        ra : float
            Right ascension in degrees.
        dec : float
            Declination in degrees.
        mapname : str
            One of ['sfd', 'planck', 'planck14', 'planck16']. 
            Other maps are avaliable in "dustmaps" but not implemented here: 
            ['bayestar', 'iphas', 'marshall', chen2014',  
            'lenz2017', 'pg2010', 'leike_ensslin_2019', 'leike2020']
            Default: 'planck' (equivalent to 'planck16').
        map_dir : str
            Path to the directory containing the dust map files. 
            Default: None.
    Returns:
    -------
        ebv : float
            E(B-V) in magnitude.
    """
    coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
    if mapname.lower()=='planck' or mapname.lower()=='planck16':
        from dustmaps.planck import PlanckGNILCQuery
        planck = PlanckGNILCQuery()
        ebv = planck(coord)
    elif mapname.lower()=='planck14':
        from dustmaps.planck import PlanckQuery
        planck = PlanckQuery()
        ebv = planck(coord)
    elif mapname.lower()=='sfd':
        from dustmaps.sfd import SFDQuery
        sfd = SFDQuery(map_dir=map_dir)
        ebv = sfd(coord)
    return ebv

        
def redden(Alam, flux):
    """
    Apply extinction to flux values.
    Parameters
    ----------
    Alam : numpy.ndarray (1-d)
        Extinction in magnitudes. 
    flux : numpy.ndarray
        Flux values.
    Returns
    -------
    flux_reddened : numpy.ndarray (1-d)
        Flux values with extinction applied.
    """

    flux_ratio = 10. ** (-0.4 * Alam)
    flux_reddened = flux * flux_ratio
    return flux_reddened


def deredden(Alam, flux):
    """
    Remove extinction from flux values.
    Parameters
    ----------
    extinction : numpy.ndarray (1-d)
        Extinction in magnitudes.
    flux : numpy.ndarray
        Flux values.
    Returns
    -------
    flux_dereddened : numpy.ndarray (1-d)
        Flux values with extinction removed.
    """

    flux_ratio = 10. ** (0.4 * Alam)
    flux_dereddened = flux * flux_ratio
    return flux_dereddened