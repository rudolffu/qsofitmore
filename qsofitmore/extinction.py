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


def getebv(ra, dec, mapname='planck', mode=None):
    """
    Query the dust map with "dustmaps" to get the line-of-sight
    E(B-V) value for a given object.
    Parameters:
    ----------
        mapname : str
            One of ['sfd', 'planck']. Other maps are 
            avaliable in "dustmaps" but not implemented here: 
            ['bayestar', 'iphas', 'marshall', chen2014',  
            'lenz2017', 'pg2010', 'leike_ensslin_2019', 'leike2020']
            Default: 'planck'.
        mode : str
            One of ['local', 'web']. Applicable only when
            mapname == 'sfd'. When 'local', query the local map 
            on disk. When 'web', query the web server.  
            Default: 'local'.
        """
    coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
    if mapname.lower()=='planck':
        from dustmaps.planck import PlanckQuery
        planck = PlanckQuery()
        ebv = planck(coord)
    elif mapname.lower()=='sfd' and mode=='local':
        from dustmaps.sfd import SFDQuery
        sfd = SFDQuery()
        ebv = sfd(coord)
    elif mapname.lower=='sfd' and mode=='web':
        from dustmaps.sfd import SFDWebQuery
        sfd = SFDWebQuery()
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