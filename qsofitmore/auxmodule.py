#!/usr/bin/env python
from astropy import units as u
from astropy.coordinates import SkyCoord
import matplotlib
import matplotlib.pyplot as plt


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
