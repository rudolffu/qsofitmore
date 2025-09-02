#!/usr/bin/env python
"""Pytest configuration and shared fixtures for qsofitmore tests"""

import pytest
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.io import fits
import tempfile
import os
from qsofitmore import QSOFitNew


@pytest.fixture
def sample_spectrum():
    """Generate a synthetic QSO spectrum for testing"""
    # Create wavelength array
    wave = np.arange(3800, 9200, 2.0)  # Typical SDSS range
    
    # Create synthetic continuum (power law + host galaxy)
    continuum = 10 * (wave / 5000)**(-1.5) + 2 * (wave / 5000)**(-0.5)
    
    # Add emission lines
    lines = {
        'Ha': {'wave': 6564.61, 'amp': 50, 'sigma': 15},
        'Hb': {'wave': 4862.68, 'amp': 20, 'sigma': 12},
        'OIII5007': {'wave': 5008.24, 'amp': 15, 'sigma': 5},
        'OIII4959': {'wave': 4960.30, 'amp': 5, 'sigma': 5},
        'MgII': {'wave': 2798.75, 'amp': 30, 'sigma': 20},
    }
    
    emission = np.zeros_like(wave)
    for line_name, params in lines.items():
        emission += params['amp'] * np.exp(-0.5 * ((wave - params['wave']) / params['sigma'])**2)
    
    # Total flux
    flux = continuum + emission
    
    # Add realistic noise
    snr = 20  # Signal-to-noise ratio
    noise = flux / snr
    flux += np.random.normal(0, noise)
    err = noise * np.ones_like(flux)
    
    return wave, flux, err


@pytest.fixture
def sample_qso_params():
    """Sample QSO parameters for testing"""
    return {
        'z': 0.1684,
        'ra': 3.97576206,
        'dec': 56.04931383,
        'name': 'TEST_QSO',
        'is_sdss': False
    }


@pytest.fixture
def temp_output_dir():
    """Temporary directory for test outputs"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup handled by tempfile


@pytest.fixture
def sample_linelist():
    """Create a sample line parameter file for testing"""
    newdata = np.rec.array([
        (6564.61,'Ha',6400.,6800.,'Ha_br',3,5e-3,0.003,0.01,0.005,0,0,0,0.05),
        (6564.61,'Ha',6400.,6800.,'Ha_na',1,1e-3,5e-4,0.0017,0.01,1,1,0,0.002),
        (4862.68,'Hb',4640.,5100.,'Hb_br',3,5e-3,0.003,0.01,0.003,0,0,0,0.01),
        (4862.68,'Hb',4640.,5100.,'Hb_na',1,1e-3,2.3e-4,0.0017,0.01,1,1,0,0.002),
        (4960.30,'Hb',4640.,5100.,'OIII4959',1,1e-3,2.3e-4,0.0017,0.01,1,1,0,0.002),
        (5008.24,'Hb',4640.,5100.,'OIII5007',1,1e-3,2.3e-4,0.0017,0.01,1,1,0,0.004),
    ], formats='float32,a20,float32,float32,a20,float32,float32,float32,float32,float32,float32,float32,float32,float32',
       names='lambda,compname,minwav,maxwav,linename,ngauss,inisig,minsig,maxsig,voff,vindex,windex,findex,fvalue')
    
    return newdata


@pytest.fixture
def qso_instance(sample_spectrum, sample_qso_params, temp_output_dir):
    """Create QSOFitNew instance for testing"""
    wave, flux, err = sample_spectrum
    
    q = QSOFitNew(
        lam=wave, 
        flux=flux * 1e17,  # Convert to expected units
        err=err * 1e17, 
        z=sample_qso_params['z'],
        ra=sample_qso_params['ra'], 
        dec=sample_qso_params['dec'],
        name=sample_qso_params['name'], 
        is_sdss=sample_qso_params['is_sdss'], 
        path=temp_output_dir + '/'
    )
    
    return q


@pytest.fixture
def reference_fit_results():
    """Reference fit results for validation (these would be from known good kmpfit runs)"""
    # These would be populated with actual reference values from validated kmpfit runs
    return {
        'continuum_params': [0., 3000., 0., 0., 3000., 0., 1., -1.5, 0., 5e3, 0., 0., 0., 0.],
        'continuum_chi2': 1.2,
        'line_params_ha': [10.0, np.log(6564.61), 0.005],  # amp, log_wave, sigma
        'line_chi2_ha': 1.1,
        'monte_carlo_std': [0.1, 50., 0.001, 0.1, 50., 0.001, 0.05, 0.02, 0.1, 100., 0.1, 0., 0., 0.]
    }