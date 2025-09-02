#!/usr/bin/env python
"""Utility functions and tests for migration testing"""

import pytest
import numpy as np
import tempfile
import os

# Try to import qsofitmore config, skip tests if not available
try:
    from qsofitmore.config import migration_config
    QSOFITMORE_AVAILABLE = True
except ImportError:
    QSOFITMORE_AVAILABLE = False
    # Create dummy config for testing
    class DummyConfig:
        rtol = 1e-6
        atol = 1e-8
    migration_config = DummyConfig()


def generate_reference_spectrum(wave_min=3800, wave_max=9200, resolution=2.0, z=0.5):
    """Generate a reference QSO spectrum for testing"""
    wave = np.arange(wave_min, wave_max, resolution)
    
    # Power-law continuum
    continuum = 10 * (wave / 5000)**(-1.5)
    
    # Add host galaxy component
    host = 2 * np.exp(-0.5 * ((wave - 5500) / 1000)**2)
    
    # Emission lines at observed wavelengths
    lines_rest = {
        'Lya': 1215.67,
        'CIV': 1549.06, 
        'CIII': 1908.73,
        'MgII': 2798.75,
        'Hb': 4862.68,
        'OIII5007': 5008.24,
        'Ha': 6564.61,
    }
    
    emission = np.zeros_like(wave)
    for line_name, rest_wave in lines_rest.items():
        obs_wave = rest_wave * (1 + z)
        if wave_min <= obs_wave <= wave_max:
            if 'broad' in line_name.lower() or line_name in ['Ha', 'Hb', 'MgII']:
                # Broad line
                sigma = 20
                amp = 30
            else:
                # Narrow line  
                sigma = 5
                amp = 15
            
            emission += amp * np.exp(-0.5 * ((wave - obs_wave) / sigma)**2)
    
    flux = continuum + host + emission
    
    # Add realistic noise
    snr = 20
    noise = flux / snr
    flux += np.random.normal(0, noise, size=len(flux))
    err = noise
    
    return wave, flux, err


def compare_fit_results(result1, result2, rtol=1e-5, atol=1e-8, param_names=None):
    """Compare two sets of fit results within tolerance"""
    if param_names is None:
        param_names = [f'param_{i}' for i in range(len(result1))]
    
    differences = []
    for i, (p1, p2, name) in enumerate(zip(result1, result2, param_names)):
        if not np.allclose([p1], [p2], rtol=rtol, atol=atol):
            rel_diff = abs(p1 - p2) / max(abs(p1), abs(p2), atol)
            differences.append({
                'parameter': name,
                'index': i,
                'value1': p1,
                'value2': p2,
                'abs_diff': abs(p1 - p2),
                'rel_diff': rel_diff
            })
    
    return differences


def create_test_linelist():
    """Create a comprehensive test line list"""
    import numpy as np
    
    lines = [
        # H-alpha complex
        (6564.61, 'Ha', 6400., 6800., 'Ha_br', 3, 5e-3, 0.003, 0.01, 0.005, 0, 0, 0, 0.05),
        (6564.61, 'Ha', 6400., 6800., 'Ha_na', 1, 1e-3, 5e-4, 0.0017, 0.01, 1, 1, 0, 0.002),
        (6549.85, 'Ha', 6400., 6800., 'NII6549', 1, 1e-3, 2.3e-4, 0.0017, 5e-3, 1, 1, 1, 0.001),
        (6585.28, 'Ha', 6400., 6800., 'NII6585', 1, 1e-3, 2.3e-4, 0.0017, 5e-3, 1, 1, 1, 0.003),
        
        # H-beta + OIII complex
        (4862.68, 'Hb', 4640., 5100., 'Hb_br', 3, 5e-3, 0.003, 0.01, 0.003, 0, 0, 0, 0.01),
        (4862.68, 'Hb', 4640., 5100., 'Hb_na', 1, 1e-3, 2.3e-4, 0.0017, 0.01, 2, 2, 0, 0.002),
        (4960.30, 'Hb', 4640., 5100., 'OIII4959', 1, 1e-3, 2.3e-4, 0.0017, 0.01, 2, 2, 0, 0.002),
        (5008.24, 'Hb', 4640., 5100., 'OIII5007', 1, 1e-3, 2.3e-4, 0.0017, 0.01, 2, 2, 0, 0.004),
        
        # MgII
        (2798.75, 'MgII', 2700., 2900., 'MgII_br', 2, 5e-3, 0.004, 0.015, 0.0017, 0, 0, 0, 0.05),
        (2798.75, 'MgII', 2700., 2900., 'MgII_na', 1, 1e-3, 2.3e-4, 0.0017, 0.01, 0, 0, 0, 0.002),
        
        # CIV
        (1549.06, 'CIV', 1500., 1700., 'CIV_br', 3, 5e-3, 0.004, 0.015, 0.015, 0, 0, 0, 0.05),
    ]
    
    linelist = np.rec.array(
        lines,
        formats='float32,a20,float32,float32,a20,float32,float32,float32,float32,float32,float32,float32,float32,float32',
        names='lambda,compname,minwav,maxwav,linename,ngauss,inisig,minsig,maxsig,voff,vindex,windex,findex,fvalue'
    )
    
    return linelist


class TestUtilities:
    """Test utility functions"""
    
    def test_reference_spectrum_generation(self):
        """Test reference spectrum generation"""
        wave, flux, err = generate_reference_spectrum(z=0.1)
        
        assert len(wave) == len(flux) == len(err)
        assert np.all(wave > 0)
        assert np.all(flux > 0) 
        assert np.all(err > 0)
        assert wave[0] < wave[-1]  # Increasing wavelength
    
    def test_fit_result_comparison(self):
        """Test fit result comparison utility"""
        result1 = [1.0, 2.0, 3.0]
        result2 = [1.001, 2.002, 3.003]  # Small differences
        
        diffs = compare_fit_results(result1, result2, rtol=1e-2)
        assert len(diffs) == 0  # Should be within tolerance
        
        # Test with larger differences
        result3 = [1.1, 2.1, 3.1]  
        diffs = compare_fit_results(result1, result3, rtol=1e-3)
        assert len(diffs) == 3  # All parameters differ
    
    def test_linelist_creation(self):
        """Test line list creation"""
        linelist = create_test_linelist()
        
        assert len(linelist) > 0
        assert 'lambda' in linelist.dtype.names
        assert 'linename' in linelist.dtype.names
        
        # Check that we have the major lines
        line_names = [name.decode() if isinstance(name, bytes) else name 
                     for name in linelist['linename']]
        assert 'Ha_br' in line_names
        assert 'Hb_br' in line_names
        assert 'OIII5007' in line_names
    
    def test_tolerance_settings(self):
        """Test tolerance setting functionality"""
        config = migration_config
        
        # Test that tolerances are reasonable
        assert 0 < config.rtol < 1
        assert 0 <= config.atol < 1
        
        # Test that they can be used in comparisons
        test_vals = [1.0, 1.000001]
        assert np.allclose(test_vals, [1.0, 1.0], rtol=config.rtol, atol=config.atol)
    
    def test_temporary_file_handling(self):
        """Test temporary file handling for tests"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, 'test.txt')
            
            with open(test_file, 'w') as f:
                f.write('test content')
            
            assert os.path.exists(test_file)
            
            with open(test_file, 'r') as f:
                content = f.read()
                assert content == 'test content'
        
        # File should be cleaned up
        assert not os.path.exists(test_file)


class TestDataValidation:
    """Test data validation utilities"""
    
    def test_spectrum_validation(self):
        """Test spectrum data validation"""
        wave, flux, err = generate_reference_spectrum()
        
        # Test basic validation
        assert not np.any(np.isnan(wave))
        assert not np.any(np.isnan(flux))
        assert not np.any(np.isnan(err))
        assert not np.any(np.isinf(wave))
        assert not np.any(np.isinf(flux))
        assert not np.any(np.isinf(err))
    
    def test_parameter_validation(self):
        """Test parameter validation"""
        # Test reasonable parameter ranges
        test_params = [1.0, 3000.0, 0.0, 1.0, 5000.0, 0.0, 10.0, -1.5, 5.0, 5000.0, 2.0]
        
        # These should be reasonable continuum parameters
        assert all(np.isfinite(p) for p in test_params)
    
    def test_line_parameter_validation(self):
        """Test line parameter validation"""
        linelist = create_test_linelist()
        
        # Test that line parameters are reasonable
        assert np.all(linelist['lambda'] > 1000)  # Reasonable wavelengths
        assert np.all(linelist['lambda'] < 10000)
        assert np.all(linelist['minsig'] > 0)     # Positive sigma limits
        assert np.all(linelist['maxsig'] > linelist['minsig'])  # Max > min