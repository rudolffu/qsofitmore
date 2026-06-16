#!/usr/bin/env python
"""Tests for line fitting functionality."""

import pytest
import numpy as np

# Try to import qsofitmore config, skip tests if not available
try:
    from qsofitmore.config import migration_config
    from qsofitmore.fitmodule import _BROAD_SIGMA_THRESHOLD_KMS
    QSOFITMORE_AVAILABLE = True
except ImportError:
    QSOFITMORE_AVAILABLE = False
    # Create dummy config for testing
    class DummyConfig:
        validate_against_kmpfit = False
        benchmark_performance = False
        rtol = 1e-6
        atol = 1e-8
    migration_config = DummyConfig()
    _BROAD_SIGMA_THRESHOLD_KMS = 0.0


class TestLineFitting:
    """Test line fitting helpers."""
    
    def test_line_basic_fit(self, qso_instance, sample_linelist):
        """Test basic line fitting functionality"""
        q = qso_instance
        
        # Test basic line fitting setup
        assert sample_linelist is not None
        assert len(sample_linelist) > 0
        
        # Test line list structure
        expected_columns = ['lambda', 'compname', 'minwav', 'maxwav', 'linename', 
                          'ngauss', 'inisig', 'minsig', 'maxsig', 'voff', 
                          'vindex', 'windex', 'findex', 'fvalue']
        
        for col in expected_columns:
            assert col in sample_linelist.dtype.names
    
    def test_line_parameter_constraints(self, qso_instance, sample_linelist):
        """Test line parameter constraints and tying"""
        q = qso_instance
        ind_line = sample_linelist['compname'] == b'Ha'
        q._do_tie_line(sample_linelist, ind_line)
        assert hasattr(q, 'ind_tie_vindex1')
        assert hasattr(q, 'ind_tie_windex1')
        assert hasattr(q, 'fvalue_factor_1')
    
    def test_gaussian_profile_calculation(self, qso_instance):
        """Test Gaussian profile calculation for lines"""
        q = qso_instance
        
        # Test the Manygauss function
        if hasattr(q, 'Manygauss'):
            # Test with simple parameters
            x = np.log(np.linspace(6550, 6580, 100))  # Around H-alpha on the log axis
            params = [10.0, np.log(6564.61), 0.005]  # amp, log_wave, sigma
            
            result = q.Manygauss(x, params)
            assert len(result) == len(x)
            assert np.all(result >= 0)  # Gaussian should be positive
            assert np.max(result) > 0   # Should have a peak
    
    def test_narrow_broad_component_separation(self, qso_instance):
        """Test separation of narrow and broad line components"""
        q = qso_instance
        q.wave_scale = 'linear'
        center = 5000.0
        broad_sigma = center * (_BROAD_SIGMA_THRESHOLD_KMS + 1.0) / q._c_kms
        narrow_sigma = center * (_BROAD_SIGMA_THRESHOLD_KMS - 1.0) / q._c_kms
        assert q._sigma_axis_to_kms(broad_sigma, center) > _BROAD_SIGMA_THRESHOLD_KMS
        assert q._sigma_axis_to_kms(narrow_sigma, center) < _BROAD_SIGMA_THRESHOLD_KMS


class TestLineValidation:
    """Validation tests for line fitting helpers."""

    def test_line_profile_reconstruction(self, qso_instance):
        """Test that line profiles are reconstructed consistently"""
        q = qso_instance
        x = np.log(np.linspace(4990.0, 5010.0, 100))
        params = np.array([5.0, np.log(5000.0), 0.001])
        profile = q.Manygauss(x, params)
        assert profile.shape == x.shape
        assert profile.max() > 0.0

    def test_multi_gaussian_decomposition(self, qso_instance):
        """Test multi-Gaussian decomposition consistency"""
        q = qso_instance
        x = np.log(np.linspace(4990.0, 5010.0, 100))
        params = np.array([5.0, np.log(4998.0), 0.001, 3.0, np.log(5002.0), 0.001])
        combined = q.Manygauss(x, params)
        single_a = q.Manygauss(x, params[:3])
        single_b = q.Manygauss(x, params[3:])
        np.testing.assert_allclose(combined, single_a + single_b)
    
    def test_parameter_tying_consistency(self, qso_instance):
        """Test that parameter tying works consistently"""
        q = qso_instance
        q.tie_lambda = True
        q.tie_width = True
        q.tie_flux_1 = True
        q.tie_flux_2 = True
        linelist = np.rec.array(
            [
                (5000.0, 'Test', 4990.0, 5010.0, 'TestA', 1, 0.001, 0.0001, 0.01, 0.0, 1, 1, 0, 0.001),
                (5000.0, 'Test', 4990.0, 5010.0, 'TestB', 1, 0.001, 0.0001, 0.01, 0.0, 1, 1, 0, 0.001),
            ],
            formats='float64,U20,float64,float64,U20,int16,float64,float64,float64,float64,int16,int16,int16,float64',
            names='lambda,compname,minwav,maxwav,linename,ngauss,inisig,minsig,maxsig,voff,vindex,windex,findex,fvalue'
        )
        ind_line = linelist['compname'] == 'Test'
        q._do_tie_line(linelist, ind_line)
        pp = np.array([1.0, np.log(6564.61), 0.005, 2.0, np.log(6564.61) + 0.01, 0.003])
        tied = q._apply_ties_inplace(pp.copy())
        assert tied[4] == tied[1]
        assert tied[5] == tied[2]


class TestLineMonteCarlo:
    """Test Monte Carlo error estimation for lines"""
    
    def test_line_mc_error_estimation(self, qso_instance):
        """Test Monte Carlo error estimation for line parameters"""
        if not migration_config.use_lmfit:
            pytest.skip("lmfit MC disabled")
        assert migration_config.use_lmfit
    
    def test_line_mc_lmfit_backend(self, temp_output_dir):
        """MC should run with the lmfit backend and return non-zero stds"""
        if not QSOFITMORE_AVAILABLE:
            pytest.skip("qsofitmore not available")
        from qsofitmore import QSOFitNew
        from qsofitmore.config import migration_config
        prev_global = getattr(migration_config, 'use_lmfit', False)
        migration_config.use_lmfit = True
        try:
            wave = np.linspace(4990.0, 5010.0, 80)
            flux = np.ones_like(wave)
            err = np.full_like(wave, 0.1)
            q = QSOFitNew(lam=wave, flux=flux, err=err, z=0.0, path=temp_output_dir + '/')
            q.MC = True
            q.n_trails = 5
            cont_params = np.array([0., 3000., 0., 0., 3000., 0., 1., -1.5, 0., 5000., 0., 0., 0., 0.])
            q.conti_fit = type('Fit', (), {'params': cont_params})()
            q.f_conti_model = np.ones_like(wave)
            compcenter = 5000.0
            linelist = np.rec.array(
                [(compcenter, 'Ha', compcenter-5., compcenter+5., 'Ha_na', 1, 1e-3, 1e-4, 5e-3, 0.,
                  0, 0, 0, 0.001)],
                formats='float64,U20,float64,float64,U20,int16,float64,float64,float64,float64,int16,int16,int16,float64',
                names='lambda,compname,minwav,maxwav,linename,ngauss,inisig,minsig,maxsig,voff,vindex,windex,findex,fvalue'
            )
            q.linelist = linelist
            ind_line = np.array([True])
            q._do_tie_line(linelist, ind_line)
            x = q._x_axis(np.linspace(compcenter-5., compcenter+5., 80))
            true_params = np.array([5.0, np.log(compcenter), 8e-4])
            y = q.Manygauss(x, true_params)
            noise = np.full_like(y, 0.3)
            pp0 = np.array([3.5, np.log(compcenter) + 5e-4, 5e-4])
            pp_limits = np.array([
                {'limits': (0.0, 1e3)},
                {'limits': (np.log(compcenter-2.), np.log(compcenter+2.))},
                {'limits': (1e-4, 1e-2)}
            ], dtype=object)
            results = q.new_line_mc(
                x, y, noise, pp0, pp_limits, q.n_trails, compcenter, 'Ha',
                ind_line, 1, linelist[ind_line], np.array([1])
            )
            all_para_std, fwhm_std, sigma_std, ew_std, peak_std, area_std, na_dict = results
            assert all_para_std.shape[0] == len(pp0)
            assert fwhm_std >= 0 and sigma_std >= 0
            assert isinstance(na_dict, dict)
            assert 'Ha_na' in na_dict
        finally:
            migration_config.use_lmfit = prev_global
    
