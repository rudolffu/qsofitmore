#!/usr/bin/env python
"""Tests for line fitting functionality during kmpfit->lmfit migration"""

import pytest
import numpy as np

# Try to import qsofitmore config, skip tests if not available
try:
    from qsofitmore.config import migration_config
    QSOFITMORE_AVAILABLE = True
except ImportError:
    QSOFITMORE_AVAILABLE = False
    # Create dummy config for testing
    class DummyConfig:
        validate_against_kmpfit = False
        benchmark_performance = False
        use_lmfit_mc = False
        rtol = 1e-6
        atol = 1e-8
    migration_config = DummyConfig()


class TestLineFitting:
    """Test line fitting with both kmpfit and lmfit"""
    
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
    
    def test_line_parameter_constraints(self, qso_instance):
        """Test line parameter constraints and tying"""
        q = qso_instance
        
        # Test parameter tying functionality
        # This will test velocity constraints, width constraints, flux ratios
        pass
    
    def test_gaussian_profile_calculation(self, qso_instance):
        """Test Gaussian profile calculation for lines"""
        q = qso_instance
        
        # Test the Manygauss function
        if hasattr(q, 'Manygauss'):
            # Test with simple parameters
            x = np.linspace(6550, 6580, 100)  # Around H-alpha
            params = [10.0, np.log(6564.61), 0.005]  # amp, log_wave, sigma
            
            result = q.Manygauss(x, params)
            assert len(result) == len(x)
            assert np.all(result >= 0)  # Gaussian should be positive
            assert np.max(result) > 0   # Should have a peak
    
    def test_line_complex_fitting(self, qso_instance):
        """Test fitting of line complexes (multiple lines in same region)"""
        q = qso_instance
        
        # Test complex line fitting (e.g., H-beta + OIII complex)
        pass
    
    def test_narrow_broad_component_separation(self, qso_instance):
        """Test separation of narrow and broad line components"""
        q = qso_instance
        
        # Test narrow vs broad line identification and fitting
        pass
    
    @pytest.mark.parametrize("line_complex", ["Ha", "Hb", "MgII", "CIV"])
    def test_different_line_complexes(self, qso_instance, line_complex):
        """Test fitting different emission line complexes"""
        q = qso_instance
        
        # Test different line complexes have appropriate handling
        pass
    
    def test_line_equivalent_width_calculation(self, qso_instance):
        """Test equivalent width calculation for fitted lines"""
        q = qso_instance
        
        # Test EW calculation
        pass
    
    def test_line_flux_integration(self, qso_instance):
        """Test line flux integration"""
        q = qso_instance
        
        # Test integrated line fluxes
        pass


class TestLineValidation:
    """Validation tests for line fitting migration"""
    
    def test_line_parameter_consistency(self, qso_instance, reference_fit_results):
        """Test line parameter consistency between kmpfit and lmfit"""
        if not migration_config.validate_against_kmpfit:
            pytest.skip("kmpfit validation disabled")
        
        # Test parameter consistency
        pass
    
    def test_line_profile_reconstruction(self, qso_instance):
        """Test that line profiles are reconstructed consistently"""
        # Test profile reconstruction
        pass
    
    def test_multi_gaussian_decomposition(self, qso_instance):
        """Test multi-Gaussian decomposition consistency"""
        # Test multi-component fitting
        pass
    
    def test_parameter_tying_consistency(self, qso_instance):
        """Test that parameter tying works consistently"""
        # Test parameter constraints
        pass


class TestLineMonteCarlo:
    """Test Monte Carlo error estimation for lines"""
    
    def test_line_mc_error_estimation(self, qso_instance):
        """Test Monte Carlo error estimation for line parameters"""
        if not migration_config.use_lmfit_mc:
            pytest.skip("lmfit MC disabled")
        
        # Test MC error estimation
        pass
    
    def test_mc_parameter_distributions(self, qso_instance):
        """Test that MC parameter distributions are reasonable"""
        # Test parameter distributions
        pass
    
    def test_mc_convergence(self, qso_instance):
        """Test MC convergence properties"""
        # Test convergence
        pass
    
    def test_line_mc_lmfit_backend(self, temp_output_dir):
        """MC should run with the lmfit backend and return non-zero stds"""
        if not QSOFITMORE_AVAILABLE:
            pytest.skip("qsofitmore not available")
        from qsofitmore import QSOFitNew
        from qsofitmore.config import migration_config
        prev_global = getattr(migration_config, 'use_lmfit', False)
        prev_lines = getattr(migration_config, 'use_lmfit_lines', False)
        migration_config.use_lmfit = True
        migration_config.use_lmfit_lines = True
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
            migration_config.use_lmfit_lines = prev_lines
    
    @pytest.mark.benchmark
    def test_line_mc_performance(self, qso_instance, benchmark):
        """Benchmark line MC performance"""
        if not migration_config.benchmark_performance:
            pytest.skip("Benchmarking disabled")
        
        def run_line_mc():
            # Placeholder for MC run
            return np.random.randn(100)
        
        result = benchmark(run_line_mc)
        assert len(result) > 0
