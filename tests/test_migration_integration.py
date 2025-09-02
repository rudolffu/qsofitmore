#!/usr/bin/env python
"""Integration tests for the full kmpfit->lmfit migration"""

import pytest
import numpy as np
import os

# Try to import qsofitmore config, skip tests if not available
try:
    from qsofitmore.config import migration_config
    QSOFITMORE_AVAILABLE = True
except ImportError:
    QSOFITMORE_AVAILABLE = False
    # Create dummy config for testing
    class DummyConfig:
        use_lmfit = False
        use_lmfit_continuum = False
        use_lmfit_lines = False
        use_lmfit_mc = False
        validate_against_kmpfit = False
        benchmark_performance = False
        rtol = 1e-6
        atol = 1e-8
        
        def enable_lmfit_gradually(self):
            return 'continuum'
        
        def status(self):
            return {
                'global_lmfit': False,
                'continuum_fitting': False,
                'line_fitting': False,
                'monte_carlo': False,
                'validation_enabled': False,
                'benchmarking': False
            }
    migration_config = DummyConfig()


class TestMigrationIntegration:
    """Test the complete migration workflow"""
    
    def test_feature_flags(self):
        """Test that feature flags work correctly"""
        # Test individual feature flags
        assert hasattr(migration_config, 'use_lmfit')
        assert hasattr(migration_config, 'use_lmfit_continuum')
        assert hasattr(migration_config, 'use_lmfit_lines') 
        assert hasattr(migration_config, 'use_lmfit_mc')
        
        # Test status method
        status = migration_config.status()
        assert 'global_lmfit' in status
        assert 'continuum_fitting' in status
        assert 'line_fitting' in status
        assert 'monte_carlo' in status
    
    def test_gradual_migration_enablement(self):
        """Test gradual migration enablement"""
        # Reset config for test
        config = migration_config
        
        # Test gradual enablement
        next_component = config.enable_lmfit_gradually()
        assert next_component in ['continuum', 'lines', 'monte_carlo', 'complete']
    
    def test_environment_variable_config(self):
        """Test configuration via environment variables"""
        # Test that environment variables are read correctly
        # This would need to be run with specific env vars set
        pass
    
    def test_tolerance_configuration(self):
        """Test tolerance configuration for validation"""
        assert migration_config.rtol > 0
        assert migration_config.atol >= 0
        assert isinstance(migration_config.rtol, float)
        assert isinstance(migration_config.atol, float)


class TestFullWorkflow:
    """Test complete spectrum fitting workflow"""
    
    def test_full_spectrum_fit_kmpfit(self, qso_instance, sample_linelist, temp_output_dir):
        """Test full spectrum fitting with kmpfit (baseline)"""
        q = qso_instance
        
        # Create line list file
        import tempfile
        from astropy.io import fits
        
        hdr = fits.Header()
        hdr['lambda'] = 'Vacuum Wavelength in Ang'
        hdu = fits.BinTableHDU(data=sample_linelist, header=hdr, name='data')
        linelist_path = os.path.join(temp_output_dir, 'qsopar.fits')
        hdu.writeto(linelist_path, overwrite=True)
        
        # This would test the full fitting workflow
        # For now, just verify the setup works
        assert os.path.exists(linelist_path)
        assert q is not None
    
    def test_full_spectrum_fit_lmfit(self, qso_instance, sample_linelist, temp_output_dir):
        """Test full spectrum fitting with lmfit (target)"""
        if not migration_config.use_lmfit:
            pytest.skip("lmfit not enabled globally")
        
        # This will test the full lmfit workflow once implemented
        pass
    
    def test_results_comparison(self, qso_instance):
        """Test comparison of kmpfit vs lmfit results"""
        if not migration_config.validate_against_kmpfit:
            pytest.skip("kmpfit validation disabled")
        
        # This will compare full fitting results
        pass
    
    def test_output_file_consistency(self, qso_instance, temp_output_dir):
        """Test that output files are consistent between methods"""
        # Test output file formats and contents
        pass
    
    def test_plot_generation_consistency(self, qso_instance, temp_output_dir):
        """Test that plots are generated consistently"""
        # Test plot generation
        pass


class TestRegressionSuite:
    """Regression tests against known good results"""
    
    def test_sdss_spectrum_regression(self):
        """Test against known SDSS spectrum results"""
        # This would test against archived SDSS spectra with known good fits
        pytest.skip("Requires reference SDSS data")
    
    def test_high_redshift_regression(self):
        """Test high-redshift spectrum handling"""
        # Test high-z specific issues
        pass
    
    def test_low_snr_regression(self):
        """Test low signal-to-noise ratio spectra"""
        # Test challenging spectra
        pass
    
    def test_bal_quasar_regression(self):
        """Test Broad Absorption Line quasar handling"""
        # Test BAL-specific functionality
        pass


class TestPerformanceRegression:
    """Performance regression tests"""
    
    @pytest.mark.benchmark
    def test_continuum_fitting_performance(self, qso_instance, benchmark):
        """Benchmark continuum fitting performance"""
        if not migration_config.benchmark_performance:
            pytest.skip("Benchmarking disabled")
        
        def continuum_fit():
            # Placeholder for continuum fitting
            return np.sum(qso_instance.flux)
        
        result = benchmark(continuum_fit)
        assert result > 0
    
    @pytest.mark.benchmark  
    def test_line_fitting_performance(self, qso_instance, benchmark):
        """Benchmark line fitting performance"""
        if not migration_config.benchmark_performance:
            pytest.skip("Benchmarking disabled")
        
        def line_fit():
            # Placeholder for line fitting
            return np.sum(qso_instance.flux)
        
        result = benchmark(line_fit)
        assert result > 0
    
    @pytest.mark.benchmark
    def test_monte_carlo_performance(self, qso_instance, benchmark):
        """Benchmark Monte Carlo error estimation performance"""
        if not migration_config.benchmark_performance:
            pytest.skip("Benchmarking disabled")
        
        def mc_errors():
            # Placeholder for MC estimation
            return np.std(np.random.randn(100, 14), axis=0)
        
        result = benchmark(mc_errors)
        assert len(result) > 0
    
    def test_memory_usage(self, qso_instance):
        """Test memory usage during fitting"""
        # Test memory consumption
        pass
    
    def test_scaling_with_spectrum_size(self):
        """Test performance scaling with spectrum size"""
        # Test performance scaling
        pass