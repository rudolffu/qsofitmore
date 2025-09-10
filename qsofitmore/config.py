#!/usr/bin/env python
import json
import os
import sys


class Config():

    def __init__(self, fname) -> None:
        pass

    def load(self):
        pass

    def save(self):
        pass


class MigrationConfig:
    """Configuration for gradual migration from kmpfit to lmfit"""
    
    def __init__(self):
        # Feature flags for migration
        self._use_lmfit = os.environ.get('QSOFITMORE_USE_LMFIT', 'false').lower() == 'true'
        self.use_lmfit_continuum = os.environ.get('QSOFITMORE_USE_LMFIT_CONTINUUM', 'false').lower() == 'true'
        self.use_lmfit_lines = os.environ.get('QSOFITMORE_USE_LMFIT_LINES', 'false').lower() == 'true'
        self.use_lmfit_mc = os.environ.get('QSOFITMORE_USE_LMFIT_MC', 'false').lower() == 'true'
        
        # Wavelength axis and velocity-param units
        # wave_scale: 'log' (default, legacy) or 'linear'
        self.wave_scale = os.environ.get('QSOFITMORE_WAVE_SCALE', 'log').strip().lower()
        if self.wave_scale not in ('log', 'linear'):
            self.wave_scale = 'log'
        # velocity_units: how to interpret inisig/minsig/maxsig/voff in line tables
        # 'lnlambda' (default, legacy) or 'km/s'
        self.velocity_units = os.environ.get('QSOFITMORE_VELOCITY_UNITS', 'lnlambda').strip().lower()
        if self.velocity_units in ('kms', 'km/s', 'kmps'):
            self.velocity_units = 'km/s'
        elif self.velocity_units != 'lnlambda':
            self.velocity_units = 'lnlambda'
        # Max width for narrow components, in km/s (default 1200)
        try:
            self.narrow_max_kms = float(os.environ.get('QSOFITMORE_NARROW_MAX_KMS', '1200'))
        except Exception:
            self.narrow_max_kms = 1200.0
        
        # Testing and validation flags
        self.validate_against_kmpfit = os.environ.get('QSOFITMORE_VALIDATE_KMPFIT', 'true').lower() == 'true'
        self.benchmark_performance = os.environ.get('QSOFITMORE_BENCHMARK', 'false').lower() == 'true'
        
        # Tolerance settings for validation
        self.rtol = float(os.environ.get('QSOFITMORE_RTOL', '1e-6'))
        self.atol = float(os.environ.get('QSOFITMORE_ATOL', '1e-8'))
        # Global override on init: enabling global lmfit turns on per-component flags by default
        if self._use_lmfit:
            self.use_lmfit_continuum = True if os.environ.get('QSOFITMORE_USE_LMFIT_CONTINUUM') is None else self.use_lmfit_continuum
            self.use_lmfit_lines = True if os.environ.get('QSOFITMORE_USE_LMFIT_LINES') is None else self.use_lmfit_lines

    @property
    def use_lmfit(self):
        return self._use_lmfit

    @use_lmfit.setter
    def use_lmfit(self, value: bool):
        """Setting global lmfit also cascades to per-component flags at runtime."""
        self._use_lmfit = bool(value)
        if self._use_lmfit:
            self.use_lmfit_continuum = True
            self.use_lmfit_lines = True
        else:
            self.use_lmfit_continuum = False
            self.use_lmfit_lines = False
    
    def enable_lmfit_gradually(self):
        """Enable lmfit components in order of risk (lowest first)"""
        if not self.use_lmfit_continuum:
            self.use_lmfit_continuum = True
            return 'continuum'
        elif not self.use_lmfit_lines:
            self.use_lmfit_lines = True
            return 'lines'
        elif not self.use_lmfit_mc:
            self.use_lmfit_mc = True
            return 'monte_carlo'
        else:
            self.use_lmfit = True
            return 'complete'
    
    def status(self):
        """Return current migration status"""
        return {
            'global_lmfit': self.use_lmfit,
            'continuum_fitting': self.use_lmfit_continuum,
            'line_fitting': self.use_lmfit_lines,
            'monte_carlo': self.use_lmfit_mc,
            'validation_enabled': self.validate_against_kmpfit,
            'benchmarking': self.benchmark_performance,
            'wave_scale': self.wave_scale,
            'velocity_units': self.velocity_units,
            'narrow_max_kms': self.narrow_max_kms,
        }


# Global migration configuration instance
migration_config = MigrationConfig()
