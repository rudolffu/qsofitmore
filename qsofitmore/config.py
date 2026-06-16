import os
import warnings


class Config():

    def __init__(self, fname) -> None:
        pass

    def load(self):
        pass

    def save(self):
        pass


class MigrationConfig:
    """Runtime configuration for optimizer backend and line-axis defaults.

    The name is retained for compatibility with existing notebooks, but lmfit is
    now the default backend and the migration-era per-component flags collapse to
    the single public ``use_lmfit`` switch.
    """
    
    def __init__(self):
        # Default to lmfit unless explicitly overridden via env/config.
        self._use_lmfit = os.environ.get('QSOFITMORE_USE_LMFIT', 'true').lower() == 'true'
        deprecated_backend_env = [
            name for name in (
                'QSOFITMORE_USE_LMFIT_CONTINUUM',
                'QSOFITMORE_USE_LMFIT_LINES',
                'QSOFITMORE_USE_LMFIT_MC',
            )
            if name in os.environ
        ]
        if deprecated_backend_env:
            warnings.warn(
                ", ".join(deprecated_backend_env)
                + " are deprecated and ignored; set QSOFITMORE_USE_LMFIT=true/false instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        
        # Wavelength axis and velocity-param units
        # wave_scale: 'auto' (default), 'log', or 'linear'
        self.wave_scale = 'auto'
        # velocity_units: how to interpret inisig/minsig/maxsig/voff in line tables
        # 'auto' (default), 'lnlambda', or 'km/s'
        self.velocity_units = 'auto'
        # Max width for narrow components, in km/s (default 1200)
        self.narrow_max_kms = 1200.0
        self.refresh_axis_from_env(force=True)
        
        # Testing and validation flags
        self.validate_against_kmpfit = os.environ.get('QSOFITMORE_VALIDATE_KMPFIT', 'true').lower() == 'true'
        self.benchmark_performance = os.environ.get('QSOFITMORE_BENCHMARK', 'false').lower() == 'true'
        
        # Tolerance settings for validation
        self.rtol = float(os.environ.get('QSOFITMORE_RTOL', '1e-6'))
        self.atol = float(os.environ.get('QSOFITMORE_ATOL', '1e-8'))

    @property
    def use_lmfit(self):
        return self._use_lmfit

    @use_lmfit.setter
    def use_lmfit(self, value: bool):
        """Select lmfit (True) or legacy kmpfit (False) for all fit components."""
        self._use_lmfit = bool(value)

    @property
    def use_lmfit_continuum(self):
        """Compatibility alias for ``use_lmfit``."""
        return self.use_lmfit

    @use_lmfit_continuum.setter
    def use_lmfit_continuum(self, value: bool):
        warnings.warn(
            "use_lmfit_continuum is deprecated; set use_lmfit instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.use_lmfit = bool(value)

    @property
    def use_lmfit_lines(self):
        """Compatibility alias for ``use_lmfit``."""
        return self.use_lmfit

    @use_lmfit_lines.setter
    def use_lmfit_lines(self, value: bool):
        warnings.warn(
            "use_lmfit_lines is deprecated; set use_lmfit instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.use_lmfit = bool(value)

    @property
    def use_lmfit_mc(self):
        """Compatibility alias for ``use_lmfit``."""
        return self.use_lmfit

    @use_lmfit_mc.setter
    def use_lmfit_mc(self, value: bool):
        warnings.warn(
            "use_lmfit_mc is deprecated; Monte Carlo uses the active backend selected by use_lmfit.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.use_lmfit = bool(value)
    
    def enable_lmfit_gradually(self):
        """Compatibility shim for the completed migration."""
        warnings.warn(
            "enable_lmfit_gradually is deprecated; lmfit is controlled by use_lmfit.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.use_lmfit = True
        return 'complete'
    
    def status(self):
        """Return current runtime status."""
        return {
            'backend': 'lmfit' if self.use_lmfit else 'kmpfit',
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

    def refresh_axis_from_env(self, force=False):
        """Refresh axis/unit settings from environment variables.

        When ``force`` is false, only variables present in the environment are
        applied. This lets notebooks set env vars after importing qsofitmore
        without clobbering explicit in-Python config edits.
        """
        if force or 'QSOFITMORE_WAVE_SCALE' in os.environ:
            wave_scale = os.environ.get('QSOFITMORE_WAVE_SCALE', self.wave_scale).strip().lower()
            self.wave_scale = wave_scale if wave_scale in ('auto', 'log', 'linear') else 'auto'

        if force or 'QSOFITMORE_VELOCITY_UNITS' in os.environ:
            velocity_units = os.environ.get('QSOFITMORE_VELOCITY_UNITS', self.velocity_units).strip().lower()
            if velocity_units in ('kms', 'km/s', 'kmps'):
                self.velocity_units = 'km/s'
            elif velocity_units in ('auto', 'lnlambda'):
                self.velocity_units = velocity_units

        if force or 'QSOFITMORE_NARROW_MAX_KMS' in os.environ:
            try:
                self.narrow_max_kms = float(os.environ.get('QSOFITMORE_NARROW_MAX_KMS', self.narrow_max_kms))
            except Exception:
                self.narrow_max_kms = 1200.0


# Global migration configuration instance
migration_config = MigrationConfig()
