#!/usr/bin/env python
"""Runtime configuration tests after the lmfit migration."""

import os

import numpy as np
import pytest
from astropy.io import fits

from qsofitmore.config import MigrationConfig, migration_config


def test_runtime_config_defaults_to_lmfit(monkeypatch):
    """A fresh config should use lmfit and auto line-table detection."""
    for env_var in (
        "QSOFITMORE_USE_LMFIT",
        "QSOFITMORE_WAVE_SCALE",
        "QSOFITMORE_VELOCITY_UNITS",
    ):
        monkeypatch.delenv(env_var, raising=False)

    cfg = MigrationConfig()

    assert cfg.use_lmfit is True
    assert cfg.wave_scale == "auto"
    assert cfg.velocity_units == "auto"
    assert cfg.status()["backend"] == "lmfit"


def test_global_backend_switch_controls_compatibility_aliases():
    """Deprecated component flags should mirror the single public switch."""
    old_backend = migration_config.use_lmfit
    try:
        migration_config.use_lmfit = False
        assert migration_config.use_lmfit_continuum is False
        assert migration_config.use_lmfit_lines is False
        assert migration_config.use_lmfit_mc is False

        migration_config.use_lmfit = True
        assert migration_config.use_lmfit_continuum is True
        assert migration_config.use_lmfit_lines is True
        assert migration_config.use_lmfit_mc is True
    finally:
        migration_config.use_lmfit = old_backend


def test_deprecated_component_env_vars_warn_and_do_not_disable_lmfit(monkeypatch):
    monkeypatch.delenv("QSOFITMORE_USE_LMFIT", raising=False)
    monkeypatch.setenv("QSOFITMORE_USE_LMFIT_LINES", "false")

    with pytest.warns(DeprecationWarning, match="deprecated and ignored"):
        cfg = MigrationConfig()

    assert cfg.use_lmfit is True
    assert cfg.use_lmfit_lines is True


def test_enable_lmfit_gradually_is_compatibility_shim():
    old_backend = migration_config.use_lmfit
    try:
        migration_config.use_lmfit = False
        with pytest.warns(DeprecationWarning, match="deprecated"):
            assert migration_config.enable_lmfit_gradually() == "complete"
        assert migration_config.use_lmfit is True
    finally:
        migration_config.use_lmfit = old_backend


def test_legacy_linelist_file_can_still_be_written(sample_linelist, temp_output_dir):
    """Existing qsopar_log.fits workflows remain readable by the fitter."""
    hdr = fits.Header()
    hdr["lambda"] = "Vacuum Wavelength in Ang"
    hdu = fits.BinTableHDU(data=sample_linelist, header=hdr, name="data")
    linelist_path = os.path.join(temp_output_dir, "qsopar_log.fits")
    hdu.writeto(linelist_path, overwrite=True)

    assert os.path.exists(linelist_path)
    assert np.isfinite(fits.getdata(linelist_path, 1)["inisig"]).all()
