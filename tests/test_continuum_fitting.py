#!/usr/bin/env python
"""Tests for continuum optimizer backend behavior."""

import numpy as np
import pytest

from qsofitmore import QSOFitNew
from qsofitmore.config import migration_config
import qsofitmore.fitmodule as fitmodule


def _basic_qso():
    wave = np.linspace(4200.0, 5200.0, 120)
    flux = 2.0 * (wave / 3000.0) ** -1.2
    err = np.full_like(wave, 0.1)
    q = QSOFitNew(wave, flux, err, z=0.0)
    q.include_iron = False
    q.BC = False
    q.poly = False
    q.broken_pl = False
    q.iron_temp_name = "BG92-VW01"
    return q, wave, flux, err


def test_lmfit_continuum_helper_runs_without_kapteyn(monkeypatch):
    """The default lmfit path should not need the optional Kapteyn backend."""
    q, wave, flux, err = _basic_qso()
    monkeypatch.setattr(fitmodule, "_kmpfit", None)
    monkeypatch.setattr(q, "Fe_flux_mgii", lambda x, pp: np.zeros_like(x))
    monkeypatch.setattr(q, "Fe_flux_balmer", lambda x, pp: np.zeros_like(x))
    monkeypatch.setattr(q, "Fe_flux_verner", lambda x, pp: np.zeros_like(x))
    monkeypatch.setattr(q, "Fe_flux_g12", lambda x, pp: np.zeros_like(x))
    monkeypatch.setattr(q, "Balmer_conti", lambda x, pp: np.zeros_like(x))
    monkeypatch.setattr(q, "Balmer_high_order", lambda x, pp: np.zeros_like(x))

    pp0 = np.array([0., 3000., 0., 0., 3000., 0., 1., -1.0, 0., 5000., 0., 0., 0., 0.])
    bounds = [None] * len(pp0)
    result = q._fit_continuum_lmfit(wave, flux, err, pp0, bounds)

    assert result.success
    assert len(result.params) == len(pp0)
    assert np.all(np.isfinite(result.params))


def test_use_lmfit_false_selects_legacy_backend_and_requires_kapteyn(monkeypatch):
    """The single public backend switch should route all fitting to kmpfit."""
    q, wave, flux, err = _basic_qso()
    old_backend = migration_config.use_lmfit
    monkeypatch.setattr(fitmodule, "_kmpfit", None)
    migration_config.use_lmfit = False
    q.initial_guess = None
    q.Fe_flux_range = None
    q.MC = False
    q.n_trails = 0
    q.flux_prereduced = flux
    q.err_prereduced = err
    try:
        with pytest.raises(ImportError, match="kapteyn is required"):
            q._DoContiFit(wave, flux, err, 0.0, 0.0, 0, 0, 0)
    finally:
        migration_config.use_lmfit = old_backend
