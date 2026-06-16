#!/usr/bin/env python
"""Tests for automatic line-table path, axis, and unit detection."""

import numpy as np
import os
import pytest
from astropy.io import fits
from astropy.table import Table

from qsofitmore import QSOFitNew
from qsofitmore.config import migration_config
from qsofitmore.line_params_io import csv_to_fits


LINE_DTYPE = [
    ("lambda", "f8"),
    ("compname", "U20"),
    ("minwav", "f8"),
    ("maxwav", "f8"),
    ("linename", "U20"),
    ("ngauss", "i2"),
    ("inisig", "f8"),
    ("minsig", "f8"),
    ("maxsig", "f8"),
    ("voff", "f8"),
    ("vindex", "i2"),
    ("windex", "i2"),
    ("findex", "i2"),
    ("fvalue", "f8"),
]


def _qso():
    wave = np.linspace(4900.0, 5100.0, 50)
    return QSOFitNew(wave, np.ones_like(wave), np.full_like(wave, 0.1), z=0.0)


def _line_table(sigmas=(1e-3, 2.3e-4, 1.7e-3, 0.01)):
    return np.rec.array(
        [(5008.24, "Hb", 4900.0, 5100.0, "OIII5007", 1, *sigmas, 0, 0, 0, 0.004)],
        dtype=LINE_DTYPE,
    )


def _write_fits(path, data=None, header=None):
    hdu = fits.BinTableHDU(data=data if data is not None else _line_table(), header=header, name="data")
    hdu.writeto(path, overwrite=True)


def test_line_table_metadata_controls_axis_and_units(tmp_path):
    header = fits.Header()
    header["WAVESCL"] = "linear"
    header["VELUNIT"] = "km/s"
    path = tmp_path / "custom.fits"
    _write_fits(path, data=_line_table((300.0, 70.0, 510.0, 1500.0)), header=header)

    config = _qso()._prepare_line_table_config(line_list_path=path)

    assert config.path == str(path)
    assert config.wave_scale == "linear"
    assert config.velocity_units == "km/s"


def test_legacy_log_table_without_metadata_falls_back_to_log_lnlambda(tmp_path):
    path = tmp_path / "custom.fits"
    _write_fits(path)

    config = _qso()._prepare_line_table_config(line_list_path=path)

    assert config.wave_scale == "log"
    assert config.velocity_units == "lnlambda"


def test_filename_infers_wave_scale(tmp_path):
    linear_path = tmp_path / "qsopar_linear.fits"
    log_path = tmp_path / "qsopar_log.fits"
    _write_fits(linear_path)
    _write_fits(log_path, data=_line_table((300.0, 70.0, 510.0, 1500.0)))

    assert _qso()._prepare_line_table_config(line_list_path=linear_path).wave_scale == "linear"
    assert _qso()._prepare_line_table_config(line_list_path=log_path).wave_scale == "log"


def test_value_heuristic_detects_linear_kms_without_metadata_or_filename(tmp_path):
    path = tmp_path / "custom.fits"
    _write_fits(path, data=_line_table((300.0, 70.0, 510.0, 1500.0)))

    config = _qso()._prepare_line_table_config(line_list_path=path)

    assert config.wave_scale == "linear"
    assert config.velocity_units == "km/s"


def test_auto_resolution_prefers_canonical_qsopar_and_warns(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    _write_fits(output / "qsopar.fits")
    _write_fits(output / "qsopar_linear.fits", data=_line_table((300.0, 70.0, 510.0, 1500.0)))

    q = _qso()
    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        with pytest.warns(UserWarning, match="Multiple line parameter files"):
            config = q._prepare_line_table_config()
    finally:
        os.chdir(old_cwd)

    assert config.path == "output/qsopar.fits"
    assert q.line_list_path == "output/qsopar.fits"


def test_csv_to_fits_writes_explicit_axis_metadata(tmp_path):
    csv_path = tmp_path / "qsopar.csv"
    fits_path = tmp_path / "qsopar.fits"
    table = Table(_line_table((300.0, 70.0, 510.0, 1500.0)))
    table.write(csv_path, format="csv")

    csv_to_fits(csv_path, fits_path, wave_scale="linear", velocity_units="km/s")

    with fits.open(fits_path) as hdul:
        assert hdul[1].header["WAVESCL"] == "linear"
        assert hdul[1].header["VELUNIT"] == "km/s"


def _run_line_fit_smoke(tmp_path, monkeypatch, path, amplitude):
    old_backend = migration_config.use_lmfit
    migration_config.use_lmfit = True
    try:
        wave = np.linspace(4900.0, 5100.0, 160)
        err = np.full_like(wave, 0.2)
        q = QSOFitNew(wave, np.ones_like(wave), err, z=0.0, path=str(tmp_path) + os.sep)
        q.MC = False
        q.n_trails = 0
        q.mask_compname = None
        q.tie_lambda = True
        q.tie_width = True
        q.tie_flux_1 = True
        q.tie_flux_2 = True
        q.poly = False
        q.BC = False
        q.broken_pl = False
        q.wave = wave
        q.conti_fit = type(
            "Fit",
            (),
            {"params": np.array([0., 3000., 0., 0., 3000., 0., 1., -1.5, 0., 5000., 0., 0., 0., 0.])},
        )()
        monkeypatch.setattr(q, "F_poly_conti", lambda x, pp: np.zeros_like(x))
        monkeypatch.setattr(q, "Balmer_conti", lambda x, pp: np.zeros_like(x))
        monkeypatch.setattr(q, "Balmer_high_order", lambda x, pp: np.zeros_like(x))

        config = q._prepare_line_table_config(line_list_path=path)
        center = 5008.24
        sigma_value = 300.0 if config.velocity_units == "km/s" else 1e-3
        sigma_axis = q._sigma_axis_units(sigma_value, center)
        center_axis = q._x_axis(np.array([center]))[0]
        line_flux = q.Manygauss(q._x_axis(wave), [amplitude, center_axis, sigma_axis])
        q.flux = line_flux + 1.0
        q.sn_obs = q.flux / err

        q._DoLineFit(wave, line_flux, err, q.conti_fit)

        assert q.line_result.size > 0
        assert any("Hb_line_status" in str(name) for name in q.line_result_name)
        return config
    finally:
        migration_config.use_lmfit = old_backend


def test_lmfit_line_fit_smoke_with_legacy_log_lnlambda_table(tmp_path, monkeypatch):
    path = tmp_path / "qsopar_log.fits"
    _write_fits(path)

    config = _run_line_fit_smoke(tmp_path, monkeypatch, path, amplitude=0.03)

    assert config.wave_scale == "log"
    assert config.velocity_units == "lnlambda"


def test_lmfit_line_fit_smoke_with_linear_kms_table(tmp_path, monkeypatch):
    header = fits.Header()
    header["WAVESCL"] = "linear"
    header["VELUNIT"] = "km/s"
    path = tmp_path / "qsopar.fits"
    _write_fits(path, data=_line_table((300.0, 70.0, 510.0, 1500.0)), header=header)

    config = _run_line_fit_smoke(tmp_path, monkeypatch, path, amplitude=20.0)

    assert config.wave_scale == "linear"
    assert config.velocity_units == "km/s"


def test_lmfit_line_fit_smoke_with_explicit_linear_csv_table(tmp_path, monkeypatch):
    path = tmp_path / "qsopar_linear.csv"
    Table(_line_table((300.0, 70.0, 510.0, 1500.0))).write(path, format="csv")

    config = _run_line_fit_smoke(tmp_path, monkeypatch, path, amplitude=20.0)

    assert config.path == str(path)
    assert config.wave_scale == "linear"
    assert config.velocity_units == "km/s"
