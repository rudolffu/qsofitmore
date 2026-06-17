"""Tests for neofit independent local fitting mode."""

import numpy as np

from qsofitmore import neofit


def _two_window_spectrum():
    wave = np.linspace(4500.0, 6900.0, 900)
    hb = 6.0 * np.exp(-0.5 * ((wave - 4861.33) / 22.0) ** 2)
    ha = 9.0 * np.exp(-0.5 * ((wave - 6562.8) / 35.0) ** 2)
    flux = 1.0 + 0.0002 * (wave - 5500.0) + hb + ha
    err = np.full_like(wave, 0.08)
    return neofit.Spectrum.from_arrays(wave, flux, err=err, z=0.0, survey="desi")


def test_fit_local_two_independent_windows():
    spec = _two_window_spectrum()
    config = neofit.LocalFitConfig(windows=[neofit.recipes.local_hbeta(), neofit.recipes.local_halpha()])

    result = neofit.fit_local(spec, config)
    table = result.to_table()

    assert result.success
    assert result.window_results["Hb_OIII"].success
    assert result.window_results["Ha_NII_SII"].success
    assert set(table["window"]) == {"Hb_OIII", "Ha_NII_SII"}
    assert result.summary()["n_success"] == 2


def test_fit_local_window_outside_coverage_fails_without_crashing():
    spec = _two_window_spectrum()
    outside = neofit.recipes.local_mgii()
    config = neofit.LocalFitConfig(windows=[outside, neofit.recipes.local_hbeta()])

    result = neofit.fit_local(spec, config)

    assert result.success
    assert not result.window_results["MgII"].success
    assert result.window_results["Hb_OIII"].success
    assert "window_not_covered" in result.warning_codes()


def test_fit_local_too_few_pixels_fails_window():
    wave = np.array([4800.0, 4820.0, 4840.0, 4860.0, 4880.0])
    flux = np.ones_like(wave)
    err = np.ones_like(wave) * 0.1
    spec = neofit.Spectrum.from_arrays(wave, flux, err=err, z=0.0)
    config = neofit.LocalFitConfig(windows=[neofit.recipes.local_hbeta()], require_min_pixels=8)

    result = neofit.fit_local(spec, config)

    assert not result.success
    assert not result.window_results["Hb_OIII"].success
    assert "window_too_few_pixels" in result.warning_codes()
