"""Lorentzian local-profile tests."""

import numpy as np
import pytest

from qsofitmore import neofit
from qsofitmore.neofit.models.lorentzian import lorentzian, lorentzian_partials


def test_lorentzian_partials_match_finite_difference():
    wave = np.linspace(4800.0, 4920.0, 200)
    params = np.array([5.0, 4862.0, 18.0])
    analytic = lorentzian_partials(wave, *params)
    numeric = np.empty_like(analytic)
    for index in range(3):
        step = 1.0e-5 * max(abs(params[index]), 1.0)
        plus = params.copy()
        minus = params.copy()
        plus[index] += step
        minus[index] -= step
        numeric[:, index] = (lorentzian(wave, *plus) - lorentzian(wave, *minus)) / (2.0 * step)

    assert np.allclose(analytic, numeric, rtol=2.0e-5, atol=2.0e-7)


def test_local_hbeta_lorentzian_recovers_profile_and_measurements():
    wave = np.linspace(4700.0, 5100.0, 600)
    flux = 1.2 + lorentzian(wave, 4.0, 4863.0, 15.0)
    err = np.full_like(wave, 0.03)
    spectrum = neofit.Spectrum.from_arrays(wave, flux, err=err, wave_frame="rest")

    result = neofit.fit_line_complex(
        spectrum,
        neofit.recipes.local_hbeta(profile="lorentzian"),
    )
    row = result.to_table().query("name == 'Hb_broad'").iloc[0]

    assert result.success
    assert abs(result.param_values["Hb_broad.center"] - 4863.0) < 0.2
    assert abs(result.param_values["Hb_broad.gamma"] - 15.0) < 0.3
    assert row["component_type"] == "lorentzian"
    assert row["fwhm"] == pytest.approx(30.0, rel=0.03)
    assert row["line_flux_input"] == pytest.approx(np.pi * 4.0 * 15.0, rel=0.03)
