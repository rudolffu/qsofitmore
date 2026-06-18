"""Tests for bundled high-order Balmer products and continuum basis."""

import numpy as np
import pytest

from qsofitmore import neofit
from qsofitmore.neofit.templates.balmer import BalmerTemplateError, evaluate_balmer_series


def test_balmer_registry_contains_twelve_science_templates():
    templates = neofit.list_balmer_templates()

    assert len(templates) == 12
    assert sum(status == "production" for status in templates.values()) == 4
    assert not any("energy_only" in name for name in templates)


def test_balmer_template_preserves_builder_metadata_and_warnings():
    pure = neofit.load_balmer_template(log10_ne=9, n_min=6, provenance="sh95")
    extended = neofit.load_balmer_template(log10_ne=10, n_min=7, provenance="k13")

    assert pure.te_k == 15000.0
    assert pure.log10_ne == 9.0
    assert pure.n_min == 6
    assert pure.n_max == 50
    assert pure.wavelength_vacuum[0] == pytest.approx(4102.9350)
    assert not pure.warnings
    assert extended.n_max == 400
    assert "extrapolated_k13_full_eq2" in extended.row_sources
    assert extended.warnings[0].code == "balmer_high_n_extension_model_dependent"


def test_energy_only_balmer_template_is_rejected():
    with pytest.raises(BalmerTemplateError, match="diagnostic-only"):
        neofit.load_balmer_template(provenance="energy_only_ext")


def test_balmer_series_broadening_conserves_integrated_flux():
    template = neofit.load_balmer_template()
    wave = np.linspace(3400.0, 4300.0, 30000)
    basis = evaluate_balmer_series(template, wave, 3000.0)

    assert np.trapezoid(basis, wave) == pytest.approx(
        np.sum(template.rel_flux_hbeta), rel=3.0e-3
    )


def test_balmer_series_fwhm_is_velocity_broadened():
    template = neofit.load_balmer_template()
    wave = np.linspace(4050.0, 4155.0, 10000)
    basis = evaluate_balmer_series(template, wave, 3000.0)
    peak = int(np.argmax(basis))
    half = 0.5 * basis[peak]
    indices = np.where(basis >= half)[0]
    measured = (wave[indices[-1]] - wave[indices[0]]) / template.wavelength_vacuum[0] * 299792.458

    assert measured == pytest.approx(3000.0, rel=0.02)


def test_balmer_continuum_is_unit_edge_and_range_limited():
    wave = np.array([1900.0, 2000.0, 3000.0, 3646.0, 3700.0])
    basis = neofit.balmer_continuum_basis(wave)

    assert basis[0] == 0.0
    assert basis[-1] == 0.0
    assert basis[-2] == pytest.approx(1.0)
    assert np.all(basis >= 0.0)
