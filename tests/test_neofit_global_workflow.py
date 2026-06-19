"""Synthetic tests for the global continuum and H-beta milestone."""

import numpy as np
import pytest

from qsofitmore import neofit
from qsofitmore.neofit.global_fit import (
    C_KMS,
    FWHM_TO_SIGMA,
    _gaussian_area_profile,
    balmer_continuum_basis,
)
from qsofitmore.neofit.global_result import GlobalContinuumResult
from qsofitmore.neofit.templates import evaluate_balmer_series, load_balmer_template, load_iron_template
from qsofitmore.neofit.templates.iron import evaluate_iron_basis


def _continuum_result(spectrum, model):
    return GlobalContinuumResult(
        success=True,
        status=1,
        message="known",
        param_values={},
        param_errors={},
        covariance=None,
        chi2=0.0,
        dof=1,
        reduced_chi2=0.0,
        wave_rest=spectrum.wave_rest.copy(),
        model=model.copy(),
        component_models={"power_law": model.copy()},
        fit_mask=spectrum.valid_mask.copy(),
        clip_mask=spectrum.valid_mask.copy(),
    )


def test_global_continuum_recovers_full_synthetic_model():
    wave = np.linspace(1900.0, 5600.0, 3200)
    power = 2.5 * (wave / 3000.0) ** -1.25
    uv = 70.0 * evaluate_iron_basis(load_iron_template("vw01"), wave, 2600.0)
    optical = 55.0 * evaluate_iron_basis(load_iron_template("park22"), wave, 3200.0)
    bc = 0.35 * balmer_continuum_basis(wave)
    series = 8.0 * evaluate_balmer_series(load_balmer_template(), wave, 2800.0)
    err = np.full_like(wave, 0.03)
    spectrum = neofit.Spectrum.from_arrays(
        wave, power + uv + optical + bc + series, err=err, wave_frame="rest", survey="desi"
    )
    config = neofit.GlobalContinuumConfig(
        power_law=neofit.PowerLawConfig(norm=2.5, slope=-1.25),
        uv_iron=neofit.IronTemplateConfig.vw01(fwhm_kms=2600.0, amp=70.0),
        optical_iron=neofit.IronTemplateConfig.park22(fwhm_kms=3200.0, amp=55.0),
        balmer_continuum=neofit.BalmerContinuumConfig(amplitude=0.35),
        balmer_series=neofit.BalmerSeriesConfig(amplitude=8.0, fwhm_kms=2800.0),
        clip_passes=0,
    )

    result = neofit.fit_global_continuum(spectrum, config)

    assert result.success
    assert result.param_values["power_law.norm"] == pytest.approx(2.5, rel=0.01)
    assert result.param_values["power_law.slope"] == pytest.approx(-1.25, abs=0.02)
    assert set(result.component_models) == {
        "power_law",
        "uv_iron",
        "optical_iron",
        "balmer_continuum",
        "balmer_series",
    }
    assert np.nanmax(np.abs(result.model - spectrum.flux)) < 0.02


def test_global_continuum_disables_uncovered_components():
    wave = np.linspace(4700.0, 5500.0, 600)
    flux = 2.0 * (wave / 3000.0) ** -1.2
    spectrum = neofit.Spectrum.from_arrays(wave, flux, err=np.full_like(wave, 0.05), wave_frame="rest")

    result = neofit.fit_global_continuum(spectrum)

    assert result.success
    assert "uv_iron" not in result.component_models
    assert "balmer_continuum" not in result.component_models
    assert "balmer_series" not in result.component_models
    assert "global_component_disabled_no_coverage" in result.warning_codes()


def test_hbeta_core_ties_oiii_ratio_and_narrow_kinematics():
    wave = np.linspace(4600.0, 5120.0, 1800)
    continuum = np.full_like(wave, 2.0)
    broad = _gaussian_area_profile(wave, 120.0, 4862.68, 2200.0)
    narrow = _gaussian_area_profile(wave, 12.0, 4862.68, 350.0)
    oiii5007 = _gaussian_area_profile(wave, 45.0, 5008.24, 350.0)
    oiii4959 = _gaussian_area_profile(wave, 45.0 / 2.98, 4960.30, 350.0)
    err = np.full_like(wave, 0.025)
    spectrum = neofit.Spectrum.from_arrays(
        wave, continuum + broad + narrow + oiii5007 + oiii4959, err=err, wave_frame="rest", survey="desi"
    )

    result = neofit.fit_hbeta_complex(
        spectrum,
        _continuum_result(spectrum, continuum),
        neofit.HbetaComplexConfig(fit_oiii_wings=False),
    )

    assert result.success
    assert result.selected_model == "core"
    assert result.param_values["OIII5007_core.flux"] > 0
    assert np.trapezoid(result.component_models["OIII5007_core"], wave) / np.trapezoid(
        result.component_models["OIII4959_core"], wave
    ) == pytest.approx(2.98, rel=1.0e-3)
    assert result.metrics["Hb_broad_fwhm_kms"] == pytest.approx(2200.0, rel=0.15)
    assert result.metrics["Hb_broad_flux_cgs"] == pytest.approx(
        result.metrics["Hb_broad_flux_input"] * 1.0e-17, rel=1.0e-6
    )


def test_hbeta_wing_selection_accepts_strong_broad_wing():
    wave = np.linspace(4600.0, 5120.0, 1800)
    continuum = np.full_like(wave, 1.5)
    line = _gaussian_area_profile(wave, 100.0, 4862.68, 2500.0)
    line += _gaussian_area_profile(wave, 35.0, 5008.24, 320.0)
    line += _gaussian_area_profile(wave, 35.0 / 2.98, 4960.30, 320.0)
    wing_velocity = -350.0
    line += _gaussian_area_profile(
        wave, 30.0, 5008.24 * np.exp(wing_velocity / C_KMS), 1100.0
    )
    line += _gaussian_area_profile(
        wave, 30.0 / 2.98, 4960.30 * np.exp(wing_velocity / C_KMS), 1100.0
    )
    err = np.full_like(wave, 0.02)
    spectrum = neofit.Spectrum.from_arrays(wave, continuum + line, err=err, wave_frame="rest")

    result = neofit.fit_hbeta_complex(spectrum, _continuum_result(spectrum, continuum))

    assert result.success
    assert result.selected_model == "wing"
    assert result.metadata["wing_candidate"]["accepted"]


def test_hbeta_wing_selection_rejects_absent_wing_and_can_fit_heii():
    wave = np.linspace(4600.0, 5120.0, 1800)
    continuum = np.full_like(wave, 1.5)
    line = _gaussian_area_profile(wave, 90.0, 4862.68, 2200.0)
    line += _gaussian_area_profile(wave, 30.0, 5008.24, 350.0)
    line += _gaussian_area_profile(wave, 30.0 / 2.98, 4960.30, 350.0)
    line += _gaussian_area_profile(wave, 12.0, 4687.02, 1800.0)
    err = np.full_like(wave, 0.02)
    spectrum = neofit.Spectrum.from_arrays(wave, continuum + line, err=err, wave_frame="rest")

    result = neofit.fit_hbeta_complex(
        spectrum,
        _continuum_result(spectrum, continuum),
        neofit.HbetaComplexConfig(heii_enabled=True),
    )

    assert result.success
    assert result.selected_model == "core"
    assert "oiii_wing_rejected" in result.warning_codes()
    assert result.param_values["HeII_broad.flux"] > 0


def test_global_workflow_refines_balmer_width_and_writes_products(tmp_path):
    wave = np.linspace(2600.0, 7100.0, 3200)
    continuum = 2.0 * (wave / 3000.0) ** -1.2
    line = _gaussian_area_profile(wave, 100.0, 4862.68, 2800.0)
    err = np.full_like(wave, 0.04)
    spectrum = neofit.Spectrum.from_arrays(
        wave, continuum + line, err=err, wave_frame="rest", survey="desi"
    )
    result = neofit.fit_global_hbeta(
        spectrum,
        neofit.GlobalContinuumConfig(
            uv_iron=None,
            optical_iron=None,
            balmer_continuum=neofit.BalmerContinuumConfig(enabled=False),
            balmer_series=neofit.BalmerSeriesConfig(enabled=False),
        ),
        neofit.HbetaComplexConfig(fit_oiii_wings=False),
    )

    files = neofit.write_global_hbeta_products(
        result,
        str(tmp_path),
        neofit.GlobalQAPlotConfig(write_other_diagnostics=True),
    )

    assert result.legacy_hbeta_success
    assert set(files) >= {
        "summary_json",
        "continuum_measurements_csv",
        "hbeta_measurements_csv",
        "full_grid_csv",
        "global_plot",
        "balmer_edge_plot",
        "hbeta_plot",
    }


def test_balmer_series_width_converges_to_summed_broad_hbeta_fwhm():
    wave = np.linspace(3300.0, 5200.0, 2200)
    continuum = 2.0 * (wave / 3000.0) ** -1.2
    series = 25.0 * evaluate_balmer_series(load_balmer_template(), wave, 2800.0)
    line = _gaussian_area_profile(wave, 100.0, 4862.68, 2800.0)
    err = np.full_like(wave, 0.04)
    spectrum = neofit.Spectrum.from_arrays(
        wave, continuum + series + line, err=err, wave_frame="rest", survey="desi"
    )
    result = neofit.fit_global_hbeta(
        spectrum,
        neofit.GlobalContinuumConfig(
            uv_iron=None,
            optical_iron=None,
            balmer_continuum=neofit.BalmerContinuumConfig(enabled=False),
            balmer_series=neofit.BalmerSeriesConfig(amplitude=25.0, fwhm_kms=2600.0),
        ),
        neofit.HbetaComplexConfig(fit_oiii_wings=False),
    )

    assert result.legacy_hbeta_success
    assert result.metadata["hbeta_sync_converged"]
    assert abs(
        result.continuum.metadata["balmer_series_fwhm_kms"]
        - result.hbeta.metrics["Hb_broad_fwhm_kms"]
    ) <= result.metadata["balmer_width_sync_tolerance_kms"]


def test_global_workflow_monte_carlo_reports_percentiles():
    wave = np.linspace(2700.0, 5500.0, 1000)
    continuum = 2.0 * (wave / 3000.0) ** -1.2
    line = _gaussian_area_profile(wave, 80.0, 4862.68, 2400.0)
    err = np.full_like(wave, 0.05)
    spectrum = neofit.Spectrum.from_arrays(wave, continuum + line, err=err, wave_frame="rest")
    config = neofit.GlobalContinuumConfig(
        uv_iron=None,
        optical_iron=None,
        balmer_continuum=neofit.BalmerContinuumConfig(enabled=False),
        balmer_series=neofit.BalmerSeriesConfig(enabled=False),
    )

    result = neofit.fit_global_hbeta(
        spectrum,
        config,
        neofit.HbetaComplexConfig(fit_oiii_wings=False),
        neofit.UncertaintyConfig(monte_carlo_trials=2, random_seed=8),
    )

    assert result.monte_carlo["n_requested"] == 2
    assert result.monte_carlo["continuum_success_count"] == 2
    assert result.monte_carlo["complex_success_counts"]["hbeta_oiii"] == 2
    assert "Hb_broad_fwhm_kms" in result.monte_carlo["percentiles"]
