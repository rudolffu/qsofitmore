from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from qsofitmore import neofit
from qsofitmore.neofit.generic_complex import (
    GenericComplexContext,
    resolve_recipe_coverage,
)
from qsofitmore.neofit.templates import evaluate_balmer_series, load_balmer_template


def _continuum_config(balmer_series):
    return neofit.GlobalContinuumConfig(
        uv_iron=None,
        optical_iron=None,
        balmer_continuum=neofit.BalmerContinuumConfig(enabled=False),
        balmer_series=balmer_series,
        clip_passes=0,
    )


def _spectrum(lo=3300.0, hi=4500.0):
    wave = np.linspace(lo, hi, 1400)
    flux = 2.0 * (wave / 4000.0) ** -1.1
    if lo < 4260.0 and hi > 3500.0:
        flux += 12.0 * evaluate_balmer_series(
            load_balmer_template(), wave, 3200.0
        )
    return neofit.Spectrum.from_arrays(
        wave,
        flux,
        err=np.full_like(wave, 0.05),
        wave_frame="rest",
        survey="desi",
    )


def test_line_registry_resolves_vacuum_and_historical_aliases():
    assert neofit.lines.resolve("OIII5007") == "oiii_5008"
    assert neofit.lines.resolve("Hβ") == "hbeta"
    assert neofit.lines.get("oiii_5008").vacuum_wavelength == pytest.approx(
        5008.24
    )
    assert neofit.lines.get("pabeta").reference is not None
    with pytest.raises(ValueError, match="unknown_line_id"):
        neofit.lines.get("definitely-not-a-line")


def test_recipe_registry_is_immutable_and_overrides_by_copy():
    recipe = neofit.recipes.get("civ")
    changed = recipe.with_component(
        "CIV_blue", enabled=True, velocity_bounds_kms=(-8000.0, 0.0)
    )
    assert recipe.components[0].velocity_bounds_kms == (-5000.0, 3000.0)
    assert recipe.components[1].enabled is False
    assert changed.components[1].enabled is True
    assert changed.components[1].velocity_bounds_kms == (-8000.0, 0.0)
    with pytest.raises(FrozenInstanceError):
        recipe.label = "changed"
    with pytest.raises(ValueError, match="unknown_complex_recipe"):
        neofit.recipes.get("unknown")


def test_continuum_only_has_no_hbeta_or_aggregate_summary_verdict():
    result = neofit.fit_global_lines(
        _spectrum(),
        _continuum_config(neofit.BalmerSeriesConfig(amplitude=12.0)),
        complexes=[],
    )
    assert result.continuum_success
    assert result.hbeta is None
    assert result.hbeta_initial is None
    assert result.line_complexes == {}
    assert result.metadata["balmer_series_fwhm_source"] == "free_global_fit"
    assert result.metadata["hbeta_sync_requested"] is False
    assert "success" not in result.summary()
    assert "complete_success" not in result.summary()
    assert "legacy_hbeta_success" not in result.summary()


def test_hbeta_absent_auto_keeps_free_width_and_warns():
    result = neofit.fit_global_lines(
        _spectrum(),
        _continuum_config(neofit.BalmerSeriesConfig(amplitude=12.0)),
        complexes=None,
    )
    assert result.hbeta is None
    assert result.metadata["balmer_series_fwhm_source"] == "free_global_fit"
    assert result.metadata["hbeta_sync_attempted"] is False
    assert "hbeta_sync_skipped_not_covered" in result.warning_codes()
    assert "balmer_series_fwhm_free_no_hbeta_anchor" in result.warning_codes()


def test_never_and_fixed_balmer_width_policies_do_not_synchronize():
    never = neofit.fit_global_lines(
        _spectrum(),
        _continuum_config(
            neofit.BalmerSeriesConfig(
                amplitude=12.0, sync_with_hbeta="never"
            )
        ),
        complexes=[],
    )
    fixed = neofit.fit_global_lines(
        _spectrum(),
        _continuum_config(
            neofit.BalmerSeriesConfig(
                amplitude=12.0, fit_fwhm=False, fwhm_kms=4100.0
            )
        ),
        complexes=[],
    )
    assert never.metadata["hbeta_sync_requested"] is False
    assert never.metadata["balmer_series_fwhm_source"] == "free_global_fit"
    assert fixed.metadata["hbeta_sync_requested"] is False
    assert fixed.metadata["balmer_series_fwhm_source"] == "fixed_config"
    assert fixed.metadata["balmer_series_fwhm_kms"] == pytest.approx(4100.0)


def test_require_without_hbeta_warns_and_continues():
    result = neofit.fit_global_lines(
        _spectrum(),
        _continuum_config(
            neofit.BalmerSeriesConfig(
                amplitude=12.0, sync_with_hbeta="require"
            )
        ),
    )
    assert result.continuum_success
    assert "hbeta_sync_required_unmet" in result.warning_codes()


def test_component_adaptive_coverage_and_nir_blend_metadata():
    spectrum = _spectrum(10700.0, 11060.0)
    recipe = neofit.recipes.get("paschen_nir")
    coverage = resolve_recipe_coverage(spectrum, recipe)
    assert coverage.status == "partially_covered"
    assert set(coverage.active_component_ids) == {
        "HeI10833_broad",
        "Pagamma_broad",
    }
    assert "Padelta_broad" in coverage.disabled_component_ids


def test_generic_fixed_ratio_compilation():
    recipe = neofit.recipes.get("halpha_nii_sii")
    context = GenericComplexContext(
        recipe,
        tuple(component.id for component in recipe.components if component.enabled),
        100.0,
    )
    theta = context.initial.copy()
    wave = np.linspace(6500.0, 6630.0, 2000)
    components = context.components(theta, wave)
    ratio = np.trapezoid(components["NII6585"], wave) / np.trapezoid(
        components["NII6550"], wave
    )
    assert ratio == pytest.approx(2.96, rel=2.0e-3)


@pytest.mark.parametrize("profile", ["gaussian", "lorentzian"])
def test_generic_profile_derivatives_match_centered_differences(profile):
    recipe = neofit.recipes.get("civ")
    recipe = replace(
        recipe,
        components=(replace(recipe.components[0], profile=profile),),
    )
    context = GenericComplexContext(recipe, ("CIV_broad",), 50.0)
    nonlinear = np.asarray(
        [context.initial[context.index[name]] for name in context.nonlinear_names]
    )
    wave = np.linspace(1480.0, 1620.0, 500)
    design, derivatives = context.separable_design(nonlinear, wave, True)
    for index in range(nonlinear.size):
        step = max(abs(nonlinear[index]) * 1.0e-5, 1.0e-4)
        plus = nonlinear.copy()
        minus = nonlinear.copy()
        plus[index] += step
        minus[index] -= step
        plus_design, _ = context.separable_design(plus, wave, False)
        minus_design, _ = context.separable_design(minus, wave, False)
        finite_difference = (plus_design - minus_design) / (2.0 * step)
        assert derivatives[index] == pytest.approx(
            finite_difference, rel=3.0e-4, abs=1.0e-9
        )
