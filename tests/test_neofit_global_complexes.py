"""Tests for Mg II/H-alpha global line complexes and QA products."""

from dataclasses import replace

import numpy as np
import pytest

from qsofitmore import neofit
from qsofitmore.host_decomp.io import SpectrumData
from qsofitmore.neofit import global_fit, global_io, host_workflow
from qsofitmore.neofit.global_fit import (
    C_KMS,
    _HalphaContext,
    _MgIIContext,
    _gaussian_area_profile,
)
from qsofitmore.neofit.global_io import (
    _BROAD_COMPONENT_STYLE,
    _COMBINED_BROAD_STYLE,
    _CONTINUUM_STYLES,
    _NARROW_STYLE,
    _WING_STYLE,
    _annotate_emission_lines,
    _configure_qa_axis,
    _flux_density_axis_label,
    _line_groups,
    _host_fraction_annotation,
    _has_host_context,
    _masked_running_median,
    _percentile_limits,
    _plot_qa,
    _qa_overview_title,
    _rounded_model_upper_limit,
    _select_zoom_complexes,
)
from qsofitmore.neofit.global_result import GlobalContinuumResult


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


def _centered_difference(function, value, step=0.01):
    return (function(value + step) - function(value - step)) / (2.0 * step)


@pytest.mark.parametrize(
    "context",
    [
        _MgIIContext(neofit.MgIIComplexConfig(), 100.0),
        _HalphaContext(neofit.HalphaComplexConfig(), 150.0),
    ],
)
def test_optional_complex_design_derivatives_match_centered_differences(context):
    wave = (
        np.linspace(2700.0, 2900.0, 700)
        if isinstance(context, _MgIIContext)
        else np.linspace(6400.0, 6800.0, 900)
    )
    _, _, nonlinear, _ = context.separable_initial_and_bounds()
    design, derivatives = context.separable_design(nonlinear, wave, True)
    assert design.shape[1] == len(context.linear_names)
    for index, derivative in enumerate(derivatives):
        def evaluate(value):
            trial = nonlinear.copy()
            trial[index] = value
            return context.separable_design(trial, wave, False)[0]

        finite = _centered_difference(evaluate, nonlinear[index])
        assert derivative == pytest.approx(finite, rel=5.0e-5, abs=1.0e-11)


def test_mgii_recovers_two_broad_components_and_metrics():
    wave = np.linspace(2680.0, 2920.0, 1200)
    continuum = np.full_like(wave, 2.0)
    line = _gaussian_area_profile(
        wave, 70.0, 2798.75 * np.exp(-100.0 / C_KMS), 2200.0
    )
    line += _gaussian_area_profile(
        wave, 30.0, 2798.75 * np.exp(200.0 / C_KMS), 7000.0
    )
    spectrum = neofit.Spectrum.from_arrays(
        wave,
        continuum + line,
        err=np.full_like(wave, 0.02),
        wave_frame="rest",
        survey="desi",
    )
    result = neofit.fit_mgii_complex(
        spectrum, _continuum_result(spectrum, continuum)
    )

    assert result.success
    assert result.metadata["optimizer_used"] == "variable_projection"
    assert result.param_values["MgII_broad1.flux"] == pytest.approx(70.0, rel=1.0e-3)
    assert result.param_values["MgII_broad2.flux"] == pytest.approx(30.0, rel=1.0e-3)
    assert result.metrics["MgII_broad_flux_input"] == pytest.approx(100.0, rel=1.0e-3)
    assert result.covariance.shape == (9, 9)


def test_halpha_recovers_tied_narrow_lines_and_fixed_nii_ratio():
    wave = np.linspace(6350.0, 6850.0, 1800)
    continuum = np.full_like(wave, 1.5)
    line = _gaussian_area_profile(wave, 80.0, 6564.61, 2200.0)
    line += _gaussian_area_profile(wave, 30.0, 6564.61, 4200.0)
    line += _gaussian_area_profile(wave, 10.0, 6564.61, 9000.0)
    for flux, center in (
        (12.0, 6564.61),
        (25.0, 6585.28),
        (25.0 / 2.96, 6549.85),
        (8.0, 6718.29),
        (6.0, 6732.67),
    ):
        line += _gaussian_area_profile(wave, flux, center, 320.0)
    spectrum = neofit.Spectrum.from_arrays(
        wave,
        continuum + line,
        err=np.full_like(wave, 0.02),
        wave_frame="rest",
        survey="desi",
    )
    result = neofit.fit_halpha_complex(
        spectrum, _continuum_result(spectrum, continuum)
    )

    assert result.success
    assert result.metrics["Ha_broad_flux_input"] == pytest.approx(120.0, rel=1.0e-3)
    assert result.metrics["Ha_narrow_flux_input"] == pytest.approx(12.0, rel=1.0e-3)
    assert result.metrics["NII6585_flux_input"] / result.metrics[
        "NII6549_flux_input"
    ] == pytest.approx(2.96, rel=1.0e-8)
    assert result.metrics["SII6718_flux_input"] == pytest.approx(8.0, rel=1.0e-3)
    assert result.metrics["SII6733_flux_input"] == pytest.approx(6.0, rel=1.0e-3)
    centers = {
        name: np.trapezoid(wave * component, wave) / np.trapezoid(component, wave)
        for name, component in result.component_models.items()
        if name in {"Ha_narrow", "NII6549", "NII6585", "SII6718", "SII6733"}
    }
    velocities = {
        name: np.log(
            centers[name]
            / {
                "Ha_narrow": 6564.61,
                "NII6549": 6549.85,
                "NII6585": 6585.28,
                "SII6718": 6718.29,
                "SII6733": 6732.67,
            }[name]
        )
        * C_KMS
        for name in centers
    }
    assert max(velocities.values()) - min(velocities.values()) < 0.1


@pytest.mark.parametrize(
    ("wave_range", "covered"),
    [
        ((2700.0, 2900.0), True),
        ((2740.0, 2860.0), False),
        ((2800.0, 3000.0), False),
    ],
)
def test_mgii_coverage_rules(wave_range, covered):
    wave = np.linspace(*wave_range, 300)
    continuum = np.ones_like(wave)
    spectrum = neofit.Spectrum.from_arrays(
        wave, continuum, err=np.full_like(wave, 0.05), wave_frame="rest"
    )
    result = neofit.fit_mgii_complex(
        spectrum, _continuum_result(spectrum, continuum)
    )
    assert result.success is covered
    assert ("line_complex_not_covered" in result.warning_codes()) is (not covered)


def test_variable_projection_matches_legacy_for_optional_complexes():
    wave = np.linspace(6350.0, 6850.0, 1500)
    continuum = np.full_like(wave, 1.5)
    line = _gaussian_area_profile(wave, 100.0, 6564.61, 2400.0)
    line += _gaussian_area_profile(wave, 15.0, 6564.61, 350.0)
    line += _gaussian_area_profile(wave, 20.0, 6585.28, 350.0)
    line += _gaussian_area_profile(wave, 20.0 / 2.96, 6549.85, 350.0)
    line += _gaussian_area_profile(wave, 7.0, 6718.29, 350.0)
    line += _gaussian_area_profile(wave, 6.0, 6732.67, 350.0)
    spectrum = neofit.Spectrum.from_arrays(
        wave, continuum + line, err=np.full_like(wave, 0.02), wave_frame="rest"
    )
    known = _continuum_result(spectrum, continuum)
    optimized = neofit.fit_halpha_complex(
        spectrum,
        known,
        neofit.HalphaComplexConfig(optimizer_method="variable_projection"),
    )
    legacy = neofit.fit_halpha_complex(
        spectrum,
        known,
        neofit.HalphaComplexConfig(optimizer_method="legacy_joint"),
    )
    assert optimized.chi2 <= legacy.chi2 + 1.0e-4
    assert optimized.metrics["Ha_broad_flux_input"] == pytest.approx(
        legacy.metrics["Ha_broad_flux_input"], rel=5.0e-3
    )
    assert optimized.metrics["Ha_broad_fwhm_kms"] == pytest.approx(
        legacy.metrics["Ha_broad_fwhm_kms"], abs=5.0
    )


def _global_spectrum(wave):
    continuum = 2.0 * (wave / 3000.0) ** -1.2
    line = _gaussian_area_profile(wave, 60.0, 2798.75, 2500.0)
    line += _gaussian_area_profile(wave, 80.0, 4862.68, 2500.0)
    line += _gaussian_area_profile(wave, 100.0, 6564.61, 2500.0)
    return neofit.Spectrum.from_arrays(
        wave,
        continuum + line,
        err=np.full_like(wave, 0.05),
        wave_frame="rest",
        survey="desi",
    )


def _simple_global_config():
    return neofit.GlobalContinuumConfig(
        uv_iron=None,
        optical_iron=None,
        balmer_continuum=neofit.BalmerContinuumConfig(enabled=False),
        balmer_series=neofit.BalmerSeriesConfig(enabled=False),
        clip_passes=0,
    )


def test_global_workflow_fits_only_covered_complexes_and_writes_qa(tmp_path):
    full = neofit.fit_global_lines(
        _global_spectrum(np.linspace(2600.0, 7000.0, 3000)),
        _simple_global_config(),
        neofit.HbetaComplexConfig(fit_oiii_wings=False),
    )
    partial = neofit.fit_global_lines(
        _global_spectrum(np.linspace(1875.0, 5130.0, 2500)),
        _simple_global_config(),
        neofit.HbetaComplexConfig(fit_oiii_wings=False),
    )
    full.metadata.update({"object_id": "synthetic-qa", "redshift": 1.23456})
    full.host_decomp_enabled = True
    files = neofit.write_global_line_products(full, str(tmp_path))

    assert set(full.line_complexes) == {
        "mgii",
        "hbeta_oiii",
        "halpha_nii_sii",
        "oii_nev_neiii_hgamma",
    }
    assert set(partial.line_complexes) == {
        "mgii",
        "hbeta_oiii",
        "oii_nev_neiii_hgamma",
    }
    assert partial.halpha is None
    assert partial.complex_statuses["halpha_nii_sii"] == "not_covered"
    assert full.continuum_success
    assert all(fit.success for fit in full.line_complexes.values())
    assert files["qa_plot"] == files["global_plot"]
    assert "host_context_plot" not in files
    assert full.metadata["host_context_plot_created"] is False
    assert full.metadata["qa_panel_count"] == 5
    assert full.metadata["qa_percentiles"] == [1.0, 99.8]
    assert full.metadata["qa_layout"] == "overview_top_complexes_bottom"
    assert full.metadata["qa_figure_size_inches"] == [10.5, 6.2]
    assert full.metadata["qa_displayed_complexes"] == [
        "mgii",
        "oii_nev_neiii_hgamma",
        "hbeta_oiii",
        "halpha_nii_sii",
    ]
    assert full.metadata["qa_omitted_complexes"] == []
    assert full.metadata["qa_smoothed_data"] is False
    assert full.metadata["qa_minor_ticks"] is True
    assert full.metadata["qa_tick_direction"] == "in"
    assert full.metadata["qa_overview_title"] == (
        "DESI TARGETID synthetic-qa — z=1.2346"
    )
    assert full.metadata["qa_overview_annotation"] == {
        "continuum_reduced_chi2": full.continuum.reduced_chi2,
        "host_state": "decomposed with pPXF",
        "host_fractions": "",
    }
    assert full.metadata["qa_overview_xlim"] == pytest.approx(
        [full.spectrum.wave_rest.min(), full.spectrum.wave_rest.max()]
    )
    assert full.metadata["qa_overview_ymin"] == 0.0
    assert full.metadata["qa_overview_model_upper_limit"] >= np.max(
        full.continuum.model
        + sum(
            (fit.model for fit in full.line_complexes.values()),
            np.zeros_like(full.continuum.model),
        )
    )
    assert set(full.metadata["qa_zoom_model_upper_limits"]) == {
        "mgii",
        "oii_nev_neiii_hgamma",
        "hbeta_oiii",
        "halpha_nii_sii",
    }
    for complex_name, upper_limit in full.metadata[
        "qa_zoom_model_upper_limits"
    ].items():
        lo, hi = {
            "mgii": (2700.0, 2900.0),
            "oii_nev_neiii_hgamma": (3380.0, 4425.0),
            "hbeta_oiii": (4640.0, 5100.0),
            "halpha_nii_sii": (6400.0, 6800.0),
        }[complex_name]
        mask = (
            full.spectrum.valid_mask
            & (full.spectrum.wave_rest >= lo)
            & (full.spectrum.wave_rest <= hi)
        )
        complex_model = full.continuum.model + sum(
            (fit.model for fit in full.line_complexes.values()),
            np.zeros_like(full.continuum.model),
        )
        assert upper_limit >= np.max(complex_model[mask])
    assert set(full.metadata["qa_zoom_ymin"].values()) == {0.0}
    for name in (
        "mgii",
        "oii_nev_neiii_hgamma",
        "hbeta_oiii",
        "halpha_nii_sii",
    ):
        assert (
            f"{full.line_complexes[name].reduced_chi2:.2f}"
            in full.metadata["qa_zoom_titles"][name]
        )
    assert set(full.metadata["qa_zoom_line_labels"]["mgii"]) == {"Mg II"}
    assert set(full.metadata["qa_zoom_line_labels"]["hbeta_oiii"]) == {
        "Hβ",
        "[O III] 4960",
        "[O III] 5008",
    }
    assert set(full.metadata["qa_zoom_line_labels"]["halpha_nii_sii"]) == {
        "[N II] 6550",
        "Hα",
        "[N II] 6585",
        "[S II] 6718",
        "[S II] 6733",
    }
    assert set(full.metadata["qa_major_emission_line_labels"]) == {
        "Mg II",
        "[O II] 3728",
        r"H$\gamma$",
        r"H$\beta$",
        "[O III] 5008",
        r"H$\alpha$",
    }
    assert {
        "mgii_measurements_csv",
        "halpha_nii_sii_measurements_csv",
    } <= set(files)
    assert files["summary_json"].endswith("neofit_global_lines_summary.json")
    assert files["compatibility_summary_json"].endswith(
        "neofit_global_hbeta_summary.json"
    )
    compatibility = neofit.write_global_hbeta_products(
        full,
        str(tmp_path / "compatibility"),
        qa_plot_config=neofit.GlobalQAPlotConfig(show_smoothed_data=True),
    )
    assert full.metadata["qa_smoothed_data"] is True
    assert compatibility["summary_json"].endswith(
        "neofit_global_hbeta_summary.json"
    )
    assert compatibility["generic_summary_json"].endswith(
        "neofit_global_lines_summary.json"
    )


def test_optional_fit_failure_preserves_legacy_success(monkeypatch):
    spectrum = _global_spectrum(np.linspace(2600.0, 7000.0, 2500))

    def fail(*args, **kwargs):
        raise RuntimeError("forced")

    monkeypatch.setattr(global_fit, "fit_halpha_complex", fail)
    result = neofit.fit_global_lines(
        spectrum,
        _simple_global_config(),
        neofit.HbetaComplexConfig(fit_oiii_wings=False),
    )
    assert result.continuum_success
    assert result.legacy_hbeta_success
    assert result.halpha is not None
    assert not result.halpha.success
    assert "optional_line_fit_failed" in result.warning_codes()


def test_global_monte_carlo_includes_covered_optional_complexes():
    result = neofit.fit_global_lines(
        _global_spectrum(np.linspace(2600.0, 7000.0, 1800)),
        _simple_global_config(),
        neofit.HbetaComplexConfig(fit_oiii_wings=False),
        uncertainty_config=neofit.UncertaintyConfig(
            monte_carlo_trials=1, random_seed=4
        ),
    )
    percentiles = result.monte_carlo["percentiles"]
    assert result.monte_carlo["continuum_success_count"] == 1
    assert result.monte_carlo["complex_success_counts"]["hbeta_oiii"] == 1
    assert "MgII_broad_fwhm_kms" in percentiles
    assert "Hb_broad_fwhm_kms" in percentiles
    assert "Ha_broad_fwhm_kms" in percentiles


def test_host_refit_monte_carlo_includes_optional_complexes(monkeypatch):
    spectrum = _global_spectrum(np.linspace(2600.0, 7000.0, 1600))
    spectrum_data = SpectrumData(
        wave_obs=spectrum.wave_obs,
        flux=spectrum.flux,
        error=spectrum.err,
        redshift=0.0,
        object_id="synthetic",
    )

    def fake_host_subtraction(data, **kwargs):
        fit_spectrum = neofit.Spectrum.from_arrays(
            data.wave_obs,
            data.flux,
            err=data.uncertainty(),
            wave_frame="rest",
            survey="desi",
        )
        host = np.zeros_like(data.flux)
        return fit_spectrum, fit_spectrum, None, None, host, data.flux.copy(), []

    monkeypatch.setattr(
        host_workflow, "_host_subtracted_spectrum", fake_host_subtraction
    )
    result = host_workflow._run_host_refit_mc(
        spectrum_data,
        n_trials=1,
        seed=3,
        redshift=0.0,
        template_root="unused",
        template_file="unused",
        host_fit_range=(3600.0, 7000.0),
        host_config=None,
        source="synthetic",
        global_config=_simple_global_config(),
        hbeta_config=neofit.HbetaComplexConfig(fit_oiii_wings=False),
        mgii_config=neofit.MgIIComplexConfig(),
        halpha_config=neofit.HalphaComplexConfig(),
    )
    assert result["continuum_success_count"] == 1
    assert result["complex_success_counts"]["hbeta_oiii"] == 1
    assert "MgII_broad_fwhm_kms" in result["percentiles"]
    assert "Ha_broad_fwhm_kms" in result["percentiles"]


def test_qa_percentiles_and_component_styles():
    values = np.arange(1001, dtype=float)
    lo, hi = _percentile_limits([values], percentiles=(1.0, 99.8), pad=0.0)
    assert lo == pytest.approx(np.percentile(values, 1.0))
    assert hi == pytest.approx(np.percentile(values, 99.8))
    assert _rounded_model_upper_limit(np.array([0.0, 11.0])) == 15.0
    assert _rounded_model_upper_limit(np.array([0.0, 4.0])) == 5.0

    wave = np.linspace(6350.0, 6850.0, 1200)
    continuum = np.ones_like(wave)
    line = _gaussian_area_profile(wave, 50.0, 6564.61, 2200.0)
    spectrum = neofit.Spectrum.from_arrays(
        wave, continuum + line, err=np.full_like(wave, 0.05), wave_frame="rest"
    )
    fit = neofit.fit_halpha_complex(spectrum, _continuum_result(spectrum, continuum))
    kinds = [kind for _, _, _, kind in _line_groups("halpha", fit)]
    assert kinds.count("broad") == 1
    assert set(kinds) <= {"broad", "narrow", "wing"}
    assert _COMBINED_BROAD_STYLE["color"] != _BROAD_COMPONENT_STYLE["color"]
    assert _BROAD_COMPONENT_STYLE["color"] == "#17becf"
    assert _COMBINED_BROAD_STYLE["linestyle"] == "-"
    assert _BROAD_COMPONENT_STYLE["linestyle"] == "-"
    assert _NARROW_STYLE["linestyle"] == "-"
    assert _CONTINUUM_STYLES["balmer_continuum"][0] == "#b8860b"
    assert _CONTINUUM_STYLES["balmer_continuum"][0] != _NARROW_STYLE["color"]
    assert _WING_STYLE["color"] == "#b2182b"
    assert _WING_STYLE["color"] != _CONTINUUM_STYLES["uv_iron"][0]
    assert _CONTINUUM_STYLES["uv_iron"] == _CONTINUUM_STYLES["optical_iron"]
    assert _CONTINUUM_STYLES["power_law"][0] == "#cc79a7"
    assert (
        _CONTINUUM_STYLES["balmer_continuum"]
        == _CONTINUUM_STYLES["balmer_series"]
    )
    assert {
        style[1] for style in _CONTINUUM_STYLES.values()
    } == {"-"}
    label = _flux_density_axis_label(
        "1e-17 erg cm^-2 s^-1 Angstrom^-1"
    )
    assert "f_\\lambda" in label
    assert "\\mathrm{\\AA}" in label


def test_qa_plot_config_and_selection_contract(monkeypatch):
    assert neofit.GlobalQAPlotConfig() == neofit.GlobalQAPlotConfig(
        figure_width=10.5,
        figure_height=6.2,
        max_zoom_panels=4,
        show_smoothed_data=False,
        smoothing_window_pixels=7,
    )
    with pytest.raises(ValueError):
        neofit.GlobalQAPlotConfig(smoothing_window_pixels=4)

    successful = type("SuccessfulFit", (), {"success": True})()
    monkeypatch.setitem(global_io._COMPLEX_WINDOWS, "civ", (1450.0, 1700.0))
    displayed, omitted = _select_zoom_complexes(
        {
            "civ": successful,
            "halpha": successful,
            "mgii": successful,
            "hbeta": successful,
        },
        3,
    )
    assert displayed == ("mgii", "hbeta", "halpha")
    assert omitted == ("civ",)


def test_qa_title_smoothing_and_tick_helpers():
    result = neofit.fit_global_lines(
        _global_spectrum(np.linspace(2600.0, 7000.0, 1800)),
        _simple_global_config(),
        neofit.HbetaComplexConfig(fit_oiii_wings=False),
    )
    assert _qa_overview_title(result) == ""
    result.metadata.update(
        {
            "object_id": "abc",
            "redshift": 0.75,
            "ra": 151.123456,
            "dec": -2.345678,
        }
    )
    assert _qa_overview_title(result) == (
        "DESI TARGETID abc — z=0.7500 — RA=151.12346 — Dec=-2.34568"
    )
    assert _qa_overview_title(
        result,
        neofit.GlobalQAPlotConfig(
            object_name="My Quasar",
            object_label="Source",
            show_coordinates=False,
        ),
    ) == (
        "Source My Quasar — z=0.7500"
    )

    smoothed = _masked_running_median(
        np.array([1.0, 100.0, 3.0, 5.0, 7.0]),
        np.array([True, False, True, True, True]),
        3,
    )
    assert smoothed == pytest.approx([1.0, np.nan, 4.0, 5.0, 6.0], nan_ok=True)

    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator

    figure, axis = plt.subplots()
    _configure_qa_axis(axis)
    assert axis.xaxis.get_tick_params(which="major")["direction"] == "in"
    assert axis.xaxis.get_tick_params(which="major")["top"] is True
    assert axis.yaxis.get_tick_params(which="major")["right"] is True
    assert not isinstance(axis.xaxis.get_minor_locator(), NullLocator)
    assert not isinstance(axis.yaxis.get_minor_locator(), NullLocator)
    axis.set_xlim(4800.0, 5050.0)
    _annotate_emission_lines(
        axis,
        (
            (4862.68, r"H$\beta$"),
            (4960.30, "[O III] 4960"),
            (5008.24, "[O III] 5008"),
        ),
        y_fraction=0.82,
    )
    text_positions = {text.get_text(): text.get_position()[1] for text in axis.texts}
    assert text_positions[r"H$\beta$"] == pytest.approx(0.82)
    assert text_positions["[O III] 4960"] == pytest.approx(0.68)
    assert text_positions["[O III] 5008"] == pytest.approx(0.68)
    assert all(text.get_color() == "#355f8a" for text in axis.texts)
    axis.set_xlim(3650.0, 3800.0)
    _annotate_emission_lines(
        axis,
        ((3728.47, "[O II] 3728"),),
        y_fraction=0.82,
    )
    assert axis.texts[-1].get_position()[1] == pytest.approx(0.68)
    plt.close(figure)


def test_qa_fixed_dimensions_smoothing_and_legends(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    full = neofit.fit_global_lines(
        _global_spectrum(np.linspace(2600.0, 7000.0, 2200)),
        _simple_global_config(),
        neofit.HbetaComplexConfig(fit_oiii_wings=False),
    )
    variants = {
        "one": replace(
            full,
            mgii=None,
            halpha=None,
            line_complexes={"hbeta": full.hbeta},
            metadata={},
        ),
        "two": replace(
            full,
            halpha=None,
            line_complexes={"mgii": full.mgii, "hbeta": full.hbeta},
            metadata={},
        ),
        "three": replace(full, metadata={}),
    }
    pixel_sizes = set()
    for name, result in variants.items():
        path = tmp_path / f"{name}.png"
        _plot_qa(result, path)
        image = plt.imread(path)
        pixel_sizes.add(image.shape[:2])
    assert pixel_sizes == {(992, 1680)}
    assert variants["one"].metadata["qa_overview_annotation"] == {
        "continuum_reduced_chi2": variants["one"].continuum.reduced_chi2,
        "host_state": "not decomposed",
        "host_fractions": "",
    }

    real_close = plt.close
    monkeypatch.setattr(plt, "close", lambda figure: None)
    smoothed_path = tmp_path / "smoothed.png"
    _plot_qa(
        variants["three"],
        smoothed_path,
        neofit.GlobalQAPlotConfig(show_smoothed_data=True),
    )
    figure = plt.gcf()
    overview_axis = figure.axes[0]
    assert figure._supxlabel.get_text() == r"Rest wavelength [$\mathrm{\AA}$]"
    assert "f_\\lambda" in figure._supylabel.get_text()
    assert all(axis.get_xlabel() == "" for axis in figure.axes)
    assert all(axis.get_ylabel() == "" for axis in figure.axes)
    overview_labels = overview_axis.get_legend_handles_labels()[1]
    assert overview_labels.count("smoothed data") == 1
    assert overview_labels.count("iron") <= 1
    assert overview_labels.count("full broad line") == 1
    assert overview_labels.count("narrow line") == 1
    assert (
        figure.axes[1].get_legend_handles_labels()[1].count("broad components")
        == 1
    )
    for axis in figure.axes[2:]:
        assert "broad components" not in axis.get_legend_handles_labels()[1]
    assert variants["three"].metadata["qa_smoothed_data"] is True
    assert variants["three"].metadata["qa_shared_axis_labels"] is True
    real_close(figure)


def test_host_context_companion_plot(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    result = neofit.fit_global_lines(
        _global_spectrum(np.linspace(2600.0, 7000.0, 2200)),
        _simple_global_config(),
        neofit.HbetaComplexConfig(fit_oiii_wings=False),
    )
    host = 0.45 * (result.spectrum.wave_rest / 5100.0) ** -0.4
    result.host_decomp_enabled = True
    result.host_model_on_quasar_grid = host
    result.total_spectrum = neofit.Spectrum.from_arrays(
        result.spectrum.wave_rest,
        result.spectrum.flux + host,
        err=result.spectrum.err,
        wave_frame="rest",
        survey="desi",
    )
    result.metadata.update(
        {
            "object_id": "host-test",
            "redshift": 0.5,
            "host_decomp_enabled": True,
            "continuum_samples": {
                "fracHost_3000": 0.2,
                "fracHost_5100": 0.3,
            },
        }
    )

    assert _has_host_context(result)
    assert "20.0\\%" in _host_fraction_annotation(result)
    assert "30.0\\%" in _host_fraction_annotation(result)
    files = neofit.write_global_line_products(
        result,
        str(tmp_path),
        neofit.GlobalQAPlotConfig(write_other_diagnostics=True),
    )

    assert files["host_context_plot"].endswith(
        "diagnostic_global_host_context.png"
    )
    assert result.metadata["host_context_plot_created"] is True
    assert result.metadata["host_context_figure_size_inches"] == [10.5, 5.2]
    assert result.metadata["host_context_ymin"] == 0.0
    assert result.metadata["host_context_fraction_annotation"]
    image = plt.imread(files["host_context_plot"])
    assert image.shape[:2] == (832, 1680)

    real_close = plt.close
    monkeypatch.setattr(plt, "close", lambda figure: None)
    qa_path = tmp_path / "qa_with_host_overview.png"
    _plot_qa(
        result,
        qa_path,
        neofit.GlobalQAPlotConfig(show_host_context_in_overview=True),
    )
    figure = plt.gcf()
    overview_labels = figure.axes[0].get_legend_handles_labels()[1]
    assert "original spectrum" in overview_labels
    assert "host galaxy" in overview_labels
    assert "host + full model" in overview_labels
    assert "full continuum" in overview_labels
    assert "host + total continuum" not in overview_labels
    for axis in figure.axes[1:]:
        zoom_labels = axis.get_legend_handles_labels()[1]
        assert "host-subtracted data" in zoom_labels
        assert "original spectrum" not in zoom_labels
        assert "host galaxy" not in zoom_labels
    assert result.metadata["qa_host_context_overview_requested"] is True
    assert result.metadata["qa_host_context_overview_used"] is True
    assert "20.0\\%" in result.metadata["qa_overview_annotation"]["host_fractions"]
    assert "30.0\\%" in result.metadata["qa_overview_annotation"]["host_fractions"]
    real_close(figure)


def test_host_context_overview_falls_back_without_host(tmp_path):
    result = neofit.fit_global_lines(
        _global_spectrum(np.linspace(2600.0, 7000.0, 1800)),
        _simple_global_config(),
        neofit.HbetaComplexConfig(fit_oiii_wings=False),
    )
    _plot_qa(
        result,
        tmp_path / "qa_without_host.png",
        neofit.GlobalQAPlotConfig(show_host_context_in_overview=True),
    )
    assert result.metadata["qa_host_context_overview_requested"] is True
    assert result.metadata["qa_host_context_overview_used"] is False
