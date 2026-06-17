"""Iron-template loading and local fitting tests for neofit."""

import numpy as np
import pytest

from qsofitmore import neofit
from qsofitmore.neofit.jacobian import model_jacobian_dense
from qsofitmore.neofit.parameters import pack_line_complex_parameters
from qsofitmore.neofit.residuals import model_vector
from qsofitmore.neofit.templates import IronTemplateError, load_iron_template, prepare_iron_template


def _write_template(path, wave, flux):
    data = np.column_stack([wave, flux])
    np.savetxt(path, data)
    return path


def _iron_template_file(tmp_path):
    wave = np.linspace(4550.0, 5150.0, 301)
    flux = (
        np.exp(-0.5 * ((wave - 4740.0) / 35.0) ** 2)
        + 0.6 * np.exp(-0.5 * ((wave - 5030.0) / 45.0) ** 2)
    )
    return _write_template(tmp_path / "external_iron.txt", wave, flux)


def _hb_config(iron=None):
    return neofit.LineComplexConfig(
        name="Hb_test",
        center=4861.33,
        window=(4700.0, 5100.0),
        components=[
            neofit.GaussianComponent(
                name="Hb_broad",
                center=4860.0,
                amp=4.0,
                sigma=18.0,
                bounds={"amp": (0.0, None), "center": (4820.0, 4900.0), "sigma": (5.0, 80.0)},
            )
        ],
        local_continuum="linear",
        iron=iron,
    )


def test_external_template_is_area_normalized(tmp_path):
    path = _iron_template_file(tmp_path)

    template = load_iron_template("external", template_path=str(path))

    assert template.name == "external"
    assert template.coverage == (4550.0, 5150.0)
    np.testing.assert_allclose(np.trapezoid(template.flux, template.wave_rest), 1.0, rtol=1e-12)


def test_template_parse_errors_have_stable_codes(tmp_path):
    descending = _write_template(tmp_path / "bad_order.txt", [5000.0, 4990.0], [1.0, 1.0])
    nonfinite = _write_template(tmp_path / "nonfinite.txt", [4990.0, 5000.0], [1.0, np.nan])
    zero = _write_template(tmp_path / "zero.txt", [4990.0, 5000.0], [0.0, 0.0])

    with pytest.raises(IronTemplateError) as excinfo:
        load_iron_template("external", template_path=str(descending))
    assert excinfo.value.code == "iron_template_not_monotonic"

    with pytest.raises(IronTemplateError) as excinfo:
        load_iron_template("external", template_path=str(nonfinite))
    assert excinfo.value.code == "iron_template_parse_failed"

    with pytest.raises(IronTemplateError) as excinfo:
        load_iron_template("external", template_path=str(zero))
    assert excinfo.value.code == "iron_template_zero_norm"

    with pytest.raises(IronTemplateError) as excinfo:
        load_iron_template("not_a_template")
    assert excinfo.value.code == "unknown_iron_template"

    with pytest.raises(IronTemplateError) as excinfo:
        load_iron_template("external")
    assert excinfo.value.code == "missing_iron_template_path"


def test_prepare_template_reports_partial_and_no_overlap(tmp_path):
    template = load_iron_template("external", template_path=str(_iron_template_file(tmp_path)))
    wave_fit = np.linspace(4700.0, 5100.0, 120)

    partial = prepare_iron_template(template, wave_fit, (4500.0, 5100.0), fwhm_kms=1200.0)
    assert partial.has_overlap
    assert "iron_template_partial_coverage" in [warning.code for warning in partial.warnings]

    no_overlap = prepare_iron_template(template, wave_fit, (6000.0, 6200.0), fwhm_kms=1200.0)
    assert not no_overlap.has_overlap
    assert np.all(no_overlap.basis == 0.0)
    assert "iron_template_no_overlap" in [warning.code for warning in no_overlap.warnings]


def test_iron_amplitude_and_fwhm_jacobian_match_finite_difference(tmp_path):
    wave = np.linspace(4700.0, 5100.0, 80)
    template = load_iron_template("external", template_path=str(_iron_template_file(tmp_path)))
    basis = prepare_iron_template(template, wave, (4700.0, 5100.0), fwhm_kms=1400.0).basis
    config = _hb_config(
        iron=neofit.IronTemplateConfig(
            template="external",
            template_path="unused.txt",
            fwhm_kms=1400.0,
            fwhm_bounds=(800.0, 2600.0),
        )
    )
    packed = pack_line_complex_parameters(
        config,
        wave,
        flux_fit=np.ones_like(wave),
        iron_basis=basis,
        iron_template=template,
    )
    theta = packed.initial.copy()
    assert packed.iron_index is not None
    assert packed.iron_fwhm_index is not None

    jac = model_jacobian_dense(theta, packed, wave)
    for index, step in [(packed.iron_index, 1.0e-6), (packed.iron_fwhm_index, 1.0)]:
        hi = theta.copy()
        lo = theta.copy()
        hi[index] += step
        lo[index] -= step
        finite_difference = (model_vector(hi, packed, wave) - model_vector(lo, packed, wave)) / (2.0 * step)

        np.testing.assert_allclose(jac[:, index], finite_difference, rtol=2e-3, atol=2e-6)


def test_fit_line_complex_recovers_external_iron_amplitude(tmp_path):
    path = _iron_template_file(tmp_path)
    wave = np.linspace(4700.0, 5100.0, 260)
    template = load_iron_template("external", template_path=str(path))
    true_fwhm = 1800.0
    prepared = prepare_iron_template(template, wave, (4700.0, 5100.0), fwhm_kms=true_fwhm)
    true_iron_amp = 1800.0
    line = 5.0 * np.exp(-0.5 * ((wave - 4862.0) / 20.0) ** 2)
    continuum = 1.2 + 0.0004 * (wave - 4900.0)
    flux = continuum + line + true_iron_amp * prepared.basis
    err = np.full_like(wave, 0.03)
    spectrum = neofit.Spectrum.from_arrays(wave, flux, err=err, z=0.0, wave_frame="rest", survey="desi")
    config = _hb_config(
        iron=neofit.IronTemplateConfig(
            template="external",
            template_path=str(path),
            amp=1000.0,
            amp_bounds=(0.0, 5000.0),
            fwhm_kms=1200.0,
            fwhm_bounds=(800.0, 3500.0),
        )
    )

    result = neofit.fit_line_complex(spectrum, config)
    table = result.to_table()
    iron_row = table[table["component_type"] == "iron"].iloc[0]

    assert result.success
    assert "iron" in result.component_models
    assert abs(result.param_values["iron.amp"] - true_iron_amp) < 80.0
    assert abs(result.param_values["iron.fwhm_kms"] - true_fwhm) < 300.0
    assert iron_row["iron_template"] == "external"
    assert abs(iron_row["iron_fwhm_kms"] - true_fwhm) < 300.0
    assert np.isfinite(iron_row["iron_flux_input"])
    assert np.isfinite(iron_row["iron_flux_cgs"])


def test_bundled_template_aliases_work_in_recipes():
    for template_name in ["bg92", "park22", "vc04"]:
        wave = np.linspace(4700.0, 5100.0, 220)
        flux = 1.0 + 5.0 * np.exp(-0.5 * ((wave - 4861.33) / 22.0) ** 2)
        err = np.full_like(wave, 0.05)
        spectrum = neofit.Spectrum.from_arrays(wave, flux, err=err, z=0.0, wave_frame="rest")
        result = neofit.fit_line_complex(spectrum, neofit.recipes.local_hbeta(iron_template=template_name))
        assert result.success
        assert "iron.amp" in result.param_values
        assert result.metadata["iron"]["template"] in {"bg92_optical", "park22_optical", "veron04_optical"}

    wave = np.linspace(2700.0, 2900.0, 180)
    flux = 1.0 + 4.0 * np.exp(-0.5 * ((wave - 2798.75) / 18.0) ** 2)
    err = np.full_like(wave, 0.05)
    spectrum = neofit.Spectrum.from_arrays(wave, flux, err=err, z=0.0, wave_frame="rest")
    result = neofit.fit_line_complex(spectrum, neofit.recipes.local_mgii(iron_template="vw01"))

    assert result.success
    assert result.metadata["iron"]["template"] == "vw01_uv"


def test_no_overlap_warns_and_drops_iron_parameter():
    wave = np.linspace(4700.0, 5100.0, 160)
    flux = 1.0 + 4.0 * np.exp(-0.5 * ((wave - 4861.33) / 22.0) ** 2)
    err = np.full_like(wave, 0.05)
    spectrum = neofit.Spectrum.from_arrays(wave, flux, err=err, z=0.0, wave_frame="rest")

    result = neofit.fit_line_complex(spectrum, neofit.recipes.local_hbeta(iron_template="vw01"))

    assert result.success
    assert "iron_template_no_overlap" in result.warning_codes()
    assert "iron.amp" not in result.param_values
    assert result.metadata["iron"]["has_overlap"] is False


def test_fit_local_no_overlap_warning_does_not_crash_other_windows():
    wave = np.linspace(2700.0, 5100.0, 600)
    mgii = 3.0 * np.exp(-0.5 * ((wave - 2798.75) / 18.0) ** 2)
    hb = 4.0 * np.exp(-0.5 * ((wave - 4861.33) / 22.0) ** 2)
    flux = 1.0 + mgii + hb
    err = np.full_like(wave, 0.05)
    spectrum = neofit.Spectrum.from_arrays(wave, flux, err=err, z=0.0, wave_frame="rest")
    config = neofit.LocalFitConfig(
        windows=[
            neofit.recipes.local_hbeta(iron_template="vw01"),
            neofit.recipes.local_mgii(iron_template="vw01"),
        ]
    )

    result = neofit.fit_local(spectrum, config)

    assert result.success
    assert result.window_results["Hb_OIII"].success
    assert result.window_results["MgII"].success
    assert "iron_template_no_overlap" in result.warning_codes()
    assert "iron.amp" not in result.window_results["Hb_OIII"].param_values
    assert "iron.amp" in result.window_results["MgII"].param_values
