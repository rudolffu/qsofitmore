"""Unit and survey metadata tests for neofit."""

import numpy as np
import pytest

from qsofitmore import neofit


def _arrays():
    wave = np.linspace(4800.0, 4920.0, 80)
    line = 5.0 * np.exp(-0.5 * ((wave - 4861.33) / 20.0) ** 2)
    flux = 1.0 + line
    err = np.full_like(wave, 0.05)
    return wave, flux, err


def _config():
    return neofit.LineComplexConfig(
        center=4861.33,
        window=(4800.0, 4920.0),
        components=[
            neofit.GaussianComponent(
                name="Hb_broad",
                center=4861.33,
                amp=4.0,
                sigma=22.0,
                bounds={"amp": (0.0, None), "sigma": (5.0, 80.0)},
            )
        ],
        local_continuum="constant",
    )


def test_survey_presets_set_cgs_scale_and_normalize_aliases():
    wave, flux, err = _arrays()
    for survey in ["desi", "DESI", "desi-dr1", "desi_edr", "sdss", "SDSS"]:
        spec = neofit.Spectrum.from_arrays(wave, flux, err=err, z=0.0, survey=survey)
        expected = "sdss" if survey.lower() == "sdss" else "desi"
        assert spec.metadata.survey == expected
        assert spec.wave_unit == "Angstrom"
        assert spec.flux_density_unit == "1e-17 erg s^-1 cm^-2 Angstrom^-1"
        assert spec.flux_density_scale_to_cgs == 1e-17


def test_unit_preset_and_input_units():
    wave, flux, err = _arrays()
    scaled = neofit.Spectrum.from_arrays(wave, flux, err=err, z=0.0, unit_preset="1e-17cgs")
    unknown = neofit.Spectrum.from_arrays(wave, flux, err=err, z=0.0, unit_preset="input")

    assert scaled.flux_density_scale_to_cgs == 1e-17
    assert unknown.flux_density_unit == "input"
    assert unknown.flux_density_scale_to_cgs is None


def test_explicit_metadata_overrides_presets():
    wave, flux, err = _arrays()
    spec = neofit.Spectrum.from_arrays(
        wave,
        flux,
        err=err,
        z=0.0,
        survey="desi",
        flux_density_unit="1e-16 erg s^-1 cm^-2 Angstrom^-1",
        flux_density_scale_to_cgs=1e-16,
        source="manual",
    )

    assert spec.metadata.survey == "desi"
    assert spec.flux_density_unit == "1e-16 erg s^-1 cm^-2 Angstrom^-1"
    assert spec.flux_density_scale_to_cgs == 1e-16
    assert spec.metadata.source == "manual"


def test_unknown_units_fit_but_do_not_report_cgs_flux():
    wave, flux, err = _arrays()
    spec = neofit.Spectrum.from_arrays(wave, flux, err=err, z=0.0)
    result = neofit.fit_line_complex(spec, _config())
    row = result.to_table().iloc[0]

    assert result.success
    assert "flux_scale_unknown_cgs_not_reported" in result.warning_codes()
    assert np.isfinite(row["line_flux_input"])
    assert np.isnan(row["line_flux_cgs"])


def test_cgs_line_flux_is_scaled_when_known():
    wave, flux, err = _arrays()
    spec = neofit.Spectrum.from_arrays(wave, flux, err=err, z=0.0, survey="sdss")
    result = neofit.fit_line_complex(spec, _config())
    row = result.to_table().iloc[0]
    expected_input = row["amp"] * row["sigma"] * np.sqrt(2.0 * np.pi)

    assert np.isclose(row["line_flux_input"], expected_input)
    assert np.isclose(row["line_flux_cgs"], expected_input * 1e-17)


def test_unknown_presets_raise_clear_errors():
    wave, flux, err = _arrays()
    with pytest.raises(ValueError, match="Unknown neofit survey preset"):
        neofit.Spectrum.from_arrays(wave, flux, err=err, z=0.0, survey="mystery")
    with pytest.raises(ValueError, match="Unknown neofit unit preset"):
        neofit.Spectrum.from_arrays(wave, flux, err=err, z=0.0, unit_preset="mystery")
