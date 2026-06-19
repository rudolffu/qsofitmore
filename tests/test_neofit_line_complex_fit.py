"""Synthetic line-complex fitting tests for neofit."""

import numpy as np

from qsofitmore import neofit


def _synthetic_spectrum(local_continuum="linear", z=0.12):
    rng = np.random.default_rng(123)
    wave_rest = np.linspace(4700.0, 5100.0, 300)
    x = (wave_rest - 4861.33) / 200.0
    continuum = 1.5 + 0.25 * x if local_continuum == "linear" else np.full_like(wave_rest, 1.5)
    line = 8.0 * np.exp(-0.5 * ((wave_rest - 4862.2) / 24.0) ** 2)
    err = np.full_like(wave_rest, 0.08)
    flux = continuum + line + rng.normal(0.0, err)
    return neofit.Spectrum.from_arrays(wave_rest * (1.0 + z), flux, err=err, z=z), wave_rest


def _config(local_continuum="linear", jacobian="analytic_dense"):
    return neofit.LineComplexConfig(
        center=4861.33,
        window=(4700.0, 5100.0),
        components=[
            neofit.GaussianComponent(
                name="Hb_broad",
                center=4858.0,
                amp=6.0,
                sigma=30.0,
                bounds={
                    "amp": (0.0, None),
                    "center": (4820.0, 4900.0),
                    "sigma": (5.0, 120.0),
                },
            )
        ],
        local_continuum=local_continuum,
        jacobian=jacobian,
    )


def test_synthetic_gaussian_recovery_with_linear_continuum():
    spec, _ = _synthetic_spectrum(local_continuum="linear")
    result = neofit.fit_line_complex(spec, _config(local_continuum="linear"))

    assert result.success
    assert result.dof > 0
    assert abs(result.param_values["Hb_broad.amp"] - 8.0) < 0.3
    assert abs(result.param_values["Hb_broad.center"] - 4862.2) < 0.5
    assert abs(result.param_values["Hb_broad.sigma"] - 24.0) < 1.0
    assert "continuum" in result.component_models
    assert "Hb_broad" in result.component_models
    assert "Hb_broad" in result.to_table()["name"].tolist()


def test_constant_and_no_continuum_modes_run():
    spec, _ = _synthetic_spectrum(local_continuum="constant")
    constant = neofit.fit_line_complex(spec, _config(local_continuum="constant"))
    no_cont = neofit.fit_line_complex(spec, _config(local_continuum=None))

    assert constant.success
    assert no_cont.success
    assert "continuum.c0" in constant.param_values
    assert "continuum.c0" not in no_cont.param_values


def test_sparse_jacobian_mode_runs():
    spec, _ = _synthetic_spectrum(local_continuum="linear")
    result = neofit.fit_line_complex(spec, _config(local_continuum="linear", jacobian="analytic_sparse"))

    assert result.success
    assert abs(result.param_values["Hb_broad.center"] - 4862.2) < 0.5


def test_invalid_pixels_are_ignored():
    spec, wave_rest = _synthetic_spectrum(local_continuum="linear")
    flux = spec.flux.copy()
    err = spec.err.copy()
    flux[10] = np.nan
    err[20] = -1.0
    mask = np.ones_like(flux, dtype=bool)
    mask[30] = False
    spec = neofit.Spectrum.from_arrays(wave_rest * (1.0 + spec.z), flux, err=err, z=spec.z, mask=mask)

    result = neofit.fit_line_complex(spec, _config(local_continuum="linear"))

    assert result.success
    assert result.wave_rest_fit.size == wave_rest.size - 3


def test_fit_windows_and_mask_windows_select_fitted_pixels():
    wave = np.linspace(4800.0, 4920.0, 121)
    line = 5.0 * np.exp(-0.5 * ((wave - 4861.33) / 18.0) ** 2)
    flux = 1.0 + line
    err = np.full_like(wave, 0.05)
    spec = neofit.Spectrum.from_arrays(wave, flux, err=err, z=0.0, wave_frame="rest")
    config = neofit.LineComplexConfig(
        center=4861.33,
        window=(4800.0, 4920.0),
        fit_windows=[(4800.0, 4850.0), (4860.0, 4920.0)],
        mask_windows=[(4880.0, 4890.0)],
        components=[
            neofit.GaussianComponent(
                name="Hb_broad",
                center=4861.33,
                amp=4.0,
                sigma=20.0,
                bounds={"amp": (0.0, None), "sigma": (5.0, 80.0)},
            )
        ],
        local_continuum="constant",
    )

    result = neofit.fit_line_complex(spec, config)

    assert result.success
    assert np.all((result.wave_rest_fit <= 4850.0) | (result.wave_rest_fit >= 4860.0))
    assert not np.any((result.wave_rest_fit >= 4880.0) & (result.wave_rest_fit <= 4890.0))
    assert np.any((result.wave_rest_window >= 4880.0) & (result.wave_rest_window <= 4890.0))
    assert result.model_window.shape == result.wave_rest_window.shape
    assert result.fit_used_window.shape == result.wave_rest_window.shape
    assert not np.any(
        result.fit_used_window[(result.wave_rest_window >= 4880.0) & (result.wave_rest_window <= 4890.0)]
    )
    assert result.metadata["fit_windows"] == [(4800.0, 4850.0), (4860.0, 4920.0)]
    assert result.metadata["mask_windows"] == [(4880.0, 4890.0)]
