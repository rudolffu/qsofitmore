"""Tests for optional host subtraction orchestration before neofit."""

import numpy as np
import pandas as pd
import pytest

from qsofitmore import neofit


def _local_config():
    return neofit.LocalFitConfig(windows=[neofit.recipes.local_hbeta()])


def test_fit_with_optional_host_decomp_without_ppxf_path(tmp_path):
    pytest.importorskip("pyarrow")
    wave = np.linspace(6200.0, 7600.0, 300)
    rest = wave / 1.45
    flux = 1.0 + 4.0 * np.exp(-0.5 * ((rest - 4861.33) / 22.0) ** 2)
    ivar = np.ones_like(flux) * 100.0
    path = tmp_path / "spectra.parquet"
    pd.DataFrame(
        {
            "targetid": ["obj"],
            "redshift": [0.45],
            "wavelength": [wave],
            "flux": [flux],
            "ivar": [ivar],
            "mask": [np.zeros_like(flux, dtype=int)],
        }
    ).to_parquet(path)

    result = neofit.fit_with_optional_host_decomp(
        str(path),
        _local_config(),
        row_index=0,
        run_host_decomp=False,
    )

    assert result.local_result.success
    assert not result.host_decomp_enabled
    assert result.host_model_on_quasar_grid is None
    assert result.fit_spectrum.flux.shape == wave.shape


def test_global_fit_kind_runs_without_host(tmp_path):
    pytest.importorskip("pyarrow")
    wave = np.linspace(3600.0, 7600.0, 1200)
    rest = wave / 1.2
    continuum = 2.0 * (rest / 3000.0) ** -1.2
    sigma = (2200.0 / 299792.458) * 4862.68 / 2.354820045
    line = 80.0 * np.exp(-0.5 * ((rest - 4862.68) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
    path = tmp_path / "global.parquet"
    pd.DataFrame(
        {
            "targetid": ["obj"],
            "redshift": [0.2],
            "wavelength": [wave],
            "flux": [continuum + line],
            "ivar": [np.full_like(wave, 400.0)],
            "mask": [np.zeros_like(wave, dtype=int)],
        }
    ).to_parquet(path)

    result = neofit.fit_with_optional_host_decomp(
        str(path),
        fit_kind="global",
        row_index=0,
        global_config=neofit.GlobalContinuumConfig(
            uv_iron=None,
            optical_iron=None,
            balmer_continuum=neofit.BalmerContinuumConfig(enabled=False),
            balmer_series=neofit.BalmerSeriesConfig(enabled=False),
        ),
        hbeta_config=neofit.HbetaComplexConfig(fit_oiii_wings=False),
    )

    assert result.success
    assert result.metadata["fit_kind"] == "global"
    assert result.metadata["targetid"] == "obj"
    assert result.metadata["ra"] is None
    assert result.metadata["dec"] is None
