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


def test_global_fit_kind_is_reserved(tmp_path):
    with pytest.raises(NotImplementedError, match="global neofit fitting is pending"):
        neofit.fit_with_optional_host_decomp(
            str(tmp_path / "missing.parquet"),
            _local_config(),
            fit_kind="global",
        )
