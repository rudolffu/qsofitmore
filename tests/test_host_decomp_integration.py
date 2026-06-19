#!/usr/bin/env python
"""Optional local pPXF/template integration checks."""

from pathlib import Path
import importlib.util

import numpy as np
import pytest

from qsofitmore.host_decomp.io import SpectrumData
from qsofitmore.host_decomp.ppxf_host import prepare_desi_for_host_decomp, predict_host_sed, run_ppxf_host_fit
from qsofitmore.host_decomp.templates import load_ppxf_npz_templates

def test_local_ppxf_template_fit_smoke():
    if importlib.util.find_spec("ppxf") is None:
        pytest.skip("pPXF is not installed")
    template_path = Path.home() / "tools/ppxf_data/spectra_emiles_9.0.npz"
    if not template_path.exists():
        pytest.skip("local pPXF E-MILES template file not found")

    wave = np.linspace(3600.0, 7000.0, 240)
    flux = 2.0 * (wave / 5100.0) ** -1.0 + 0.15 * np.sin(wave / 500.0)
    ivar = np.full_like(wave, 100.0)
    spec = SpectrumData(wave_obs=wave, flux=flux, ivar=ivar, redshift=0.0, object_id="synthetic")
    templates = load_ppxf_npz_templates(write_report=False)
    prep = prepare_desi_for_host_decomp(spec, fit_range=(3700.0, 6800.0))
    fit = run_ppxf_host_fit(prep, templates, quiet=True)
    sed = predict_host_sed(fit)

    assert fit.status == "success"
    assert fit.host_model.shape == prep.wave_rest.shape
    assert sed.flags["template_covers_1um"]
