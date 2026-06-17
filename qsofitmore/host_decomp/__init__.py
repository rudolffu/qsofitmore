"""Optional pPXF + qsofitmore host-decomposition helpers.

This subpackage intentionally keeps pPXF imports lazy. Importing
``qsofitmore`` or ``qsofitmore.host_decomp`` does not require pPXF or any
stellar-template data to be installed.
"""

from .config import HostDecompConfig, default_config
from .euclid import EuclidHostPrediction, predict_host_for_euclid_spectrum
from .io import SpectrumData, read_sparcli_spectrum
from .ppxf_host import (
    HostSED,
    PPXFHostFitResult,
    PreprocessedSpectrum,
    prepare_desi_for_host_decomp,
    predict_host_sed,
    run_ppxf_host_fit,
    run_ppxf_qsofitmore_decomposition,
)
from .templates import PPXFTemplateLibrary, load_ppxf_npz_templates

__all__ = [
    "HostDecompConfig",
    "EuclidHostPrediction",
    "HostSED",
    "PPXFHostFitResult",
    "PPXFTemplateLibrary",
    "PreprocessedSpectrum",
    "SpectrumData",
    "default_config",
    "load_ppxf_npz_templates",
    "prepare_desi_for_host_decomp",
    "predict_host_for_euclid_spectrum",
    "predict_host_sed",
    "read_sparcli_spectrum",
    "run_ppxf_host_fit",
    "run_ppxf_qsofitmore_decomposition",
]
