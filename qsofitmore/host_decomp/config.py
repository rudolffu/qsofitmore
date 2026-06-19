"""Configuration defaults for optional DESI pPXF host decomposition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple


DEFAULT_LINE_CENTERS = {
    "MgII": 2798.0,
    "NeV3426": 3426.0,
    "OII3727": 3727.0,
    "Hdelta": 4102.0,
    "Hgamma": 4341.0,
    "Hbeta": 4861.0,
    "OIII4959": 4959.0,
    "OIII5007": 5007.0,
    "HeI5876": 5876.0,
    "OI6300": 6300.0,
    "Halpha": 6563.0,
    "NII6548": 6548.0,
    "NII6583": 6583.0,
    "SII6716": 6716.0,
    "SII6731": 6731.0,
}


DEFAULT_LINE_MASK_WIDTHS = {
    "default": 800.0,
    "MgII": 1800.0,
    "Hdelta": 1400.0,
    "Hgamma": 1400.0,
    "Hbeta": 1800.0,
    "Halpha": 2200.0,
}


DEFAULT_BROAD_LINE_MASK_WIDTHS = {
    "default": 1200.0,
    "MgII": 3500.0,
    "Hdelta": 2600.0,
    "Hgamma": 2600.0,
    "Hbeta": 3500.0,
    "Halpha": 4500.0,
}


@dataclass
class HostDecompConfig:
    """Runtime config for the optional pPXF host-decomposition workflow."""

    template_root: str = "~/tools/ppxf_data"
    template_file: str = "spectra_emiles_9.0.npz"
    template_family: str = "emiles"
    fit_range: Tuple[float, float] = (3600.0, 7000.0)
    line_mask_widths: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_LINE_MASK_WIDTHS))
    broad_line_mask_widths: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_BROAD_LINE_MASK_WIDTHS))
    polynomial_degree: int = 4
    multiplicative_polynomial_degree: int = 0
    use_regularization: bool = False
    agn_powerlaw_slopes: Sequence[float] = (-2.0, -1.5, -1.0, -0.5, 0.0)
    run_qsofitmore: bool = False
    n_iterations: int = 1
    euclid_scaling_mode: str = "free_scale"
    continuum_windows: List[Tuple[float, float]] = field(
        default_factory=lambda: [(4100.0, 4300.0), (5200.0, 5600.0), (6000.0, 6200.0)]
    )
    output_dir: str = "outputs/ppxf_qsofitmore"


def default_config() -> HostDecompConfig:
    """Return a fresh default host-decomposition config."""

    return HostDecompConfig()
