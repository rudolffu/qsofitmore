"""Bundled high-order Balmer-series templates."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..warnings import NeoFitWarning

C_KMS = 299792.458
FWHM_TO_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))
_PROVENANCE_ALIASES = {
    "sh95": "sh95",
    "pure": "sh95",
    "k13": "sh95_k13full_ext",
    "k13full": "sh95_k13full_ext",
    "sh95_k13full_ext": "sh95_k13full_ext",
    "asymptotic": "sh95_asymptotic_ext",
    "sh95_asymptotic_ext": "sh95_asymptotic_ext",
}


class BalmerTemplateError(ValueError):
    """Balmer-template error with a stable code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class BalmerSeriesTemplate:
    """One H-beta-relative Balmer line-list template."""

    name: str
    n_upper: np.ndarray
    wavelength_vacuum: np.ndarray
    rel_flux_hbeta: np.ndarray
    te_k: float
    log10_ne: float
    n_min: int
    n_max: int
    provenance: str
    source_path: str
    row_sources: Tuple[str, ...] = ()
    warnings: List[NeoFitWarning] = field(default_factory=list)


def _data_dir() -> Path:
    return Path(__file__).resolve().parent / "data" / "balmer"


def _canonical_provenance(provenance: str) -> str:
    key = str(provenance).strip().lower().replace("-", "_")
    if "energy" in key:
        raise BalmerTemplateError(
            "unsupported_balmer_template",
            "The energy-only high-n Balmer extension is diagnostic-only and is not bundled.",
        )
    if key not in _PROVENANCE_ALIASES:
        raise BalmerTemplateError("unknown_balmer_template", f"Unknown Balmer provenance: {provenance!r}")
    return _PROVENANCE_ALIASES[key]


def _filename(log10_ne: int, n_min: int, provenance: str) -> str:
    canonical = _canonical_provenance(provenance)
    n_max = 50 if canonical == "sh95" else 400
    return (
        f"balmer_caseB_T15000_logNe{int(log10_ne):02d}_"
        f"n{int(n_min):03d}_n{n_max:03d}_{canonical}.csv"
    )


def list_balmer_templates() -> Dict[str, str]:
    """Return bundled template names and their production/systematics status."""

    out = {}
    for logne in (9, 10):
        for n_min in (6, 7):
            for provenance in ("sh95", "sh95_k13full_ext", "sh95_asymptotic_ext"):
                name = Path(_filename(logne, n_min, provenance)).stem
                out[name] = "production" if provenance == "sh95" else "systematics"
    return out


def load_balmer_template(
    *,
    log10_ne: int = 9,
    n_min: int = 6,
    provenance: str = "sh95",
) -> BalmerSeriesTemplate:
    """Load one bundled Case-B, 15000 K Balmer line list."""

    if int(log10_ne) not in (9, 10):
        raise BalmerTemplateError("unknown_balmer_template", "log10_ne must be 9 or 10.")
    if int(n_min) not in (6, 7):
        raise BalmerTemplateError("unknown_balmer_template", "n_min must be 6 or 7.")
    canonical = _canonical_provenance(provenance)
    path = _data_dir() / _filename(int(log10_ne), int(n_min), canonical)
    if not path.exists():
        raise BalmerTemplateError("missing_balmer_template", f"Bundled Balmer template is missing: {path}")

    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    n_upper = np.array([int(row["n_upper"]) for row in rows], dtype=int)
    wavelength = np.array([float(row["lambda_vacuum_angstrom"]) for row in rows], dtype=float)
    flux = np.array([float(row["rel_flux_hbeta"]) for row in rows], dtype=float)
    warnings: List[NeoFitWarning] = []
    if canonical != "sh95":
        warnings.append(
            NeoFitWarning(
                code="balmer_high_n_extension_model_dependent",
                message="The n=51-400 Balmer extension is model-dependent and intended for systematics fits.",
                context={"provenance": canonical},
            )
        )
    return BalmerSeriesTemplate(
        name=path.stem,
        n_upper=n_upper,
        wavelength_vacuum=wavelength,
        rel_flux_hbeta=flux,
        te_k=float(rows[0]["te_k"]),
        log10_ne=float(rows[0]["log10_ne_cm3"]),
        n_min=int(n_upper.min()),
        n_max=int(n_upper.max()),
        provenance=canonical,
        source_path=str(path),
        row_sources=tuple(row["source"] for row in rows),
        warnings=warnings,
    )


def evaluate_balmer_series(
    template: BalmerSeriesTemplate,
    wave_rest: np.ndarray,
    fwhm_kms: float,
) -> np.ndarray:
    """Evaluate an H-beta-relative, area-normalized broadened line series."""

    wave_rest = np.asarray(wave_rest, dtype=float)
    if not np.isfinite(fwhm_kms) or fwhm_kms <= 0:
        raise ValueError("Balmer-series FWHM must be positive and finite.")
    basis = np.zeros_like(wave_rest)
    sigma = (float(fwhm_kms) / C_KMS) * template.wavelength_vacuum / FWHM_TO_SIGMA
    for center, area, width in zip(template.wavelength_vacuum, template.rel_flux_hbeta, sigma):
        if width <= 0:
            continue
        u = (wave_rest - center) / width
        basis += area * np.exp(-0.5 * u * u) / (np.sqrt(2.0 * np.pi) * width)
    return basis


def evaluate_balmer_series_with_derivative(
    template: BalmerSeriesTemplate,
    wave_rest: np.ndarray,
    fwhm_kms: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the broadened Balmer series and its FWHM derivative."""

    wave_rest = np.asarray(wave_rest, dtype=float)
    if not np.isfinite(fwhm_kms) or fwhm_kms <= 0:
        raise ValueError("Balmer-series FWHM must be positive and finite.")
    basis = np.zeros_like(wave_rest)
    derivative = np.zeros_like(wave_rest)
    sigma = (float(fwhm_kms) / C_KMS) * template.wavelength_vacuum / FWHM_TO_SIGMA
    for center, area, width in zip(template.wavelength_vacuum, template.rel_flux_hbeta, sigma):
        if width <= 0:
            continue
        u = (wave_rest - center) / width
        profile = area * np.exp(-0.5 * u * u) / (np.sqrt(2.0 * np.pi) * width)
        basis += profile
        derivative += profile * (u * u - 1.0) / float(fwhm_kms)
    return basis, derivative
