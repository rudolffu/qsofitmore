"""Iron-template objects and grid preparation for neofit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..warnings import NeoFitWarning


C_KMS = 299792.458
FWHM_TO_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


class IronTemplateError(ValueError):
    """Exception carrying a stable warning/error code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class IronTemplate:
    """Normalized iron-template spectrum."""

    name: str
    wave_rest: np.ndarray
    flux: np.ndarray
    wave_unit: str = "Angstrom"
    flux_unit: str = "arbitrary area-normalized"
    reference: Optional[str] = None
    source_path: Optional[str] = None
    coverage: Optional[Tuple[float, float]] = None
    notes: List[str] = field(default_factory=list)
    normalization: str = "area"

    def __post_init__(self) -> None:
        self.wave_rest = np.asarray(self.wave_rest, dtype=float)
        self.flux = np.asarray(self.flux, dtype=float)
        if self.coverage is None and self.wave_rest.size:
            self.coverage = (float(self.wave_rest.min()), float(self.wave_rest.max()))


@dataclass
class PreparedIronTemplate:
    """Iron template evaluated on one local fitting grid."""

    template: IronTemplate
    basis: np.ndarray
    fwhm_kms: float
    warnings: List[NeoFitWarning] = field(default_factory=list)

    @property
    def has_overlap(self) -> bool:
        return bool(np.any(np.isfinite(self.basis) & (self.basis != 0)))

    @property
    def flux_integral_unit_amp(self) -> float:
        return float("nan")


def _log_grid(wave_min: float, wave_max: float, velocity_step_kms: float) -> np.ndarray:
    dlog = velocity_step_kms / C_KMS
    return np.exp(np.arange(np.log(wave_min), np.log(wave_max) + 0.5 * dlog, dlog))


def _broaden_template_with_derivative(
    template: IronTemplate,
    fwhm_kms: float,
    velocity_step_kms: float = 25.0,
):
    if fwhm_kms <= 0 or not np.isfinite(fwhm_kms):
        raise IronTemplateError("iron_template_parse_failed", "Iron template FWHM must be positive and finite.")
    wave = template.wave_rest
    flux = template.flux
    grid = _log_grid(float(wave.min()), float(wave.max()), velocity_step_kms)
    sampled = np.interp(grid, wave, flux, left=0.0, right=0.0)
    sigma_pix = (float(fwhm_kms) / FWHM_TO_SIGMA) / float(velocity_step_kms)
    half = max(1, int(np.ceil(4.0 * sigma_pix)))
    x = np.arange(-half, half + 1, dtype=float)
    raw_kernel = np.exp(-0.5 * (x / sigma_pix) ** 2)
    raw_derivative = raw_kernel * x**2 / sigma_pix**3
    kernel_sum = raw_kernel.sum()
    kernel = raw_kernel / kernel_sum
    kernel_derivative_sigma = (
        raw_derivative * kernel_sum - raw_kernel * raw_derivative.sum()
    ) / kernel_sum**2
    sigma_derivative_fwhm = 1.0 / (FWHM_TO_SIGMA * float(velocity_step_kms))
    kernel_derivative_fwhm = kernel_derivative_sigma * sigma_derivative_fwhm
    return (
        grid,
        np.convolve(sampled, kernel, mode="same"),
        np.convolve(sampled, kernel_derivative_fwhm, mode="same"),
    )


def _broaden_template(template: IronTemplate, fwhm_kms: float, velocity_step_kms: float = 25.0):
    grid, broadened, _ = _broaden_template_with_derivative(
        template, fwhm_kms, velocity_step_kms
    )
    return grid, broadened


def _apply_coverage_taper(
    template: IronTemplate,
    wave_rest_fit: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=float).copy()
    coverage = template.coverage or (float(template.wave_rest.min()), float(template.wave_rest.max()))
    span = float(coverage[1] - coverage[0])
    taper_width = min(100.0, max(20.0, 0.05 * span))
    left = (wave_rest_fit >= coverage[0]) & (wave_rest_fit < coverage[0] + taper_width)
    right = (wave_rest_fit > coverage[1] - taper_width) & (wave_rest_fit <= coverage[1])
    if np.any(left):
        phase = (wave_rest_fit[left] - coverage[0]) / taper_width
        values[left] *= 0.5 - 0.5 * np.cos(np.pi * phase)
    if np.any(right):
        phase = (coverage[1] - wave_rest_fit[right]) / taper_width
        values[right] *= 0.5 - 0.5 * np.cos(np.pi * phase)
    return values


def evaluate_iron_basis(
    template: IronTemplate,
    wave_rest_fit: np.ndarray,
    fwhm_kms: float,
    velocity_step_kms: float = 25.0,
) -> np.ndarray:
    """Return a broadened iron-template basis on a rest-frame fitting grid."""

    wave_rest_fit = np.asarray(wave_rest_fit, dtype=float)
    broadened_wave, broadened_flux = _broaden_template(template, fwhm_kms, velocity_step_kms=velocity_step_kms)
    basis = np.interp(wave_rest_fit, broadened_wave, broadened_flux, left=0.0, right=0.0)
    return _apply_coverage_taper(template, wave_rest_fit, basis)


def evaluate_iron_basis_with_derivative(
    template: IronTemplate,
    wave_rest_fit: np.ndarray,
    fwhm_kms: float,
    velocity_step_kms: float = 25.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the broadened iron basis and its FWHM derivative."""

    wave_rest_fit = np.asarray(wave_rest_fit, dtype=float)
    broadened_wave, broadened_flux, derivative_flux = _broaden_template_with_derivative(
        template,
        fwhm_kms,
        velocity_step_kms=velocity_step_kms,
    )
    basis = np.interp(wave_rest_fit, broadened_wave, broadened_flux, left=0.0, right=0.0)
    derivative = np.interp(
        wave_rest_fit, broadened_wave, derivative_flux, left=0.0, right=0.0
    )
    return (
        _apply_coverage_taper(template, wave_rest_fit, basis),
        _apply_coverage_taper(template, wave_rest_fit, derivative),
    )


def prepare_iron_template(
    template: IronTemplate,
    wave_rest_fit: np.ndarray,
    window: Tuple[float, float],
    fwhm_kms: float,
    velocity_step_kms: float = 25.0,
) -> PreparedIronTemplate:
    """Broaden and interpolate an iron template onto a fit grid."""

    wave_rest_fit = np.asarray(wave_rest_fit, dtype=float)
    warnings: List[NeoFitWarning] = []
    coverage = template.coverage or (float(template.wave_rest.min()), float(template.wave_rest.max()))
    lo, hi = map(float, window)
    context = {"template": template.name, "window": (lo, hi), "coverage": coverage}

    if hi < coverage[0] or lo > coverage[1]:
        warnings.append(
            NeoFitWarning(
                code="iron_template_no_overlap",
                message="Iron template has no overlap with the requested local fitting window.",
                severity="error",
                context=context,
            )
        )
        return PreparedIronTemplate(template, np.zeros_like(wave_rest_fit), fwhm_kms, warnings)

    if lo < coverage[0] or hi > coverage[1]:
        warnings.append(
            NeoFitWarning(
                code="iron_template_partial_coverage",
                message="Iron template only partially covers the requested local fitting window.",
                context=context,
            )
        )

    basis = evaluate_iron_basis(template, wave_rest_fit, fwhm_kms, velocity_step_kms=velocity_step_kms)
    return PreparedIronTemplate(template, basis, float(fwhm_kms), warnings)
