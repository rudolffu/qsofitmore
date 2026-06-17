"""Packed parameter-vector helpers for neofit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import GaussianComponent, LineComplexConfig
from .templates.iron import IronTemplate


@dataclass(frozen=True)
class ComponentIndex:
    """Parameter indices for one Gaussian component."""

    name: str
    amp: int
    center: int
    sigma: int


@dataclass(frozen=True)
class PackedParameters:
    """Packed optimizer-facing parameter vector metadata."""

    names: Tuple[str, ...]
    initial: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    components: Tuple[ComponentIndex, ...]
    continuum_mode: Optional[str]
    continuum_indices: Tuple[int, ...]
    wave_ref: float
    wave_scale: float
    iron_index: Optional[int] = None
    iron_fwhm_index: Optional[int] = None
    iron_basis: Optional[np.ndarray] = None
    iron_template: Optional[IronTemplate] = None
    iron_template_name: Optional[str] = None
    iron_velocity_step_kms: float = 25.0

    def unpack(self, theta: np.ndarray) -> Dict[str, float]:
        """Return ``{parameter_name: value}`` for a packed vector."""

        theta = np.asarray(theta, dtype=float)
        if theta.shape != self.initial.shape:
            raise ValueError("theta has the wrong shape for this PackedParameters object.")
        return {name: float(theta[i]) for i, name in enumerate(self.names)}


def _bound(component: GaussianComponent, field_name: str, default: Tuple[float, float]) -> Tuple[float, float]:
    lo, hi = component.bounds.get(field_name, default)
    lo = -np.inf if lo is None else float(lo)
    hi = np.inf if hi is None else float(hi)
    if hi <= lo:
        raise ValueError(f"Invalid bounds for {component.name}.{field_name}: upper <= lower.")
    return lo, hi


def _clip_initial(value: float, lo: float, hi: float) -> float:
    return float(np.clip(value, lo, hi))


def _normalize_bounds(bounds: Tuple[Optional[float], Optional[float]], label: str) -> Tuple[float, float]:
    lo, hi = bounds
    lo = -np.inf if lo is None else float(lo)
    hi = np.inf if hi is None else float(hi)
    if hi <= lo:
        raise ValueError(f"Invalid bounds for {label}: upper <= lower.")
    return lo, hi


def pack_line_complex_parameters(
    config: LineComplexConfig,
    wave_fit: np.ndarray,
    flux_fit: Optional[np.ndarray] = None,
    iron_basis: Optional[np.ndarray] = None,
    iron_template: Optional[IronTemplate] = None,
    iron_template_name: Optional[str] = None,
) -> PackedParameters:
    """Create initial theta, bounds, and index metadata for one line complex."""

    names: List[str] = []
    initial: List[float] = []
    lower: List[float] = []
    upper: List[float] = []
    components: List[ComponentIndex] = []

    for component in config.components:
        amp_lo, amp_hi = _bound(component, "amp", (-np.inf, np.inf))
        center_lo, center_hi = _bound(component, "center", (-np.inf, np.inf))
        sigma_lo, sigma_hi = _bound(component, "sigma", (1.0e-12, np.inf))

        amp_idx = len(names)
        names.append(f"{component.name}.amp")
        initial.append(_clip_initial(component.amp, amp_lo, amp_hi))
        lower.append(amp_lo)
        upper.append(amp_hi)

        center_idx = len(names)
        names.append(f"{component.name}.center")
        initial.append(_clip_initial(component.center, center_lo, center_hi))
        lower.append(center_lo)
        upper.append(center_hi)

        sigma_idx = len(names)
        names.append(f"{component.name}.sigma")
        initial.append(_clip_initial(component.sigma, sigma_lo, sigma_hi))
        lower.append(sigma_lo)
        upper.append(sigma_hi)

        components.append(ComponentIndex(component.name, amp_idx, center_idx, sigma_idx))

    wave_fit = np.asarray(wave_fit, dtype=float)
    wave_ref = float(config.center)
    half_window = 0.5 * (float(config.window[1]) - float(config.window[0]))
    wave_scale = half_window if half_window > 0 else max(float(np.nanstd(wave_fit)), 1.0)
    continuum_indices: List[int] = []

    if config.local_continuum is not None:
        c0 = 0.0
        if flux_fit is not None:
            finite = np.asarray(flux_fit, dtype=float)
            finite = finite[np.isfinite(finite)]
            if finite.size:
                c0 = float(np.nanmedian(finite))
        c0_idx = len(names)
        names.append("continuum.c0")
        initial.append(c0)
        lower.append(-np.inf)
        upper.append(np.inf)
        continuum_indices.append(c0_idx)
        if config.local_continuum == "linear":
            c1_idx = len(names)
            names.append("continuum.c1")
            initial.append(0.0)
            lower.append(-np.inf)
            upper.append(np.inf)
            continuum_indices.append(c1_idx)

    iron_index = None
    iron_fwhm_index = None
    if config.iron is not None and config.iron.enabled and (iron_basis is not None or iron_template is not None):
        basis = None if iron_basis is None else np.asarray(iron_basis, dtype=float)
        if basis is not None and basis.shape != wave_fit.shape:
            raise ValueError("iron_basis must have the same shape as wave_fit.")
        amp_lo, amp_hi = _normalize_bounds(config.iron.amp_bounds, "iron.amp")
        iron_index = len(names)
        names.append("iron.amp")
        initial.append(_clip_initial(config.iron.amp, amp_lo, amp_hi))
        lower.append(amp_lo)
        upper.append(amp_hi)
        fwhm_lo, fwhm_hi = _normalize_bounds(config.iron.fwhm_bounds, "iron.fwhm_kms")
        iron_fwhm_index = len(names)
        names.append("iron.fwhm_kms")
        initial.append(_clip_initial(config.iron.fwhm_kms, fwhm_lo, fwhm_hi))
        lower.append(fwhm_lo)
        upper.append(fwhm_hi)
    else:
        basis = None

    return PackedParameters(
        names=tuple(names),
        initial=np.asarray(initial, dtype=float),
        lower=np.asarray(lower, dtype=float),
        upper=np.asarray(upper, dtype=float),
        components=tuple(components),
        continuum_mode=config.local_continuum,
        continuum_indices=tuple(continuum_indices),
        wave_ref=wave_ref,
        wave_scale=float(wave_scale),
        iron_index=iron_index,
        iron_fwhm_index=iron_fwhm_index,
        iron_basis=basis,
        iron_template=iron_template,
        iron_template_name=iron_template_name,
    )
