"""Local continuum primitives for neofit."""

from __future__ import annotations

from typing import Optional

import numpy as np


def normalized_coordinate(wave: np.ndarray, wave_ref: float, scale: float) -> np.ndarray:
    """Return a stable local coordinate for a small wavelength window."""

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Continuum normalization scale must be positive and finite.")
    return (np.asarray(wave, dtype=float) - float(wave_ref)) / float(scale)


def continuum(wave: np.ndarray, mode: Optional[str], params: np.ndarray, wave_ref: float, scale: float) -> np.ndarray:
    """Evaluate the MVP local continuum model."""

    if mode is None:
        return np.zeros_like(np.asarray(wave, dtype=float), dtype=float)
    if mode == "constant":
        return np.full_like(np.asarray(wave, dtype=float), float(params[0]), dtype=float)
    if mode == "linear":
        x = normalized_coordinate(wave, wave_ref, scale)
        return float(params[0]) + float(params[1]) * x
    raise ValueError(f"Unsupported local continuum mode: {mode!r}")


def continuum_partials(wave: np.ndarray, mode: Optional[str], wave_ref: float, scale: float) -> np.ndarray:
    """Return dense model derivatives for the local continuum parameters."""

    wave = np.asarray(wave, dtype=float)
    if mode is None:
        return np.zeros((wave.size, 0), dtype=float)
    if mode == "constant":
        return np.ones((wave.size, 1), dtype=float)
    if mode == "linear":
        x = normalized_coordinate(wave, wave_ref, scale)
        out = np.empty((wave.size, 2), dtype=float)
        out[:, 0] = 1.0
        out[:, 1] = x
        return out
    raise ValueError(f"Unsupported local continuum mode: {mode!r}")
