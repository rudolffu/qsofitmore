"""Gaussian model primitives for neofit."""

from __future__ import annotations

import numpy as np


def gaussian(wave: np.ndarray, amp: float, center: float, sigma: float) -> np.ndarray:
    """Evaluate ``amp * exp(-0.5 * ((wave - center) / sigma)**2)``."""

    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Gaussian sigma must be positive and finite.")
    wave = np.asarray(wave, dtype=float)
    u = (wave - float(center)) / float(sigma)
    return float(amp) * np.exp(-0.5 * u * u)


def gaussian_partials(wave: np.ndarray, amp: float, center: float, sigma: float) -> np.ndarray:
    """Return dense model derivatives with columns ``amp, center, sigma``."""

    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Gaussian sigma must be positive and finite.")
    wave = np.asarray(wave, dtype=float)
    amp = float(amp)
    center = float(center)
    sigma = float(sigma)
    delta = wave - center
    expo = np.exp(-0.5 * (delta / sigma) ** 2)
    model = amp * expo
    out = np.empty((wave.size, 3), dtype=float)
    out[:, 0] = expo
    out[:, 1] = model * delta / sigma**2
    out[:, 2] = model * delta**2 / sigma**3
    return out
