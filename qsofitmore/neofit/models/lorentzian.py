"""Lorentzian model primitives for neofit."""

from __future__ import annotations

import numpy as np


def lorentzian(wave: np.ndarray, amp: float, center: float, gamma: float) -> np.ndarray:
    """Evaluate a peak-normalized Lorentzian profile."""

    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("Lorentzian gamma must be positive and finite.")
    wave = np.asarray(wave, dtype=float)
    delta = wave - float(center)
    gamma2 = float(gamma) ** 2
    return float(amp) * gamma2 / (delta * delta + gamma2)


def lorentzian_partials(wave: np.ndarray, amp: float, center: float, gamma: float) -> np.ndarray:
    """Return dense derivatives with columns ``amp, center, gamma``."""

    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("Lorentzian gamma must be positive and finite.")
    wave = np.asarray(wave, dtype=float)
    amp = float(amp)
    center = float(center)
    gamma = float(gamma)
    delta = wave - center
    gamma2 = gamma * gamma
    denom = delta * delta + gamma2
    out = np.empty((wave.size, 3), dtype=float)
    out[:, 0] = gamma2 / denom
    out[:, 1] = 2.0 * amp * gamma2 * delta / denom**2
    out[:, 2] = 2.0 * amp * gamma * delta**2 / denom**2
    return out
