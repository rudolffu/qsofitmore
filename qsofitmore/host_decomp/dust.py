"""Dust/extinction helpers for host-decomposition preprocessing."""

from __future__ import annotations

from typing import Optional

import numpy as np

from qsofitmore.extinction import deredden, f99law


def apply_galactic_dereddening(
    wave_obs: np.ndarray,
    flux: np.ndarray,
    ebv: Optional[float] = None,
    rv: float = 3.1,
) -> np.ndarray:
    """Apply Galactic dereddening when an E(B-V) value is supplied."""

    if ebv is None or float(ebv) == 0.0:
        return np.asarray(flux, dtype=float)
    alam = f99law(np.asarray(wave_obs, dtype=float), float(ebv), Rv=rv)
    return deredden(alam, np.asarray(flux, dtype=float))
