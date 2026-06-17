"""Euclid host-contamination prediction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .ppxf_host import HostSED


@dataclass
class EuclidHostPrediction:
    wave_obs: np.ndarray
    wave_rest: np.ndarray
    euclid_flux: Optional[np.ndarray]
    predicted_host_flux: np.ndarray
    host_subtracted_flux: Optional[np.ndarray]
    scale_factor: float
    scale_mode: str
    warnings: list


def _continuum_mask(wave_rest: np.ndarray, windows: Sequence[Tuple[float, float]]) -> np.ndarray:
    mask = np.zeros_like(wave_rest, dtype=bool)
    for lo, hi in windows:
        mask |= (wave_rest >= lo) & (wave_rest <= hi)
    return mask


def _nonnegative_scale(model: np.ndarray, flux: np.ndarray, mask: np.ndarray) -> float:
    good = mask & np.isfinite(model) & np.isfinite(flux) & (model > 0)
    if not np.any(good):
        return 0.0
    denom = float(np.sum(model[good] ** 2))
    if denom <= 0:
        return 0.0
    return max(0.0, float(np.sum(model[good] * flux[good]) / denom))


def predict_host_for_euclid_spectrum(
    desi_host_sed: HostSED,
    euclid_wave_obs: np.ndarray,
    z: float,
    euclid_flux: Optional[np.ndarray] = None,
    scale_mode: str = "free_scale",
    aperture_scale: Optional[float] = None,
    continuum_windows: Sequence[Tuple[float, float]] = ((10000.0, 12000.0), (14500.0, 17000.0)),
) -> EuclidHostPrediction:
    """Interpolate a DESI-derived host SED onto a Euclid observed grid."""

    wave_obs = np.asarray(euclid_wave_obs, dtype=float)
    wave_rest = wave_obs / (1.0 + float(z))
    warnings = []
    host = np.interp(wave_rest, desi_host_sed.wave_rest, desi_host_sed.host_flux, left=np.nan, right=np.nan)
    if np.any(~np.isfinite(host)):
        warnings.append("euclid_grid_extends_beyond_host_sed_coverage")

    mode = scale_mode.lower()
    if mode == "desi_fiber_scaled":
        scale = 1.0 if aperture_scale is None else float(aperture_scale)
    elif mode == "image_prior_scale":
        if aperture_scale is None:
            raise ValueError("image_prior_scale requires aperture_scale.")
        scale = float(aperture_scale)
    elif mode == "free_scale":
        if euclid_flux is None:
            warnings.append("free_scale_without_euclid_flux_uses_unity")
            scale = 1.0 if aperture_scale is None else float(aperture_scale)
        else:
            mask = _continuum_mask(wave_rest, continuum_windows)
            scale = _nonnegative_scale(host, np.asarray(euclid_flux, dtype=float), mask)
    else:
        raise ValueError(f"Unknown Euclid scale_mode: {scale_mode}")

    scaled_host = host * scale
    flux = None if euclid_flux is None else np.asarray(euclid_flux, dtype=float)
    subtracted = None if flux is None else flux - scaled_host
    return EuclidHostPrediction(
        wave_obs=wave_obs,
        wave_rest=wave_rest,
        euclid_flux=flux,
        predicted_host_flux=scaled_host,
        host_subtracted_flux=subtracted,
        scale_factor=scale,
        scale_mode=mode,
        warnings=warnings,
    )


def write_euclid_prediction(prediction: EuclidHostPrediction, output_dir: str) -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "euclid_host_prediction.csv"
    data = {
        "wave_obs": prediction.wave_obs,
        "wave_rest": prediction.wave_rest,
        "predicted_host_flux": prediction.predicted_host_flux,
    }
    if prediction.euclid_flux is not None:
        data["euclid_flux"] = prediction.euclid_flux
    if prediction.host_subtracted_flux is not None:
        data["host_subtracted_flux"] = prediction.host_subtracted_flux
    pd.DataFrame(data).to_csv(path, index=False)
    return str(path)
