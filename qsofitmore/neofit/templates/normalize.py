"""Normalization and parsing utilities for neofit iron templates."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def read_two_column_template(path: str, wave_col: int = 0, flux_col: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Read a text template file with at least wavelength and flux columns."""

    template_path = Path(path).expanduser()
    if not template_path.exists():
        raise FileNotFoundError(f"Iron template file not found: {template_path}")
    suffix = template_path.suffix.lower()
    if suffix in (".fits", ".fit", ".fz") or template_path.name.lower().endswith(".fits.gz"):
        try:
            from astropy.io import fits
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Reading FITS iron templates requires astropy.") from exc
        with fits.open(template_path) as hdul:
            data = np.asarray(hdul[0].data, dtype=float).ravel()
            hdr = hdul[0].header
            crval = float(hdr.get("CRVAL1", 0.0))
            cdelt = float(hdr.get("CDELT1", 1.0))
            crpix = float(hdr.get("CRPIX1", 1.0))
            pix = np.arange(data.size, dtype=float) + 1.0
            wave = crval + (pix - crpix) * cdelt
        return wave, data

    try:
        data = np.genfromtxt(template_path, comments="#")
    except Exception as exc:
        raise ValueError(f"Cannot parse iron template: {template_path}") from exc
    if data.ndim != 2 or data.shape[1] <= max(wave_col, flux_col):
        raise ValueError(
            f"Cannot parse iron template {template_path}: expected at least "
            f"{max(wave_col, flux_col) + 1} columns."
        )
    return np.asarray(data[:, wave_col], dtype=float), np.asarray(data[:, flux_col], dtype=float)


def validate_template_arrays(wave: np.ndarray, flux: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and return template arrays in their original order."""

    wave = np.asarray(wave, dtype=float).ravel()
    flux = np.asarray(flux, dtype=float).ravel()
    if wave.shape != flux.shape or wave.size < 2:
        raise ValueError("Iron template wavelength and flux arrays must be 1D arrays of the same length.")
    if not np.all(np.isfinite(wave)) or not np.all(np.isfinite(flux)):
        raise ValueError("Iron template contains non-finite wavelength or flux values.")
    if np.any(np.diff(wave) <= 0):
        raise ValueError("Iron template wavelengths must be strictly increasing.")
    return wave, flux


def area_normalize(wave: np.ndarray, flux: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Normalize positive template flux so its native-coverage integral is one."""

    wave, flux = validate_template_arrays(wave, flux)
    positive_flux = np.clip(flux, 0.0, None)
    area = float(np.trapezoid(positive_flux, wave))
    if not np.isfinite(area) or area <= 0:
        raise ValueError("Iron template has zero positive normalization area.")
    return wave, positive_flux / area, area
