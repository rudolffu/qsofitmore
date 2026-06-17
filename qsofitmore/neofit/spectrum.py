"""Spectrum container for neofit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .metadata import SpectrumMetadata, resolve_spectrum_metadata


@dataclass(frozen=True)
class Spectrum:
    """Array-only spectrum input for neofit."""

    wave_obs: np.ndarray
    flux: np.ndarray
    err: np.ndarray
    z: float
    metadata: SpectrumMetadata = field(default_factory=SpectrumMetadata)
    mask: Optional[np.ndarray] = None

    @classmethod
    def from_arrays(
        cls,
        wave: np.ndarray,
        flux: np.ndarray,
        err: Optional[np.ndarray] = None,
        ivar: Optional[np.ndarray] = None,
        z: float = 0.0,
        wave_frame: str = "observed",
        mask: Optional[np.ndarray] = None,
        survey: Optional[str] = None,
        unit_preset: Optional[str] = None,
        wave_unit: Optional[str] = None,
        flux_density_unit: Optional[str] = None,
        flux_density_scale_to_cgs: Optional[float] = None,
        source: Optional[str] = None,
        metadata: Optional[SpectrumMetadata] = None,
    ) -> "Spectrum":
        """Build a spectrum from plain arrays.

        ``wave_frame`` may be ``"observed"`` or ``"rest"``. Internally the
        observed wavelength is stored and rest wavelength is derived from ``z``.
        """

        wave = np.asarray(wave, dtype=float)
        flux = np.asarray(flux, dtype=float)
        if wave.ndim != 1 or flux.ndim != 1:
            raise ValueError("wave and flux must be 1D arrays.")
        if wave.shape != flux.shape:
            raise ValueError("wave and flux must have the same shape.")
        if not np.isfinite(z):
            raise ValueError("z must be finite.")

        if err is None:
            if ivar is None:
                raise ValueError("Either err or ivar must be provided.")
            ivar = np.asarray(ivar, dtype=float)
            if ivar.shape != wave.shape:
                raise ValueError("ivar must have the same shape as wave.")
            err_arr = np.full_like(ivar, np.inf, dtype=float)
            good = ivar > 0
            err_arr[good] = 1.0 / np.sqrt(ivar[good])
        else:
            err_arr = np.asarray(err, dtype=float)
            if err_arr.shape != wave.shape:
                raise ValueError("err must have the same shape as wave.")

        mask_arr = None
        if mask is not None:
            mask_arr = np.asarray(mask, dtype=bool)
            if mask_arr.shape != wave.shape:
                raise ValueError("mask must have the same shape as wave.")

        frame = str(wave_frame).strip().lower()
        if frame == "observed":
            wave_obs = wave
        elif frame == "rest":
            wave_obs = wave * (1.0 + float(z))
        else:
            raise ValueError("wave_frame must be 'observed' or 'rest'.")

        return cls(
            wave_obs=wave_obs.copy(),
            flux=flux.copy(),
            err=err_arr.copy(),
            z=float(z),
            metadata=resolve_spectrum_metadata(
                survey=survey,
                unit_preset=unit_preset,
                wave_unit=wave_unit,
                flux_density_unit=flux_density_unit,
                flux_density_scale_to_cgs=flux_density_scale_to_cgs,
                source=source,
                metadata=metadata,
            ),
            mask=None if mask_arr is None else mask_arr.copy(),
        )

    @property
    def wave_rest(self) -> np.ndarray:
        """Rest-frame wavelength array."""

        return self.wave_obs / (1.0 + self.z)

    @property
    def valid_mask(self) -> np.ndarray:
        """Finite, positive-error pixels allowed for fitting."""

        valid = (
            np.isfinite(self.wave_obs)
            & np.isfinite(self.flux)
            & np.isfinite(self.err)
            & (self.wave_obs > 0)
            & (self.err > 0)
        )
        if self.mask is not None:
            valid &= self.mask
        return valid

    @property
    def wave_unit(self) -> str:
        """Wavelength unit label."""

        return self.metadata.wave_unit

    @property
    def flux_density_unit(self) -> str:
        """Flux-density unit label for the input arrays."""

        return self.metadata.flux_density_unit

    @property
    def flux_density_scale_to_cgs(self) -> Optional[float]:
        """Multiplicative scale from input flux density to cgs, if known."""

        return self.metadata.flux_density_scale_to_cgs
