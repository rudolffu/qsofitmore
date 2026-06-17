"""Model assembly and weighted residuals for neofit."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .models.continuum import continuum
from .models.gaussian import gaussian
from .parameters import PackedParameters
from .templates.iron import evaluate_iron_basis


def iron_basis_vector(theta: np.ndarray, packed: PackedParameters, wave: np.ndarray) -> Optional[np.ndarray]:
    """Return the current iron basis, evaluating fitted FWHM when configured."""

    if packed.iron_index is None:
        return None
    if packed.iron_template is not None and packed.iron_fwhm_index is not None:
        fwhm_kms = float(theta[packed.iron_fwhm_index])
        return evaluate_iron_basis(
            packed.iron_template,
            wave,
            fwhm_kms=fwhm_kms,
            velocity_step_kms=packed.iron_velocity_step_kms,
        )
    return packed.iron_basis


def model_vector(theta: np.ndarray, packed: PackedParameters, wave: np.ndarray) -> np.ndarray:
    """Evaluate the total local line-complex model."""

    theta = np.asarray(theta, dtype=float)
    wave = np.asarray(wave, dtype=float)
    model = np.zeros_like(wave, dtype=float)
    for component in packed.components:
        model += gaussian(wave, theta[component.amp], theta[component.center], theta[component.sigma])
    iron_basis = iron_basis_vector(theta, packed, wave)
    if iron_basis is not None:
        model += theta[packed.iron_index] * iron_basis
    if packed.continuum_mode is not None:
        cont_theta = theta[list(packed.continuum_indices)]
        model += continuum(wave, packed.continuum_mode, cont_theta, packed.wave_ref, packed.wave_scale)
    return model


def model_components(theta: np.ndarray, packed: PackedParameters, wave: np.ndarray) -> Dict[str, np.ndarray]:
    """Evaluate individual component models on ``wave``."""

    theta = np.asarray(theta, dtype=float)
    wave = np.asarray(wave, dtype=float)
    components: Dict[str, np.ndarray] = {}
    for component in packed.components:
        components[component.name] = gaussian(wave, theta[component.amp], theta[component.center], theta[component.sigma])
    iron_basis = iron_basis_vector(theta, packed, wave)
    if iron_basis is not None:
        components["iron"] = theta[packed.iron_index] * iron_basis
    if packed.continuum_mode is not None:
        cont_theta = theta[list(packed.continuum_indices)]
        components["continuum"] = continuum(wave, packed.continuum_mode, cont_theta, packed.wave_ref, packed.wave_scale)
    return components


def weighted_residual(theta: np.ndarray, packed: PackedParameters, wave: np.ndarray, flux: np.ndarray, err: np.ndarray) -> np.ndarray:
    """Return optimizer residuals using ``(flux - model) / err``."""

    return (np.asarray(flux, dtype=float) - model_vector(theta, packed, wave)) / np.asarray(err, dtype=float)


def model_and_residual(
    theta: np.ndarray,
    packed: PackedParameters,
    wave: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return total model and weighted residual."""

    model = model_vector(theta, packed, wave)
    residual = (np.asarray(flux, dtype=float) - model) / np.asarray(err, dtype=float)
    return model, residual
