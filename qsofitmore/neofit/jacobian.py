"""Analytic Jacobians for neofit residual functions."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .models.continuum import continuum_partials
from .models.gaussian import gaussian_partials
from .parameters import PackedParameters
from .residuals import iron_basis_vector
from .templates.iron import evaluate_iron_basis


def _iron_fwhm_derivative(theta: np.ndarray, packed: PackedParameters, wave: np.ndarray) -> np.ndarray:
    """Finite-difference derivative of iron basis with respect to FWHM."""

    if packed.iron_template is None or packed.iron_fwhm_index is None:
        return np.zeros_like(wave, dtype=float)
    fwhm = float(theta[packed.iron_fwhm_index])
    lower = float(packed.lower[packed.iron_fwhm_index])
    upper = float(packed.upper[packed.iron_fwhm_index])
    step = max(abs(fwhm) * 1.0e-4, 0.5)
    lo = max(fwhm - step, lower) if np.isfinite(lower) else fwhm - step
    hi = min(fwhm + step, upper) if np.isfinite(upper) else fwhm + step
    if hi <= lo:
        return np.zeros_like(wave, dtype=float)
    basis_lo = evaluate_iron_basis(
        packed.iron_template,
        wave,
        fwhm_kms=lo,
        velocity_step_kms=packed.iron_velocity_step_kms,
    )
    basis_hi = evaluate_iron_basis(
        packed.iron_template,
        wave,
        fwhm_kms=hi,
        velocity_step_kms=packed.iron_velocity_step_kms,
    )
    return (basis_hi - basis_lo) / (hi - lo)


def model_jacobian_dense(theta: np.ndarray, packed: PackedParameters, wave: np.ndarray) -> np.ndarray:
    """Return dense ``dmodel/dtheta`` for one local line complex."""

    theta = np.asarray(theta, dtype=float)
    wave = np.asarray(wave, dtype=float)
    jac = np.zeros((wave.size, theta.size), dtype=float)
    for component in packed.components:
        part = gaussian_partials(wave, theta[component.amp], theta[component.center], theta[component.sigma])
        jac[:, component.amp] = part[:, 0]
        jac[:, component.center] = part[:, 1]
        jac[:, component.sigma] = part[:, 2]
    iron_basis = iron_basis_vector(theta, packed, wave)
    if packed.iron_index is not None and iron_basis is not None:
        jac[:, packed.iron_index] = iron_basis
        if packed.iron_fwhm_index is not None:
            jac[:, packed.iron_fwhm_index] = theta[packed.iron_index] * _iron_fwhm_derivative(theta, packed, wave)
    if packed.continuum_mode is not None:
        cont_part = continuum_partials(wave, packed.continuum_mode, packed.wave_ref, packed.wave_scale)
        for local_col, param_idx in enumerate(packed.continuum_indices):
            jac[:, param_idx] = cont_part[:, local_col]
    return jac


def residual_jacobian_dense(
    theta: np.ndarray,
    packed: PackedParameters,
    wave: np.ndarray,
    err: np.ndarray,
) -> np.ndarray:
    """Return dense ``d((flux - model) / err)/dtheta``."""

    return -model_jacobian_dense(theta, packed, wave) / np.asarray(err, dtype=float)[:, np.newaxis]


def residual_jacobian_sparse(
    theta: np.ndarray,
    packed: PackedParameters,
    wave: np.ndarray,
    err: np.ndarray,
) -> sparse.csr_matrix:
    """Return CSR residual Jacobian.

    TODO: assemble true block-sparse Jacobians when fitting multiple independent
    line windows. For the MVP, a single local complex is small enough that the
    dense analytic derivative can be converted to CSR.
    """

    return sparse.csr_matrix(residual_jacobian_dense(theta, packed, wave, err))
