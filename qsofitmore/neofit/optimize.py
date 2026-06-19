"""Optimizer wrappers for neofit."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import least_squares

from .jacobian import residual_jacobian_dense, residual_jacobian_sparse
from .parameters import PackedParameters
from .residuals import weighted_residual


def run_least_squares(
    packed: PackedParameters,
    wave: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    jacobian: str = "analytic_dense",
    max_nfev: Optional[int] = None,
):
    """Run SciPy least-squares for a packed local line-complex model."""

    if jacobian == "analytic_dense":
        jac = lambda theta: residual_jacobian_dense(theta, packed, wave, err)
    elif jacobian == "analytic_sparse":
        jac = lambda theta: residual_jacobian_sparse(theta, packed, wave, err)
    elif jacobian == "finite_difference":
        jac = "2-point"
    else:
        raise ValueError("jacobian must be 'analytic_dense', 'analytic_sparse', or 'finite_difference'.")

    return least_squares(
        lambda theta: weighted_residual(theta, packed, wave, flux, err),
        packed.initial,
        bounds=(packed.lower, packed.upper),
        jac=jac,
        max_nfev=max_nfev,
    )
