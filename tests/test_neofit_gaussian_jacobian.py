"""Finite-difference checks for neofit analytic Gaussian derivatives."""

import numpy as np

from qsofitmore.neofit.models.gaussian import gaussian, gaussian_partials


def test_gaussian_partials_match_finite_difference():
    wave = np.linspace(4800.0, 4920.0, 25)
    theta = np.array([8.0, 4861.3, 22.0])
    eps = np.array([1e-6, 1e-5, 1e-5])

    analytic = gaussian_partials(wave, theta[0], theta[1], theta[2])
    finite_difference = np.empty_like(analytic)
    for i in range(3):
        hi = theta.copy()
        lo = theta.copy()
        hi[i] += eps[i]
        lo[i] -= eps[i]
        finite_difference[:, i] = (
            gaussian(wave, hi[0], hi[1], hi[2]) - gaussian(wave, lo[0], lo[1], lo[2])
        ) / (2.0 * eps[i])

    np.testing.assert_allclose(analytic, finite_difference, rtol=1e-5, atol=1e-7)
