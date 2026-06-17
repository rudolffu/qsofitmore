"""Tests for neofit spectrum handling."""

import numpy as np
import pytest

from qsofitmore.neofit import Spectrum


def test_spectrum_from_arrays_validates_shapes():
    wave = np.arange(5.0)
    flux = np.ones(4)
    err = np.ones(5)

    with pytest.raises(ValueError, match="same shape"):
        Spectrum.from_arrays(wave, flux, err=err, z=0.1)


def test_spectrum_from_arrays_valid_mask_and_rest_frame():
    wave_rest = np.array([4000.0, 4100.0, 4200.0, 4300.0])
    flux = np.array([1.0, np.nan, 3.0, 4.0])
    err = np.array([0.1, 0.1, -1.0, 0.2])
    spec = Spectrum.from_arrays(wave_rest, flux, err=err, z=0.5, wave_frame="rest")

    np.testing.assert_allclose(spec.wave_rest, wave_rest)
    np.testing.assert_allclose(spec.wave_obs, wave_rest * 1.5)
    np.testing.assert_array_equal(spec.valid_mask, [True, False, False, True])


def test_spectrum_accepts_ivar():
    wave = np.array([4000.0, 4100.0])
    flux = np.ones(2)
    ivar = np.array([4.0, 0.0])
    spec = Spectrum.from_arrays(wave, flux, ivar=ivar, z=0.0)

    np.testing.assert_allclose(spec.err[0], 0.5)
    assert not spec.valid_mask[1]
