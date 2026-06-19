"""Diagnostic plots for optional pPXF host-decomposition workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .euclid import EuclidHostPrediction
from .ppxf_host import HostSED, PPXFHostFitResult


def _setup_matplotlib():
    import matplotlib.pyplot as plt

    return plt


def _finite_percentile_limits(values, percentiles=(1.0, 99.0), pad_fraction=0.08):
    arrays = [np.ravel(np.asarray(value, dtype=float)) for value in values if value is not None]
    if not arrays:
        return None
    data = np.concatenate(arrays)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return None
    lo, hi = np.nanpercentile(data, percentiles)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if lo == hi:
        pad = abs(lo) * pad_fraction if lo != 0 else 1.0
    else:
        pad = (hi - lo) * pad_fraction
    return lo - pad, hi + pad


def _host_sed_prediction_on_desi_grid(fit: PPXFHostFitResult, host_sed: Optional[HostSED]) -> Optional[np.ndarray]:
    if host_sed is None:
        return None
    wave = fit.preprocessed.wave_rest
    predicted = np.interp(wave, host_sed.wave_rest, host_sed.host_flux, left=np.nan, right=np.nan)
    finite_fit_host = np.isfinite(fit.host_model)
    outside_fit = np.ones_like(wave, dtype=bool)
    if np.any(finite_fit_host):
        fit_min = np.nanmin(wave[finite_fit_host])
        fit_max = np.nanmax(wave[finite_fit_host])
        outside_fit = (wave < fit_min) | (wave > fit_max)
    return np.where(outside_fit & np.isfinite(predicted), predicted, np.nan)


def plot_desi_ppxf_fit(fit: PPXFHostFitResult, output_path: str, host_sed: Optional[HostSED] = None) -> str:
    plt = _setup_matplotlib()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    wave = fit.preprocessed.wave_rest
    axes[0].plot(wave, fit.preprocessed.flux, color="0.2", lw=0.8, label="DESI spectrum")
    axes[0].plot(wave, fit.host_model, color="tab:green", lw=1.0, label="pPXF host")
    predicted_host = _host_sed_prediction_on_desi_grid(fit, host_sed)
    if predicted_host is not None and np.any(np.isfinite(predicted_host)):
        axes[0].plot(wave, predicted_host, color="tab:green", lw=1.0, ls="--", label="host SED prediction")
    axes[0].plot(wave, fit.agn_model, color="tab:orange", lw=1.0, label="AGN continuum")
    axes[0].plot(wave, fit.total_model, color="tab:blue", lw=1.0, label="total continuum")
    masked = fit.preprocessed.emission_mask
    if np.any(masked):
        ymin, ymax = np.nanpercentile(fit.preprocessed.flux, [2, 98])
        axes[0].fill_between(wave, ymin, ymax, where=masked, color="tab:red", alpha=0.12, label="masked lines")
    flux_limits = _finite_percentile_limits(
        [fit.preprocessed.flux, fit.host_model, predicted_host, fit.agn_model, fit.total_model],
        percentiles=(1.0, 99.0),
    )
    if flux_limits is not None:
        axes[0].set_ylim(*flux_limits)
    axes[0].legend(loc="best", fontsize=8)
    axes[0].set_ylabel("Flux density [input units]")
    axes[1].plot(wave, fit.residual, color="0.25", lw=0.8)
    axes[1].axhline(0.0, color="0.7", lw=0.8)
    residual_limits = _finite_percentile_limits([fit.residual], percentiles=(1.0, 99.0))
    if residual_limits is not None:
        axes[1].set_ylim(*residual_limits)
    axes[1].set_xlabel("Rest wavelength [Angstrom]")
    axes[1].set_ylabel("Residual")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def plot_host_sed_prediction(sed: HostSED, output_path: str) -> str:
    plt = _setup_matplotlib()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ax.plot(sed.wave_rest, sed.host_flux, color="tab:green", lw=1.0)
    for wave, label in [(5100, "5100A"), (10000, "1.0um"), (16000, "1.6um"), (22000, "2.2um")]:
        ax.axvline(wave, color="0.7", ls="--", lw=0.8)
        ax.text(wave, 0.98, label, rotation=90, transform=ax.get_xaxis_transform(), va="top", ha="right", fontsize=8)
    limits = _finite_percentile_limits([sed.host_flux], percentiles=(1.0, 99.0))
    if limits is not None:
        ax.set_ylim(*limits)
    ax.set_xlabel("Rest wavelength [Angstrom]")
    ax.set_ylabel("Host flux density [input units]")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def plot_euclid_prediction(prediction: EuclidHostPrediction, output_path: str) -> str:
    plt = _setup_matplotlib()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    if prediction.euclid_flux is not None:
        ax.plot(prediction.wave_obs, prediction.euclid_flux, color="0.2", lw=0.8, label="Euclid spectrum")
    ax.plot(prediction.wave_obs, prediction.predicted_host_flux, color="tab:green", lw=1.0, label="predicted host")
    if prediction.host_subtracted_flux is not None:
        ax.plot(prediction.wave_obs, prediction.host_subtracted_flux, color="tab:blue", lw=0.8, label="host-subtracted")
    limits = _finite_percentile_limits(
        [prediction.euclid_flux, prediction.predicted_host_flux, prediction.host_subtracted_flux],
        percentiles=(1.0, 99.0),
    )
    if limits is not None:
        ax.set_ylim(*limits)
    ax.set_xlabel("Observed wavelength [Angstrom]")
    ax.set_ylabel("Flux density [input units]")
    ax.legend(loc="best", fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)
