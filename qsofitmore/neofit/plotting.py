"""Lightweight plotting helpers for neofit result objects."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .result import FitResult, LocalFitResult


def _setup_matplotlib():
    import matplotlib.pyplot as plt

    return plt


def _percentile_limits(values, percentiles=(1.0, 99.0), pad_fraction=0.08):
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
    pad = (hi - lo) * pad_fraction if hi != lo else (abs(lo) * pad_fraction if lo != 0 else 1.0)
    return lo - pad, hi + pad


def _with_gap_breaks(wave, values, gap_factor=5.0):
    wave = np.asarray(wave, dtype=float)
    values = np.asarray(values, dtype=float)
    if wave.size < 3:
        return wave, values
    steps = np.diff(wave)
    finite_steps = steps[np.isfinite(steps) & (steps > 0)]
    if finite_steps.size == 0:
        return wave, values
    typical = float(np.nanmedian(finite_steps))
    if not np.isfinite(typical) or typical <= 0:
        return wave, values
    breaks = np.where(steps > gap_factor * typical)[0]
    if breaks.size == 0:
        return wave, values
    wave_out = wave.astype(float).copy()
    values_out = values.astype(float).copy()
    for index in breaks[::-1]:
        wave_out = np.insert(wave_out, index + 1, np.nan)
        values_out = np.insert(values_out, index + 1, np.nan)
    return wave_out, values_out


def plot_line_result(
    result: FitResult,
    output_path: Optional[str] = None,
    ax=None,
    show_components: bool = True,
    title: Optional[str] = None,
):
    """Plot one fitted local line-complex result.

    Returns the Matplotlib axes when ``output_path`` is not provided, otherwise
    saves the figure and returns the written path.
    """

    plt = _setup_matplotlib()
    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    else:
        fig = ax.figure

    if not result.success or result.wave_rest_fit.size == 0:
        ax.text(0.5, 0.5, result.message or "fit failed", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        if result.wave_rest_window.size:
            wave = result.wave_rest_window
            flux = result.flux_window
            model = result.model_window
            component_models = result.component_models_window or result.component_models
        else:
            wave = result.wave_rest_fit
            flux = result.flux_fit
            model = result.model
            component_models = result.component_models

        plot_window = result.metadata.get("plot_window")
        if plot_window is not None and len(plot_window) == 2:
            lo, hi = map(float, plot_window)
            view = (wave >= lo) & (wave <= hi)
            if np.any(view):
                wave = wave[view]
                flux = flux[view]
                model = model[view]
                component_models = {name: np.asarray(component)[view] for name, component in component_models.items()}
        else:
            lo = hi = None

        plot_wave, plot_flux = _with_gap_breaks(wave, flux)
        _, plot_model = _with_gap_breaks(wave, model)
        ax.plot(plot_wave, plot_flux, color="0.25", lw=0.9, label="data")
        ax.plot(plot_wave, plot_model, color="tab:blue", lw=1.2, label="model")
        if show_components:
            for name, component in component_models.items():
                component_wave, component_values = _with_gap_breaks(wave, component)
                if name == "continuum":
                    ax.plot(component_wave, component_values, color="tab:orange", lw=1.0, ls="--", label="continuum")
                else:
                    ax.plot(component_wave, component_values, lw=0.9, ls=":", label=name)
        limits = _percentile_limits([flux, model])
        if limits is not None:
            ax.set_ylim(*limits)
        if plot_window is not None and len(plot_window) == 2:
            ax.set_xlim(float(plot_window[0]), float(plot_window[1]))
        ax.set_xlabel("Rest wavelength [Angstrom]", fontsize=10)
        ax.set_ylabel(f"Flux density [{result.metadata.get('flux_density_unit', 'input')}]", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.legend(loc="best", fontsize=8)
    if title:
        ax.set_title(title, fontsize=11)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
        if owns_figure:
            plt.close(fig)
        return str(path)
    return ax


def plot_local_result(
    result: LocalFitResult,
    output_path: Optional[str] = None,
    show_components: bool = True,
):
    """Plot all windows from a local fit in stacked panels."""

    plt = _setup_matplotlib()
    n_windows = max(len(result.window_results), 1)
    fig, axes = plt.subplots(n_windows, 1, figsize=(8, 3.2 * n_windows), squeeze=False, constrained_layout=True)
    unit = "input"
    for ax, (window, fit) in zip(axes.ravel(), result.window_results.items()):
        plot_line_result(fit, ax=ax, show_components=show_components, title=window)
        unit = fit.metadata.get("flux_density_unit", unit)
        ax.set_ylabel("")
    fig.supylabel(f"Flux density [{unit}]", fontsize=10)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return str(path)
    return axes.ravel()


def save_local_window_plots(
    result: LocalFitResult,
    output_dir: str,
    show_components: bool = True,
) -> Dict[str, str]:
    """Write one PNG per local-fit window and return ``{window: path}``."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for window, fit in result.window_results.items():
        files[window] = plot_line_result(
            fit,
            output_path=str(out / f"{window}_neofit.png"),
            show_components=show_components,
            title=window,
        )
    return files
