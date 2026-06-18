"""Output products and diagnostic plots for the global neofit workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from .global_result import NeoFitWorkflowResult


@dataclass(frozen=True)
class GlobalQAPlotConfig:
    """Rendering options for the global continuum and emission-line QA plot."""

    figure_width: float = 15.3
    figure_height: float = 6.2
    max_zoom_panels: int = 3
    show_smoothed_data: bool = False
    smoothing_window_pixels: int = 7

    def __post_init__(self) -> None:
        if self.figure_width <= 0 or self.figure_height <= 0:
            raise ValueError("QA figure dimensions must be positive.")
        if self.max_zoom_panels < 1:
            raise ValueError("max_zoom_panels must be at least one.")
        if self.smoothing_window_pixels < 1 or self.smoothing_window_pixels % 2 == 0:
            raise ValueError("smoothing_window_pixels must be a positive odd integer.")


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)


def _percentile_limits(values, percentiles: Tuple[float, float] = (1.0, 99.0), pad: float = 0.08):
    arrays = [np.ravel(np.asarray(value, dtype=float)) for value in values if value is not None]
    if not arrays:
        return None
    data = np.concatenate(arrays)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return None
    lo, hi = np.percentile(data, percentiles)
    width = hi - lo
    margin = width * pad if width > 0 else max(abs(lo) * pad, 1.0)
    return float(lo - margin), float(hi + margin)


def _plot_global(result: NeoFitWorkflowResult, path: Path, window: Optional[Tuple[float, float]] = None) -> str:
    import matplotlib.pyplot as plt

    spectrum = result.spectrum
    continuum = result.continuum
    wave = spectrum.wave_rest
    valid = spectrum.valid_mask
    if window is not None:
        valid &= (wave >= window[0]) & (wave <= window[1])
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    ax.plot(wave[valid], spectrum.flux[valid], color="0.35", lw=0.75, label="host-subtracted data")
    ax.plot(wave[valid], continuum.model[valid], color="tab:blue", lw=1.4, label="global continuum")
    colors = {
        "power_law": "tab:orange",
        "uv_iron": "tab:purple",
        "optical_iron": "tab:red",
        "balmer_continuum": "tab:green",
        "balmer_series": "tab:brown",
    }
    for name, component in continuum.component_models.items():
        ax.plot(wave[valid], component[valid], lw=0.9, ls="--", color=colors.get(name), label=name)
    used = continuum.clip_mask & valid
    if np.any(used):
        ax.scatter(wave[used], spectrum.flux[used], s=5, color="k", alpha=0.25, label="fit pixels")
    limits = _percentile_limits([spectrum.flux[valid], continuum.model[valid]])
    if limits:
        ax.set_ylim(*limits)
    if window is not None:
        ax.set_xlim(*window)
    ax.set_xlabel("Rest wavelength [Angstrom]", fontsize=10)
    ax.set_ylabel(f"Flux density [{spectrum.flux_density_unit}]", fontsize=9)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8, ncol=3)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


_COMPLEX_WINDOWS = {
    "mgii": (2700.0, 2900.0),
    "hbeta": (4600.0, 5120.0),
    "halpha": (6400.0, 6800.0),
}
_SPECIES_COLORS = {
    "MgII": "tab:blue",
    "Hb": "tab:cyan",
    "HeII": "tab:pink",
    "OIII": "tab:green",
    "Ha": "tab:red",
    "NII": "tab:purple",
    "SII": "tab:brown",
}
_IRON_STYLE = ("#7570b3", "--")
_CONTINUUM_STYLES = {
    "power_law": ("#d95f02", ":"),
    "uv_iron": _IRON_STYLE,
    "optical_iron": _IRON_STYLE,
    "balmer_continuum": ("#66a61e", (0, (5, 2))),
    "balmer_series": ("#a6761d", (0, (1, 2))),
}
_COMBINED_BROAD_STYLE = {"color": "#1f77b4", "linestyle": "-", "linewidth": 1.6}
_BROAD_COMPONENT_STYLE = {"color": "#7f7f7f", "linestyle": "--", "linewidth": 1.15}
_NARROW_STYLE = {"color": "#2ca02c", "linestyle": "-", "linewidth": 1.15}
_WING_STYLE = {"color": "#9467bd", "linestyle": "-.", "linewidth": 1.15}
_MAJOR_EMISSION_LINES = (
    (1908.73, r"C III]"),
    (2798.75, r"Mg II"),
    (3728.47, r"[O II]"),
    (4341.68, r"H$\gamma$"),
    (4862.68, r"H$\beta$"),
    (5008.24, r"[O III]"),
    (6564.61, r"H$\alpha$"),
)
_ZOOM_EMISSION_LINES = {
    "mgii": ((2798.75, r"Mg II"),),
    "hbeta": (
        (4862.68, r"H$\beta$"),
        (4960.30, r"[O III] 4959"),
        (5008.24, r"[O III] 5007"),
    ),
    "halpha": (
        (6549.85, r"[N II] 6548"),
        (6564.61, r"H$\alpha$"),
        (6585.28, r"[N II] 6584"),
        (6718.29, r"[S II] 6716"),
        (6732.67, r"[S II] 6731"),
    ),
}
_ZOOM_PRIORITY = ("hbeta", "mgii", "halpha")
_LINE_MARKER_STYLE = {
    "color": "0.35",
    "linestyle": ":",
    "linewidth": 0.65,
    "alpha": 0.55,
}


def _species_from_component(name: str) -> str:
    for species in ("MgII", "HeII", "OIII", "NII", "SII", "Hb", "Ha"):
        if name.startswith(species):
            return species
    return "Hb"


def _line_groups(name: str, fit) -> Tuple[Tuple[str, np.ndarray, str, str], ...]:
    broad_names = [key for key in fit.component_models if "broad" in key and "wing" not in key]
    groups = []
    if broad_names:
        broad_sum = sum(
            (fit.component_models[key] for key in broad_names),
            np.zeros_like(fit.model),
        )
        label = {"mgii": "Mg II broad", "hbeta": "Hβ broad", "halpha": "Hα broad"}[name]
        species = {"mgii": "MgII", "hbeta": "Hb", "halpha": "Ha"}[name]
        groups.append((label, broad_sum, species, "broad"))
    for component_name, component in fit.component_models.items():
        if component_name in broad_names:
            continue
        kind = "wing" if "wing" in component_name else "narrow"
        groups.append(
            (
                component_name.replace("_", " "),
                component,
                _species_from_component(component_name),
                kind,
            )
        )
    return tuple(groups)


def _broad_component_names(fit) -> Tuple[str, ...]:
    return tuple(
        name
        for name in fit.component_models
        if "broad" in name and "wing" not in name
    )


def _combined_broad_profile(fit) -> np.ndarray:
    return sum(
        (fit.component_models[name] for name in _broad_component_names(fit)),
        np.zeros_like(fit.model),
    )


def _select_zoom_complexes(
    line_complexes: Mapping[str, object],
    max_zoom_panels: int,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    available = [
        name
        for name, fit in line_complexes.items()
        if name in _COMPLEX_WINDOWS and bool(getattr(fit, "success", False))
    ]
    priority_order = [
        name for name in _ZOOM_PRIORITY if name in available
    ] + sorted(
        (name for name in available if name not in _ZOOM_PRIORITY),
        key=lambda name: _COMPLEX_WINDOWS[name][0],
    )
    selected = set(priority_order[:max_zoom_panels])
    displayed = tuple(
        sorted(selected, key=lambda name: _COMPLEX_WINDOWS[name][0])
    )
    omitted = tuple(
        sorted(
            (name for name in available if name not in selected),
            key=lambda name: _COMPLEX_WINDOWS[name][0],
        )
    )
    return displayed, omitted


def _masked_running_median(
    values: np.ndarray,
    valid: np.ndarray,
    window_pixels: int,
) -> np.ndarray:
    series = pd.Series(
        np.where(np.asarray(valid, dtype=bool), np.asarray(values, dtype=float), np.nan)
    )
    smoothed = series.rolling(
        window=window_pixels,
        center=True,
        min_periods=1,
    ).median().to_numpy()
    smoothed[~np.asarray(valid, dtype=bool)] = np.nan
    return smoothed


def _format_reduced_chi2(value: float) -> str:
    return f"{value:.2f}" if np.isfinite(value) else "n/a"


def _qa_overview_title(result: NeoFitWorkflowResult) -> str:
    parts = ["Global continuum and emission-line QA"]
    object_id = result.metadata.get("object_id")
    if object_id not in (None, ""):
        parts.append(f"ID {object_id}")
    redshift = result.metadata.get("redshift")
    if redshift is not None and np.isfinite(redshift):
        parts.append(f"z={float(redshift):.4f}")
    return " — ".join(parts)


def _configure_qa_axis(axis) -> None:
    axis.minorticks_on()
    axis.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True,
        labelsize=8,
    )
    axis.tick_params(which="major", length=4.0)
    axis.tick_params(which="minor", length=2.2)


def _annotate_emission_lines(axis, lines, *, y_fraction: float) -> Tuple[str, ...]:
    labels = []
    x_min, x_max = axis.get_xlim()
    for line_wave, line_label in lines:
        if not x_min <= line_wave <= x_max:
            continue
        axis.axvline(line_wave, zorder=0.5, **_LINE_MARKER_STYLE)
        axis.text(
            line_wave,
            y_fraction,
            line_label,
            transform=axis.get_xaxis_transform(),
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="0.25",
            alpha=0.72,
        )
        labels.append(line_label)
    return tuple(labels)


def _flux_density_axis_label(flux_density_unit: str) -> str:
    normalized = str(flux_density_unit).lower().replace("angstrom", "aa")
    if (
        "1e-17" in normalized
        and "erg" in normalized
        and "cm" in normalized
        and "aa" in normalized
    ):
        return (
            r"$f_\lambda\ "
            r"[10^{-17}\,\mathrm{erg}\,\mathrm{s}^{-1}\,"
            r"\mathrm{cm}^{-2}\,\mathrm{\AA}^{-1}]$"
        )
    return rf"$f_\lambda$ [{flux_density_unit}]"


def _rounded_model_upper_limit(model_values: np.ndarray) -> Optional[float]:
    """Return a compact rounded upper limit with headroom above the fitted model."""

    values = np.asarray(model_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    peak = float(np.max(values))
    if peak <= 0:
        return None
    target = 1.2 * peak
    magnitude = 10.0 ** np.floor(np.log10(target))
    normalized = target / magnitude
    for nice_value in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0):
        if normalized <= nice_value:
            return float(nice_value * magnitude)
    return float(10.0 * magnitude)


def _full_line_model(result: NeoFitWorkflowResult) -> np.ndarray:
    return sum(
        (
            complex_result.model
            for complex_result in result.line_complexes.values()
            if complex_result.success
        ),
        np.zeros_like(result.continuum.model),
    )


def _plot_qa(
    result: NeoFitWorkflowResult,
    path: Path,
    plot_config: Optional[GlobalQAPlotConfig] = None,
) -> str:
    import matplotlib.pyplot as plt

    config = plot_config or GlobalQAPlotConfig()
    available, omitted = _select_zoom_complexes(
        result.line_complexes,
        config.max_zoom_panels,
    )
    result.metadata["qa_panel_count"] = 1 + len(available)
    result.metadata["qa_percentiles"] = [1.0, 99.8]
    result.metadata["qa_layout"] = "overview_top_complexes_bottom"
    result.metadata["qa_figure_size_inches"] = [
        float(config.figure_width),
        float(config.figure_height),
    ]
    result.metadata["qa_max_zoom_panels"] = int(config.max_zoom_panels)
    result.metadata["qa_displayed_complexes"] = list(available)
    result.metadata["qa_omitted_complexes"] = list(omitted)
    result.metadata["qa_smoothed_data"] = bool(config.show_smoothed_data)
    result.metadata["qa_smoothing_window_pixels"] = int(
        config.smoothing_window_pixels
    )
    result.metadata["qa_tick_direction"] = "in"
    result.metadata["qa_minor_ticks"] = True
    result.metadata["qa_zoom_model_upper_limits"] = {}
    result.metadata["qa_zoom_ymin"] = {}
    result.metadata["qa_zoom_titles"] = {}
    result.metadata["qa_zoom_line_labels"] = {}
    ncols = max(len(available), 1)
    fig = plt.figure(
        figsize=(config.figure_width, config.figure_height),
        constrained_layout=True,
    )
    grid = fig.add_gridspec(2, ncols, height_ratios=(1.0, 0.82))
    overview_axis = fig.add_subplot(grid[0, :])
    zoom_axes = [fig.add_subplot(grid[1, index]) for index in range(ncols)]
    spectrum = result.spectrum
    wave = spectrum.wave_rest
    valid = spectrum.valid_mask
    line_model = _full_line_model(result)
    full_model = result.continuum.model + line_model
    smoothed_data = (
        _masked_running_median(
            spectrum.flux,
            valid,
            config.smoothing_window_pixels,
        )
        if config.show_smoothed_data
        else None
    )

    def plot_common(ax, panel_mask, title):
        ax.plot(
            wave[panel_mask],
            spectrum.flux[panel_mask],
            color="0.45",
            lw=0.65,
            label="host-subtracted data",
        )
        if smoothed_data is not None:
            ax.plot(
                wave[panel_mask],
                smoothed_data[panel_mask],
                color="0.25",
                lw=0.8,
                alpha=0.8,
                label="smoothed data",
            )
        ax.plot(
            wave[panel_mask],
            full_model[panel_mask],
            color="black",
            lw=1.8,
            label="full model",
        )
        ax.plot(
            wave[panel_mask],
            result.continuum.model[panel_mask],
            color="tab:orange",
            lw=1.25,
            ls="--",
            label="total continuum",
        )
        iron_label_used = False
        for component_name, component in result.continuum.component_models.items():
            if not np.any(np.abs(component[panel_mask]) > 0):
                continue
            color, linestyle = _CONTINUUM_STYLES.get(component_name, ("0.5", ":"))
            if component_name in ("uv_iron", "optical_iron"):
                label = "iron" if not iron_label_used else "_nolegend_"
                iron_label_used = True
            else:
                label = component_name.replace("_", " ")
            ax.plot(
                wave[panel_mask],
                component[panel_mask],
                color=color,
                ls=linestyle,
                lw=0.9,
                label=label,
            )
        limits = _percentile_limits(
            [spectrum.flux[panel_mask], full_model[panel_mask]],
            percentiles=(1.0, 99.8),
        )
        if limits:
            ax.set_ylim(*limits)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(_flux_density_axis_label(spectrum.flux_density_unit), fontsize=9)
        _configure_qa_axis(ax)

    overview_title = _qa_overview_title(result)
    result.metadata["qa_overview_title"] = overview_title
    plot_common(overview_axis, valid, overview_title)
    broad_label_used = False
    narrow_label_used = False
    wing_label_used = False
    for complex_name in available:
        fit = result.line_complexes[complex_name]
        combined = _combined_broad_profile(fit)
        overview_axis.plot(
            wave[valid],
            combined[valid],
            label="full broad line" if not broad_label_used else "_nolegend_",
            **_COMBINED_BROAD_STYLE,
        )
        broad_label_used = True
        for label, component, species, kind in _line_groups(complex_name, fit):
            if kind == "broad":
                continue
            style = _WING_STYLE if kind == "wing" else _NARROW_STYLE
            if kind == "wing":
                legend_label = "outflow wing" if not wing_label_used else "_nolegend_"
                wing_label_used = True
            else:
                legend_label = "narrow line" if not narrow_label_used else "_nolegend_"
                narrow_label_used = True
            overview_axis.plot(
                wave[valid],
                component[valid],
                label=legend_label,
                **style,
            )
        lo, hi = _COMPLEX_WINDOWS[complex_name]
        overview_axis.axvspan(lo, hi, color="0.7", alpha=0.10)
    valid_wave = wave[valid]
    if valid_wave.size:
        overview_axis.set_xlim(float(valid_wave.min()), float(valid_wave.max()))
        result.metadata["qa_overview_xlim"] = [
            float(valid_wave.min()),
            float(valid_wave.max()),
        ]
        result.metadata["qa_major_emission_line_labels"] = list(
            _annotate_emission_lines(
                overview_axis,
                _MAJOR_EMISSION_LINES,
                y_fraction=0.82,
            )
        )
    overview_upper = _rounded_model_upper_limit(full_model[valid])
    if overview_upper is not None:
        overview_axis.set_ylim(
            0.0,
            max(overview_upper, 0.0),
        )
        result.metadata["qa_overview_ymin"] = 0.0
        result.metadata["qa_overview_model_upper_limit"] = overview_upper
    host_state = (
        "pPXF-subtracted"
        if result.host_decomp_enabled
        or result.metadata.get("host_decomp_enabled", False)
        else "not subtracted"
    )
    overview_annotation = (
        rf"$\chi^2_\nu(\mathrm{{cont.}})="
        f"{_format_reduced_chi2(result.continuum.reduced_chi2)}$\n"
        f"Host: {host_state}"
    )
    overview_axis.text(
        0.01,
        0.97,
        overview_annotation,
        transform=overview_axis.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.72,
        },
    )
    result.metadata["qa_overview_annotation"] = {
        "continuum_reduced_chi2": float(result.continuum.reduced_chi2),
        "host_state": host_state,
    }
    overview_axis.set_xlabel(r"Rest wavelength [$\mathrm{\AA}$]", fontsize=9)
    overview_axis.legend(
        fontsize=6.5,
        ncol=4,
        loc="best",
        framealpha=0.72,
        borderpad=0.35,
        handlelength=2.4,
    )

    for axis, complex_name in zip(zoom_axes, available):
        lo, hi = _COMPLEX_WINDOWS[complex_name]
        panel_mask = valid & (wave >= lo) & (wave <= hi)
        title = {
            "mgii": "Mg II",
            "hbeta": "Hβ / [O III]",
            "halpha": "Hα / [N II] / [S II]",
        }[complex_name]
        fit = result.line_complexes[complex_name]
        title = (
            f"{title}  |  "
            rf"$\chi^2_\nu={_format_reduced_chi2(fit.reduced_chi2)}$"
        )
        result.metadata.setdefault("qa_zoom_titles", {})[complex_name] = title
        plot_common(axis, panel_mask, title)
        combined = _combined_broad_profile(fit)
        axis.plot(
            wave[panel_mask],
            combined[panel_mask],
            label="_nolegend_",
            **_COMBINED_BROAD_STYLE,
        )
        broad_component_label_used = False
        broad_names = set(_broad_component_names(fit))
        for component_name, component in fit.component_models.items():
            if component_name in broad_names:
                axis.plot(
                    wave[panel_mask],
                    component[panel_mask],
                    label=(
                        "broad components"
                        if not broad_component_label_used
                        else "_nolegend_"
                    ),
                    **_BROAD_COMPONENT_STYLE,
                )
                broad_component_label_used = True
                continue
            kind = "wing" if "wing" in component_name else "narrow"
            style = _WING_STYLE if kind == "wing" else _NARROW_STYLE
            axis.plot(
                wave[panel_mask],
                component[panel_mask],
                label="_nolegend_",
                **style,
            )
        limits = axis.get_ylim()
        zoom_upper = _rounded_model_upper_limit(full_model[panel_mask])
        axis.set_ylim(0.0, max(zoom_upper if zoom_upper is not None else limits[1], 0.0))
        result.metadata.setdefault("qa_zoom_model_upper_limits", {})[
            complex_name
        ] = zoom_upper
        result.metadata.setdefault("qa_zoom_ymin", {})[complex_name] = 0.0
        axis.set_xlim(lo, hi)
        axis.axhline(0.0, color="0.55", lw=0.65, zorder=0)
        result.metadata.setdefault("qa_zoom_line_labels", {})[complex_name] = list(
            _annotate_emission_lines(
                axis,
                _ZOOM_EMISSION_LINES[complex_name],
                y_fraction=0.82,
            )
        )
        axis.set_xlabel(r"Rest wavelength [$\mathrm{\AA}$]", fontsize=9)
        handles, labels = axis.get_legend_handles_labels()
        unique = [
            (handle, label)
            for handle, label in zip(handles, labels)
            if label == "broad components"
        ]
        if unique:
            axis.legend(
                [unique[0][0]],
                [unique[0][1]],
                fontsize=6.5,
                loc="best",
                framealpha=0.72,
                borderpad=0.35,
            )
    if not available:
        zoom_axes[0].set_visible(False)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def _plot_hbeta(result: NeoFitWorkflowResult, path: Path) -> str:
    import matplotlib.pyplot as plt

    fit = result.hbeta
    view = (fit.wave_rest >= 4600.0) & (fit.wave_rest <= 5120.0)
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    ax.plot(
        fit.wave_rest[view],
        fit.flux_continuum_subtracted[view],
        color="0.3",
        lw=0.8,
        label="continuum-subtracted data",
    )
    ax.plot(fit.wave_rest[view], fit.model[view], color="tab:blue", lw=1.4, label=fit.selected_model)
    for name, component in fit.component_models.items():
        ax.plot(fit.wave_rest[view], component[view], lw=0.8, ls="--", label=name)
    limits = _percentile_limits([fit.flux_continuum_subtracted[view], fit.model[view]])
    if limits:
        ax.set_ylim(*limits)
    ax.set_xlim(4600.0, 5120.0)
    ax.set_xlabel("Rest wavelength [Angstrom]", fontsize=10)
    ax.set_ylabel(f"Flux density [{result.spectrum.flux_density_unit}]", fontsize=9)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=7, ncol=3)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def _measurement_row(fit) -> Dict[str, object]:
    row = {}
    row.update(fit.param_values)
    row.update({f"{key}_err": value for key, value in fit.param_errors.items()})
    row.update(fit.metrics)
    row.update({f"{key}_err": value for key, value in fit.metric_errors.items()})
    row.update(
        {
            "success": fit.success,
            "selected_model": fit.selected_model,
            "reduced_chi2": fit.reduced_chi2,
            "bic": fit.bic,
        }
    )
    return row


def _write_complex_products(out: Path, name: str, fit, files: Dict[str, str]) -> None:
    filenames = {
        "mgii": ("mgii_measurements.csv", "mgii_model.csv"),
        "hbeta": ("hbeta_oiii_measurements.csv", "hbeta_oiii_model.csv"),
        "halpha": ("halpha_nii_sii_measurements.csv", "halpha_nii_sii_model.csv"),
    }
    measurement_name, model_name = filenames[name]
    measurement_path = out / measurement_name
    pd.DataFrame([_measurement_row(fit)]).to_csv(measurement_path, index=False)
    files[f"{name}_measurements_csv"] = str(measurement_path)

    lo, hi = _COMPLEX_WINDOWS[name]
    view = (fit.wave_rest >= lo) & (fit.wave_rest <= hi)
    model_grid = {
        "wave_rest": fit.wave_rest[view],
        "continuum_subtracted": fit.flux_continuum_subtracted[view],
        "err": fit.err[view],
        "model": fit.model[view],
        "fit_used": fit.fit_mask[view].astype(int),
    }
    for component_name, component in fit.component_models.items():
        model_grid[component_name] = component[view]
    model_path = out / model_name
    pd.DataFrame(model_grid).to_csv(model_path, index=False)
    files[f"{name}_model_csv"] = str(model_path)


def write_global_line_products(
    result: NeoFitWorkflowResult,
    output_dir: str,
    qa_plot_config: Optional[GlobalQAPlotConfig] = None,
) -> Dict[str, str]:
    """Write standard global-continuum and multi-complex products."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files: Dict[str, str] = {}

    summary_path = out / "neofit_global_lines_summary.json"
    compatibility_summary_path = out / "neofit_global_hbeta_summary.json"
    summary_payload = json.dumps(
        result.summary(), indent=2, sort_keys=True, default=_json_default
    )
    summary_path.write_text(
        summary_payload,
        encoding="utf-8",
    )
    compatibility_summary_path.write_text(
        summary_payload,
        encoding="utf-8",
    )
    files["summary_json"] = str(summary_path)
    files["compatibility_summary_json"] = str(compatibility_summary_path)

    continuum_row = {}
    continuum_row.update(result.continuum.param_values)
    continuum_row.update({f"{key}_err": value for key, value in result.continuum.param_errors.items()})
    continuum_row.update(result.metadata.get("continuum_samples", {}))
    for key in (
        "balmer_series_implied_hbeta_flux_input",
        "balmer_series_implied_hbeta_flux_cgs",
        "balmer_series_fwhm_kms",
    ):
        if key in result.continuum.metadata:
            continuum_row[key] = result.continuum.metadata[key]
    continuum_row.update(
        {
            "success": result.continuum.success,
            "reduced_chi2": result.continuum.reduced_chi2,
            "balmer_template": result.continuum.metadata.get("balmer_template"),
        }
    )
    continuum_path = out / "global_continuum_measurements.csv"
    pd.DataFrame([continuum_row]).to_csv(continuum_path, index=False)
    files["continuum_measurements_csv"] = str(continuum_path)

    for complex_name, fit in result.line_complexes.items():
        _write_complex_products(out, complex_name, fit, files)

    spectrum = result.spectrum
    grid = {
        "wave_obs": spectrum.wave_obs,
        "wave_rest": spectrum.wave_rest,
        "flux_fit_input": spectrum.flux,
        "err": spectrum.err,
        "global_continuum": result.continuum.model,
        "continuum_subtracted": spectrum.flux - result.continuum.model,
        "fit_used_continuum": result.continuum.clip_mask.astype(int),
        "full_model": result.continuum.model + _full_line_model(result),
    }
    if result.total_spectrum is not None:
        grid["flux_total_before_host"] = result.total_spectrum.flux
    if result.host_model_on_quasar_grid is not None:
        grid["ppxf_host_model"] = result.host_model_on_quasar_grid
    for name, component in result.continuum.component_models.items():
        grid[f"continuum_{name}"] = component
    for complex_name, fit in result.line_complexes.items():
        grid[f"{complex_name}_model"] = fit.model
        grid[f"fit_used_{complex_name}"] = fit.fit_mask.astype(int)
        for component_name, component in fit.component_models.items():
            grid[f"line_{complex_name}_{component_name}"] = component
    grid_path = out / "neofit_global_lines_full_grid.csv"
    pd.DataFrame(grid).to_csv(grid_path, index=False)
    files["full_grid_csv"] = str(grid_path)
    compatibility_grid_path = out / "neofit_global_hbeta_full_grid.csv"
    pd.DataFrame(grid).to_csv(compatibility_grid_path, index=False)
    files["compatibility_full_grid_csv"] = str(compatibility_grid_path)

    files["global_plot"] = _plot_qa(
        result,
        out / "diagnostic_global_continuum.png",
        qa_plot_config,
    )
    files["qa_plot"] = files["global_plot"]
    files["balmer_edge_plot"] = _plot_global(
        result, out / "diagnostic_balmer_edge.png", window=(3300.0, 4300.0)
    )
    files["hbeta_plot"] = _plot_hbeta(result, out / "diagnostic_hbeta_oiii.png")
    result.output_files.update(files)
    summary_payload = json.dumps(
        result.summary(), indent=2, sort_keys=True, default=_json_default
    )
    summary_path.write_text(summary_payload, encoding="utf-8")
    compatibility_summary_path.write_text(summary_payload, encoding="utf-8")
    return files


def write_global_hbeta_products(
    result: NeoFitWorkflowResult,
    output_dir: str,
    qa_plot_config: Optional[GlobalQAPlotConfig] = None,
) -> Dict[str, str]:
    """Compatibility wrapper retaining H-beta summary/full-grid paths."""

    files = write_global_line_products(result, output_dir, qa_plot_config)
    files["generic_summary_json"] = files["summary_json"]
    files["generic_full_grid_csv"] = files["full_grid_csv"]
    files["summary_json"] = files["compatibility_summary_json"]
    files["full_grid_csv"] = files["compatibility_full_grid_csv"]
    result.output_files.update(files)
    summary_payload = json.dumps(
        result.summary(), indent=2, sort_keys=True, default=_json_default
    )
    Path(files["summary_json"]).write_text(summary_payload, encoding="utf-8")
    Path(files["generic_summary_json"]).write_text(summary_payload, encoding="utf-8")
    return files
