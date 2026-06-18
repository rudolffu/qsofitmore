"""Optional pPXF host subtraction before neofit fitting."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .api import fit_local
from .config import (
    GlobalContinuumConfig,
    HalphaComplexConfig,
    HbetaComplexConfig,
    LocalFitConfig,
    MgIIComplexConfig,
    UncertaintyConfig,
)
from .global_fit import fit_global_hbeta, fit_global_lines
from .global_result import NeoFitWorkflowResult
from .result import LocalFitResult
from .spectrum import Spectrum


@dataclass
class NeoFitHostWorkflowResult:
    """Result of optional host subtraction followed by a neofit fit."""

    total_spectrum: Spectrum
    fit_spectrum: Spectrum
    local_result: LocalFitResult
    host_decomp_enabled: bool
    host_fit: Optional[Any] = None
    host_sed: Optional[Any] = None
    host_model_on_quasar_grid: Optional[np.ndarray] = None
    host_subtracted_flux: Optional[np.ndarray] = None
    host_warnings: Optional[list] = None
    metadata: Optional[Dict[str, Any]] = None


def _good_mask_from_spectrum_data(spectrum_data: Any, extra_mask: Optional[np.ndarray] = None) -> np.ndarray:
    wave = np.asarray(spectrum_data.wave_obs, dtype=float)
    flux = np.asarray(spectrum_data.flux, dtype=float)
    err = np.asarray(spectrum_data.uncertainty(), dtype=float)
    good = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(err) & (wave > 0) & (err > 0)
    if spectrum_data.ivar is not None:
        ivar = np.asarray(spectrum_data.ivar, dtype=float)
        good &= np.isfinite(ivar) & (ivar > 0)
    if spectrum_data.mask is not None:
        good &= np.asarray(spectrum_data.mask) == 0
    if extra_mask is not None:
        good &= np.asarray(extra_mask, dtype=bool)
    return good


def _spectrum_from_arrays(
    wave_obs: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    redshift: float,
    mask: Optional[np.ndarray],
    source: str,
) -> Spectrum:
    return Spectrum.from_arrays(
        wave_obs,
        flux,
        err=err,
        z=float(redshift),
        wave_frame="observed",
        mask=mask,
        survey="desi",
        source=source,
    )


def _spectrum_from_spectrum_data(spectrum_data: Any, source: str) -> Spectrum:
    good = _good_mask_from_spectrum_data(spectrum_data)
    return _spectrum_from_arrays(
        np.asarray(spectrum_data.wave_obs, dtype=float),
        np.asarray(spectrum_data.flux, dtype=float),
        np.asarray(spectrum_data.uncertainty(), dtype=float),
        float(spectrum_data.redshift),
        good,
        source=source,
    )


def _host_subtracted_spectrum(
    spectrum_data: Any,
    *,
    redshift: Optional[float],
    template_root: str,
    template_file: str,
    fit_range: Tuple[float, float],
    host_config: Optional[Any],
    source: str,
) -> Tuple[Spectrum, Spectrum, Any, Any, np.ndarray, np.ndarray, list]:
    from qsofitmore.host_decomp.config import default_config
    from qsofitmore.host_decomp.ppxf_host import (
        prepare_desi_for_host_decomp,
        predict_host_sed,
        predict_host_sed_on_grid,
        run_ppxf_host_fit,
    )
    from qsofitmore.host_decomp.templates import load_ppxf_npz_templates

    cfg = host_config or default_config()
    templates = load_ppxf_npz_templates(template_root=template_root, template_file=template_file)
    prep = prepare_desi_for_host_decomp(
        spectrum_data,
        redshift=redshift,
        fit_range=fit_range,
        line_mask_widths=cfg.line_mask_widths,
        broad_line_mask_widths=cfg.broad_line_mask_widths,
    )
    host_fit = run_ppxf_host_fit(
        prep,
        templates,
        agn_powerlaw_slopes=cfg.agn_powerlaw_slopes,
        polynomial_degree=cfg.polynomial_degree,
        multiplicative_polynomial_degree=cfg.multiplicative_polynomial_degree,
    )
    host_sed = predict_host_sed(host_fit)
    host_on_grid, grid_warnings = predict_host_sed_on_grid(host_sed, prep.wave_rest)
    host_warnings = list(host_fit.warnings) + list(host_sed.warnings) + list(grid_warnings)
    finite_host = np.isfinite(host_on_grid)
    host_subtracted_flux = np.asarray(prep.flux, dtype=float) - np.where(finite_host, host_on_grid, 0.0)

    total_spectrum = _spectrum_from_arrays(
        prep.wave_obs,
        prep.flux,
        prep.error,
        prep.redshift,
        np.isfinite(prep.wave_obs) & np.isfinite(prep.flux) & np.isfinite(prep.error) & (prep.error > 0),
        source=source,
    )
    fit_spectrum = _spectrum_from_arrays(
        prep.wave_obs,
        host_subtracted_flux,
        prep.error,
        prep.redshift,
        np.isfinite(prep.wave_obs)
        & np.isfinite(host_subtracted_flux)
        & np.isfinite(prep.error)
        & (prep.error > 0)
        & finite_host,
        source=f"{source}; host_subtracted=ppxf_sed_grid",
    )
    return total_spectrum, fit_spectrum, host_fit, host_sed, host_on_grid, host_subtracted_flux, host_warnings


def fit_with_optional_host_decomp(
    input_path: str,
    local_config: Optional[LocalFitConfig] = None,
    *,
    row_index: Optional[int] = None,
    redshift: Optional[float] = None,
    object_id: Optional[str] = None,
    run_host_decomp: bool = False,
    fit_kind: str = "local",
    template_root: str = "~/tools/ppxf_data",
    template_file: str = "spectra_emiles_9.0.npz",
    host_fit_range: Tuple[float, float] = (3600.0, 7000.0),
    host_config: Optional[Any] = None,
    global_config: Optional[GlobalContinuumConfig] = None,
    hbeta_config: Optional[HbetaComplexConfig] = None,
    mgii_config: Optional[MgIIComplexConfig] = None,
    halpha_config: Optional[HalphaComplexConfig] = None,
    uncertainty_config: Optional[UncertaintyConfig] = None,
):
    """Read a spectrum, optionally subtract a pPXF host, then run neofit.

    ``fit_kind`` may be ``"local"`` or ``"global"``.
    """

    if fit_kind == "global":
        return fit_global_lines_workflow(
            input_path,
            row_index=row_index,
            redshift=redshift,
            object_id=object_id,
            run_host_decomp=run_host_decomp,
            template_root=template_root,
            template_file=template_file,
            host_fit_range=host_fit_range,
            host_config=host_config,
            global_config=global_config,
            hbeta_config=hbeta_config,
            mgii_config=mgii_config,
            halpha_config=halpha_config,
            uncertainty_config=uncertainty_config,
        )
    if fit_kind != "local":
        raise ValueError("fit_kind must be 'local' or 'global'.")
    if local_config is None:
        raise ValueError("local_config is required when fit_kind='local'.")

    from qsofitmore.host_decomp.io import read_sparcli_spectrum

    spectrum_data = read_sparcli_spectrum(input_path, row_index=row_index, redshift=redshift, object_id=object_id)
    source = f"{input_path}:row_index={row_index}"
    if run_host_decomp:
        total_spectrum, fit_spectrum, host_fit, host_sed, host_on_grid, host_subtracted_flux, host_warnings = (
            _host_subtracted_spectrum(
                spectrum_data,
                redshift=redshift,
                template_root=template_root,
                template_file=template_file,
                fit_range=host_fit_range,
                host_config=host_config,
                source=source,
            )
        )
    else:
        total_spectrum = _spectrum_from_spectrum_data(spectrum_data, source=source)
        fit_spectrum = total_spectrum
        host_fit = None
        host_sed = None
        host_on_grid = None
        host_subtracted_flux = None
        host_warnings = []

    local_result = fit_local(fit_spectrum, local_config)
    metadata = {
        "input_path": input_path,
        "row_index": row_index,
        "object_id": object_id or spectrum_data.object_id or spectrum_data.targetid,
        "targetid": spectrum_data.targetid,
        "ra": spectrum_data.ra,
        "dec": spectrum_data.dec,
        "redshift": fit_spectrum.z,
        "fit_kind": fit_kind,
        "host_decomp_enabled": bool(run_host_decomp),
        "host_model_source": "template_weighted_sed_on_quasar_grid" if run_host_decomp else None,
    }
    return NeoFitHostWorkflowResult(
        total_spectrum=total_spectrum,
        fit_spectrum=fit_spectrum,
        local_result=local_result,
        host_decomp_enabled=bool(run_host_decomp),
        host_fit=host_fit,
        host_sed=host_sed,
        host_model_on_quasar_grid=host_on_grid,
        host_subtracted_flux=host_subtracted_flux,
        host_warnings=host_warnings,
        metadata=metadata,
    )


def _summarize_mc_results(samples: Dict[str, list], n_requested: int, n_success: int) -> Dict[str, Any]:
    percentiles = {}
    for name, values in samples.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            p16, p50, p84 = np.percentile(finite, [16.0, 50.0, 84.0])
            percentiles[name] = {"p16": float(p16), "p50": float(p50), "p84": float(p84)}
    return {"n_requested": int(n_requested), "n_success": int(n_success), "percentiles": percentiles}


def _run_host_refit_mc(
    spectrum_data: Any,
    *,
    n_trials: int,
    seed: Optional[int],
    redshift: Optional[float],
    template_root: str,
    template_file: str,
    host_fit_range: Tuple[float, float],
    host_config: Optional[Any],
    source: str,
    global_config: Optional[GlobalContinuumConfig],
    hbeta_config: Optional[HbetaComplexConfig],
    mgii_config: Optional[MgIIComplexConfig],
    halpha_config: Optional[HalphaComplexConfig],
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    samples: Dict[str, list] = {}
    successes = 0
    error = np.asarray(spectrum_data.uncertainty(), dtype=float)
    for _ in range(int(n_trials)):
        noisy_data = replace(
            spectrum_data,
            flux=np.asarray(spectrum_data.flux, dtype=float) + rng.normal(0.0, error),
        )
        try:
            _, fit_spectrum, _, _, host_on_grid, _, _ = _host_subtracted_spectrum(
                noisy_data,
                redshift=redshift,
                template_root=template_root,
                template_file=template_file,
                fit_range=host_fit_range,
                host_config=host_config,
                source=source,
            )
            trial = fit_global_lines(
                fit_spectrum,
                global_config,
                hbeta_config,
                mgii_config,
                halpha_config,
                UncertaintyConfig(monte_carlo_trials=0),
                host_model_on_grid=host_on_grid,
            )
            if not trial.success:
                continue
            successes += 1
            values = dict(trial.continuum.param_values)
            for complex_result in trial.line_complexes.values():
                values.update(complex_result.metrics)
            for name, value in values.items():
                if np.isfinite(value):
                    samples.setdefault(name, []).append(float(value))
        except Exception:
            continue
    return _summarize_mc_results(samples, n_trials, successes)


def fit_global_lines_workflow(
    input_path: str,
    *,
    row_index: Optional[int] = None,
    redshift: Optional[float] = None,
    object_id: Optional[str] = None,
    run_host_decomp: bool = False,
    template_root: str = "~/tools/ppxf_data",
    template_file: str = "spectra_emiles_9.0.npz",
    host_fit_range: Tuple[float, float] = (3600.0, 7000.0),
    host_config: Optional[Any] = None,
    global_config: Optional[GlobalContinuumConfig] = None,
    hbeta_config: Optional[HbetaComplexConfig] = None,
    mgii_config: Optional[MgIIComplexConfig] = None,
    halpha_config: Optional[HalphaComplexConfig] = None,
    uncertainty_config: Optional[UncertaintyConfig] = None,
) -> NeoFitWorkflowResult:
    """Read one spectrum and run optional pPXF plus global multi-line neofit."""

    from qsofitmore.host_decomp.io import read_sparcli_spectrum

    uncertainty = uncertainty_config or UncertaintyConfig()
    spectrum_data = read_sparcli_spectrum(
        input_path, row_index=row_index, redshift=redshift, object_id=object_id
    )
    source = f"{input_path}:row_index={row_index}"
    if run_host_decomp:
        total_spectrum, fit_spectrum, host_fit, host_sed, host_on_grid, _, host_warnings = (
            _host_subtracted_spectrum(
                spectrum_data,
                redshift=redshift,
                template_root=template_root,
                template_file=template_file,
                fit_range=host_fit_range,
                host_config=host_config,
                source=source,
            )
        )
        primary_uncertainty = (
            replace(uncertainty, monte_carlo_trials=0)
            if uncertainty.monte_carlo_trials > 0 and uncertainty.refit_host_in_mc
            else uncertainty
        )
    else:
        total_spectrum = _spectrum_from_spectrum_data(spectrum_data, source=source)
        fit_spectrum = total_spectrum
        host_fit = None
        host_sed = None
        host_on_grid = None
        host_warnings = []
        primary_uncertainty = uncertainty

    workflow = fit_global_lines(
        fit_spectrum,
        global_config,
        hbeta_config,
        mgii_config,
        halpha_config,
        primary_uncertainty,
        host_model_on_grid=host_on_grid,
    )
    workflow.host_decomp_enabled = bool(run_host_decomp)
    workflow.total_spectrum = total_spectrum
    workflow.host_fit = host_fit
    workflow.host_sed = host_sed
    workflow.host_model_on_quasar_grid = host_on_grid
    workflow.host_warnings = [str(item) for item in host_warnings]
    workflow.metadata.update(
        {
            "input_path": input_path,
            "row_index": row_index,
            "object_id": object_id or spectrum_data.object_id or spectrum_data.targetid,
            "targetid": spectrum_data.targetid,
            "ra": spectrum_data.ra,
            "dec": spectrum_data.dec,
            "redshift": fit_spectrum.z,
            "fit_kind": "global",
            "host_decomp_enabled": bool(run_host_decomp),
            "host_model_source": "template_weighted_sed_on_quasar_grid" if run_host_decomp else None,
        }
    )
    if run_host_decomp and uncertainty.monte_carlo_trials > 0 and uncertainty.refit_host_in_mc:
        workflow.monte_carlo = _run_host_refit_mc(
            spectrum_data,
            n_trials=uncertainty.monte_carlo_trials,
            seed=uncertainty.random_seed,
            redshift=redshift,
            template_root=template_root,
            template_file=template_file,
            host_fit_range=host_fit_range,
            host_config=host_config,
            source=source,
            global_config=global_config,
            hbeta_config=hbeta_config,
            mgii_config=mgii_config,
            halpha_config=halpha_config,
        )
        workflow.metadata["uncertainty_mode"] = "covariance+monte_carlo_host_refit"
    return workflow


def fit_global_hbeta_workflow(
    input_path: str,
    *,
    row_index: Optional[int] = None,
    redshift: Optional[float] = None,
    object_id: Optional[str] = None,
    run_host_decomp: bool = False,
    template_root: str = "~/tools/ppxf_data",
    template_file: str = "spectra_emiles_9.0.npz",
    host_fit_range: Tuple[float, float] = (3600.0, 7000.0),
    host_config: Optional[Any] = None,
    global_config: Optional[GlobalContinuumConfig] = None,
    hbeta_config: Optional[HbetaComplexConfig] = None,
    uncertainty_config: Optional[UncertaintyConfig] = None,
) -> NeoFitWorkflowResult:
    """Compatibility wrapper for :func:`fit_global_lines_workflow`."""

    return fit_global_lines_workflow(
        input_path,
        row_index=row_index,
        redshift=redshift,
        object_id=object_id,
        run_host_decomp=run_host_decomp,
        template_root=template_root,
        template_file=template_file,
        host_fit_range=host_fit_range,
        host_config=host_config,
        global_config=global_config,
        hbeta_config=hbeta_config,
        uncertainty_config=uncertainty_config,
    )
