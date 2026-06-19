"""Public neofit fitting API."""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from .config import LineComplexConfig, LocalFitConfig
from .optimize import run_least_squares
from .parameters import pack_line_complex_parameters
from .residuals import iron_basis_vector, model_and_residual, model_components, model_vector
from .result import FitResult, LocalFitResult
from .spectrum import Spectrum
from .warnings import NeoFitWarning


def _prepare_iron_component(
    config: LineComplexConfig,
    wave_fit: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[Any], Optional[str], Optional[Dict[str, Any]], List[NeoFitWarning]]:
    """Load and prepare an optional iron template once for a local fit."""

    if config.iron is None or not config.iron.enabled:
        return None, None, None, None, []

    from .templates.iron import prepare_iron_template
    from .templates.registry import load_iron_template

    template = load_iron_template(
        config.iron.template,
        template_path=config.iron.template_path,
        normalization=config.iron.normalization,
    )
    prepared = prepare_iron_template(
        template,
        wave_fit,
        tuple(config.window),
        fwhm_kms=config.iron.fwhm_kms,
    )
    metadata = {
        "enabled": True,
        "template": template.name,
        "template_requested": config.iron.template,
        "template_source_path": template.source_path,
        "template_reference": template.reference,
        "template_coverage_min": float(template.coverage[0]) if template.coverage else float("nan"),
        "template_coverage_max": float(template.coverage[1]) if template.coverage else float("nan"),
        "normalization": template.normalization,
        "fwhm_initial_kms": float(config.iron.fwhm_kms),
        "fwhm_bounds": tuple(config.iron.fwhm_bounds),
        "fwhm_param": "iron.fwhm_kms" if prepared.has_overlap else None,
        "has_overlap": bool(prepared.has_overlap),
        "amp_param": "iron.amp" if prepared.has_overlap else None,
        "notes": list(template.notes),
    }
    basis = prepared.basis if prepared.has_overlap else None
    fit_template = template if prepared.has_overlap else None
    return basis, fit_template, template.name, metadata, list(prepared.warnings)


def _window_name(config: LineComplexConfig, index: int = 0) -> str:
    return config.name or f"line_complex_{index}"


def _flux_scale_warning(spectrum: Spectrum) -> List[NeoFitWarning]:
    if spectrum.flux_density_scale_to_cgs is not None:
        return []
    return [
        NeoFitWarning(
            code="flux_scale_unknown_cgs_not_reported",
            message="Flux-density scale to cgs is unknown; cgs line fluxes are not reported.",
        )
    ]


def _window_selector(wave_rest: np.ndarray, windows: List[Tuple[float, float]]) -> np.ndarray:
    selector = np.zeros_like(wave_rest, dtype=bool)
    for lo, hi in windows:
        selector |= (wave_rest >= float(lo)) & (wave_rest <= float(hi))
    return selector


def _fit_pixel_mask(spectrum: Spectrum, config: LineComplexConfig) -> np.ndarray:
    wave_rest = spectrum.wave_rest
    base_windows = list(config.fit_windows) if config.fit_windows is not None else [tuple(config.window)]
    fit_mask = spectrum.valid_mask & _window_selector(wave_rest, base_windows)
    if config.mask_windows:
        fit_mask &= ~_window_selector(wave_rest, list(config.mask_windows))
    lo, hi = map(float, config.window)
    fit_mask &= (wave_rest >= lo) & (wave_rest <= hi)
    return fit_mask


def fit_line_complex(
    spectrum: Spectrum,
    config: LineComplexConfig,
    jacobian: Optional[str] = None,
) -> FitResult:
    """Fit one local Gaussian line complex on the spectrum rest-frame grid."""

    wave_rest = spectrum.wave_rest
    lo, hi = map(float, config.window)
    fit_mask = _fit_pixel_mask(spectrum, config)
    window_mask = spectrum.valid_mask & (wave_rest >= lo) & (wave_rest <= hi)
    n_pixels = int(np.count_nonzero(fit_mask))
    if n_pixels == 0:
        raise ValueError("No valid pixels fall inside the requested line-complex window.")

    wave_fit = wave_rest[fit_mask]
    flux_fit = spectrum.flux[fit_mask]
    err_fit = spectrum.err[fit_mask]
    iron_basis, iron_template, iron_template_name, iron_metadata, iron_warnings = _prepare_iron_component(config, wave_fit)
    packed = pack_line_complex_parameters(
        config,
        wave_fit,
        flux_fit=flux_fit,
        iron_basis=iron_basis,
        iron_template=iron_template,
        iron_template_name=iron_template_name,
    )
    if n_pixels <= packed.initial.size:
        raise ValueError(
            f"Too few valid pixels ({n_pixels}) for {packed.initial.size} fitted parameters."
        )

    jac_mode = config.jacobian if jacobian is None else jacobian
    optimizer_result = run_least_squares(
        packed,
        wave_fit,
        flux_fit,
        err_fit,
        jacobian=jac_mode,
        max_nfev=config.max_nfev,
    )
    model, residual = model_and_residual(optimizer_result.x, packed, wave_fit, flux_fit, err_fit)
    wave_window = wave_rest[window_mask]
    flux_window = spectrum.flux[window_mask]
    err_window = spectrum.err[window_mask]
    model_window = model_vector(optimizer_result.x, packed, wave_window)
    residual_window = (flux_window - model_window) / err_window
    fit_used_window = fit_mask[window_mask]
    chi2 = float(np.sum(residual * residual))
    dof = int(max(wave_fit.size - optimizer_result.x.size, 0))
    reduced_chi2 = float(chi2 / dof) if dof > 0 else float("nan")
    param_values = packed.unpack(optimizer_result.x)
    warnings = _flux_scale_warning(spectrum)
    warnings.extend(iron_warnings)
    if not optimizer_result.success:
        warnings.append(
            NeoFitWarning(
                code="fit_failed",
                message=str(optimizer_result.message),
                context={"window": config.name, "status": int(optimizer_result.status)},
            )
        )
    metadata = spectrum.metadata.to_dict()
    metadata.update(
        {
            "window_name": config.name,
            "window": tuple(config.window),
            "fit_windows": list(config.fit_windows) if config.fit_windows is not None else [tuple(config.window)],
            "mask_windows": list(config.mask_windows),
            "plot_window": tuple(config.plot_window) if config.plot_window is not None else tuple(config.window),
            "line_center": float(config.center),
            "jacobian": jac_mode,
        }
    )
    if iron_metadata is not None:
        amp = param_values.get("iron.amp")
        fwhm = param_values.get("iron.fwhm_kms")
        iron_flux_input = float("nan")
        iron_flux_cgs = float("nan")
        final_iron_basis = iron_basis_vector(optimizer_result.x, packed, wave_fit)
        if amp is not None and final_iron_basis is not None:
            iron_model = float(amp) * final_iron_basis
            iron_flux_input = float(np.trapezoid(iron_model, wave_fit))
            scale = spectrum.metadata.flux_density_scale_to_cgs
            iron_flux_cgs = iron_flux_input * float(scale) if scale is not None else float("nan")
        iron_metadata.update(
            {
                "iron_amp": float(amp) if amp is not None else float("nan"),
                "fwhm_kms": float(fwhm) if fwhm is not None else float("nan"),
                "iron_flux_input": iron_flux_input,
                "iron_flux_cgs": iron_flux_cgs,
            }
        )
        metadata["iron"] = iron_metadata

    return FitResult(
        success=bool(optimizer_result.success),
        status=int(optimizer_result.status),
        message=str(optimizer_result.message),
        theta=np.asarray(optimizer_result.x, dtype=float),
        param_names=list(packed.names),
        param_values=param_values,
        chi2=chi2,
        dof=dof,
        reduced_chi2=reduced_chi2,
        model=model,
        residual=residual,
        wave_rest_fit=wave_fit,
        flux_fit=flux_fit,
        err_fit=err_fit,
        wave_rest_window=wave_window,
        flux_window=flux_window,
        err_window=err_window,
        model_window=model_window,
        residual_window=residual_window,
        fit_used_window=fit_used_window,
        component_models=model_components(optimizer_result.x, packed, wave_fit),
        component_models_window=model_components(optimizer_result.x, packed, wave_window),
        warnings=warnings,
        metadata=metadata,
        optimizer_result=optimizer_result,
    )


def _n_parameters(config: LineComplexConfig) -> int:
    n = 3 * len(config.components)
    if config.local_continuum == "constant":
        n += 1
    elif config.local_continuum == "linear":
        n += 2
    if config.iron is not None and config.iron.enabled:
        n += 2
    return n


def _validate_local_window(
    spectrum: Spectrum,
    config: LineComplexConfig,
    local_config: LocalFitConfig,
    window_name: str,
) -> Tuple[bool, List[NeoFitWarning]]:
    wave_rest = spectrum.wave_rest
    valid = spectrum.valid_mask
    warnings: List[NeoFitWarning] = []
    if not np.any(valid):
        warnings.append(
            NeoFitWarning(
                code="all_pixels_invalid",
                message="No finite pixels with positive errors are available.",
                severity="error",
                context={"window": window_name},
            )
        )
        return False, warnings

    valid_wave = wave_rest[valid]
    coverage_min = float(np.nanmin(valid_wave))
    coverage_max = float(np.nanmax(valid_wave))
    lo, hi = map(float, config.window)
    context = {
        "window": window_name,
        "requested_window": (lo, hi),
        "coverage": (coverage_min, coverage_max),
    }

    if hi < coverage_min or lo > coverage_max:
        warnings.append(
            NeoFitWarning(
                code="window_not_covered",
                message="Requested local window is outside the valid rest-frame spectrum coverage.",
                severity="error",
                context=context,
            )
        )
        return False, warnings
    if lo < coverage_min or hi > coverage_max:
        warnings.append(
            NeoFitWarning(
                code="window_not_covered",
                message="Requested local window is only partially covered by the valid rest-frame spectrum.",
                context=context,
            )
        )

    if config.center < lo or config.center > hi or config.center < coverage_min or config.center > coverage_max:
        warnings.append(
            NeoFitWarning(
                code="line_center_outside_coverage",
                message="Nominal line center is outside the requested window or valid spectrum coverage.",
                severity="error",
                context={**context, "line_center": float(config.center)},
            )
        )
        return False, warnings

    fit_mask = _fit_pixel_mask(spectrum, config)
    n_pixels = int(np.count_nonzero(fit_mask))
    n_required = max(int(local_config.require_min_pixels), _n_parameters(config) + 1)
    if n_pixels < n_required:
        warnings.append(
            NeoFitWarning(
                code="window_too_few_pixels",
                message=f"Only {n_pixels} valid pixels are available; at least {n_required} are required.",
                severity="error",
                context={**context, "n_pixels": n_pixels, "n_required": n_required},
            )
        )
        return False, warnings

    if local_config.edge_buffer > 0:
        covered_lo = max(lo, coverage_min)
        covered_hi = min(hi, coverage_max)
        if (config.center - covered_lo) <= local_config.edge_buffer or (covered_hi - config.center) <= local_config.edge_buffer:
            warnings.append(
                NeoFitWarning(
                    code="line_center_near_edge",
                    message="Nominal line center is close to the covered local-window edge.",
                    context={**context, "line_center": float(config.center), "edge_buffer": local_config.edge_buffer},
                )
            )
    return True, warnings


def fit_local(spectrum: Spectrum, config: LocalFitConfig) -> LocalFitResult:
    """Fit one or more local line-complex windows independently."""

    window_results = {}
    all_warnings: List[NeoFitWarning] = []
    for index, window_config in enumerate(config.windows):
        name = _window_name(window_config, index=index)
        can_fit, warnings = _validate_local_window(spectrum, window_config, config, name)
        metadata = spectrum.metadata.to_dict()
        metadata.update({"window_name": name, "mode": config.mode, "window": tuple(window_config.window)})
        if not can_fit:
            all_warnings.extend(warnings)
            window_results[name] = FitResult.failed(
                "Local window validation failed.",
                warnings=warnings,
                metadata=metadata,
            )
            continue
        try:
            result = fit_line_complex(spectrum, window_config)
            result.metadata["window_name"] = name
            if warnings:
                result.warnings.extend(warnings)
            window_results[name] = result
            all_warnings.extend(result.warnings)
        except Exception as exc:
            code = getattr(exc, "code", "fit_failed")
            warning = NeoFitWarning(
                code=code,
                message=str(exc),
                severity="error",
                context={"window": name},
            )
            all_warnings.extend(warnings)
            all_warnings.append(warning)
            window_results[name] = FitResult.failed(str(exc), warnings=warnings + [warning], metadata=metadata)

    return LocalFitResult(
        success=any(result.success for result in window_results.values()),
        window_results=window_results,
        warnings=all_warnings,
        metadata={"mode": config.mode, "n_windows": len(config.windows)},
    )
