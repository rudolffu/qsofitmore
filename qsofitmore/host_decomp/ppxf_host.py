"""Optional DESI pPXF host fitting plus qsofitmore handoff."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import json
import warnings

import numpy as np
import pandas as pd

from .config import DEFAULT_LINE_CENTERS, HostDecompConfig, default_config
from .dust import apply_galactic_dereddening
from .io import DEFAULT_FLUX_DENSITY_UNIT, SpectrumData, read_sparcli_spectrum
from .templates import PPXFTemplateLibrary, SAMPLE_WAVELENGTHS, load_ppxf_npz_templates


_C_KMS = 299792.458


@dataclass
class PreprocessedSpectrum:
    """DESI spectrum prepared for pPXF host fitting."""

    wave_obs: np.ndarray
    wave_rest: np.ndarray
    flux: np.ndarray
    error: np.ndarray
    ivar: Optional[np.ndarray]
    fit_mask: np.ndarray
    emission_mask: np.ndarray
    wave_log: np.ndarray
    log_wave: np.ndarray
    flux_log: np.ndarray
    noise_log: np.ndarray
    emission_mask_log: np.ndarray
    normalization: float
    redshift: float
    velscale: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class PPXFHostFitResult:
    """Output of the first pPXF host-continuum fit."""

    preprocessed: PreprocessedSpectrum
    templates: PPXFTemplateLibrary
    host_model_log: np.ndarray
    agn_model_log: np.ndarray
    total_model_log: np.ndarray
    residual_log: np.ndarray
    host_model: np.ndarray
    agn_model: np.ndarray
    total_model: np.ndarray
    residual: np.ndarray
    stellar_weights: np.ndarray
    agn_weights: np.ndarray
    stellar_template_scales: np.ndarray
    agn_slopes: np.ndarray
    stellar_velocity: float
    stellar_sigma: float
    chi2: float
    reduced_chi2: float
    status: str
    warnings: List[str] = field(default_factory=list)
    ppxf_result: Any = None


@dataclass
class HostSED:
    """Template-weighted host SED prediction."""

    wave_rest: np.ndarray
    host_flux: np.ndarray
    samples: Dict[str, float]
    flags: Dict[str, bool]
    warnings: List[str]


@dataclass
class HostDecompWorkflowResult:
    """Combined pPXF + qsofitmore decomposition result."""

    spectrum: SpectrumData
    ppxf_result: PPXFHostFitResult
    host_sed: HostSED
    host_subtracted_flux: np.ndarray
    qsofitmore_status: str
    qsofitmore_result_path: Optional[str]
    output_files: Dict[str, str]
    summary: Dict[str, Any]


def _require_ppxf():
    try:
        from ppxf.ppxf import ppxf
    except Exception as exc:
        raise RuntimeError(
            "pPXF is required for host decomposition but is not importable. "
            "Install it locally with `pip install ppxf` and keep template data outside qsofitmore."
        ) from exc
    return ppxf


def make_emission_line_mask(
    wave_rest: np.ndarray,
    line_mask_widths: Optional[Dict[str, float]] = None,
    broad_line_mask_widths: Optional[Dict[str, float]] = None,
    use_broad_masks: bool = True,
) -> np.ndarray:
    """Return a boolean mask for emission-line regions on a rest-frame grid.

    Widths are interpreted as velocity half-widths in km/s.
    """

    wave_rest = np.asarray(wave_rest, dtype=float)
    mask = np.zeros_like(wave_rest, dtype=bool)
    widths = dict(line_mask_widths or {})
    broad_widths = dict(broad_line_mask_widths or {})
    for name, center in DEFAULT_LINE_CENTERS.items():
        width = widths.get(name, widths.get("default", 800.0))
        if use_broad_masks and name in {"MgII", "Hdelta", "Hgamma", "Hbeta", "Halpha"}:
            width = broad_widths.get(name, broad_widths.get("default", width))
        delta = float(center) * float(width) / _C_KMS
        mask |= (wave_rest >= center - delta) & (wave_rest <= center + delta)
    return mask


def _log_resample(wave: np.ndarray, flux: np.ndarray, noise: np.ndarray, emission_mask: np.ndarray):
    wave = np.asarray(wave, dtype=float)
    log_wave = np.linspace(np.log(wave[0]), np.log(wave[-1]), len(wave))
    wave_log = np.exp(log_wave)
    flux_log = np.interp(wave_log, wave, flux)
    noise_log = np.interp(wave_log, wave, noise)
    mask_float = np.interp(wave_log, wave, emission_mask.astype(float))
    emission_mask_log = mask_float > 0.1
    velscale = float(np.diff(log_wave).mean() * _C_KMS)
    return wave_log, log_wave, flux_log, noise_log, emission_mask_log, velscale


def prepare_desi_for_host_decomp(
    spectrum: SpectrumData,
    redshift: Optional[float] = None,
    fit_range: Tuple[float, float] = (3600.0, 7000.0),
    ebv: Optional[float] = None,
    line_mask_widths: Optional[Dict[str, float]] = None,
    broad_line_mask_widths: Optional[Dict[str, float]] = None,
    use_broad_line_masks: bool = True,
) -> PreprocessedSpectrum:
    """Clean, rest-frame, mask, normalize, and log-resample a DESI spectrum."""

    z = spectrum.redshift if redshift is None else redshift
    if z is None:
        raise ValueError("A redshift is required for DESI host decomposition.")
    z = float(z)

    wave_obs = np.asarray(spectrum.wave_obs, dtype=float)
    flux = np.asarray(spectrum.flux, dtype=float)
    err = np.asarray(spectrum.uncertainty(), dtype=float)
    valid = np.isfinite(wave_obs) & np.isfinite(flux) & np.isfinite(err) & (wave_obs > 0) & (err > 0)
    if spectrum.ivar is not None:
        ivar = np.asarray(spectrum.ivar, dtype=float)
        valid &= np.isfinite(ivar) & (ivar > 0)
    else:
        ivar = None
    if spectrum.mask is not None:
        mask = np.asarray(spectrum.mask)
        valid &= mask == 0

    warnings_out: List[str] = []
    if np.sum(valid) < 20:
        warnings_out.append("few_valid_pixels_after_cleaning")

    wave_obs = wave_obs[valid]
    flux = apply_galactic_dereddening(wave_obs, flux[valid], ebv=ebv)
    err = err[valid]
    ivar_clean = ivar[valid] if ivar is not None else None
    order = np.argsort(wave_obs)
    wave_obs = wave_obs[order]
    flux = flux[order]
    err = err[order]
    if ivar_clean is not None:
        ivar_clean = ivar_clean[order]

    wave_rest = wave_obs / (1.0 + z)
    fit_mask = (wave_rest >= fit_range[0]) & (wave_rest <= fit_range[1])
    if np.sum(fit_mask) < 20:
        raise ValueError(
            f"Too few pixels in rest-frame fit range {fit_range}: {int(np.sum(fit_mask))}"
        )

    emission_mask = make_emission_line_mask(
        wave_rest,
        line_mask_widths=line_mask_widths,
        broad_line_mask_widths=broad_line_mask_widths,
        use_broad_masks=use_broad_line_masks,
    )
    fit_wave = wave_rest[fit_mask]
    fit_flux = flux[fit_mask]
    fit_err = err[fit_mask]
    fit_emission_mask = emission_mask[fit_mask]

    normalization = float(np.nanmedian(np.abs(fit_flux[np.isfinite(fit_flux)])))
    if not np.isfinite(normalization) or normalization <= 0:
        normalization = 1.0
        warnings_out.append("normalization_fallback_to_one")
    fit_flux_norm = fit_flux / normalization
    fit_err_norm = np.clip(fit_err / normalization, 1e-12, np.inf)

    wave_log, log_wave, flux_log, noise_log, emission_mask_log, velscale = _log_resample(
        fit_wave, fit_flux_norm, fit_err_norm, fit_emission_mask
    )

    return PreprocessedSpectrum(
        wave_obs=wave_obs,
        wave_rest=wave_rest,
        flux=flux,
        error=err,
        ivar=ivar_clean,
        fit_mask=fit_mask,
        emission_mask=emission_mask,
        wave_log=wave_log,
        log_wave=log_wave,
        flux_log=flux_log,
        noise_log=noise_log,
        emission_mask_log=emission_mask_log,
        normalization=normalization,
        redshift=z,
        velscale=velscale,
        metadata=dict(spectrum.metadata),
        warnings=warnings_out,
    )


def _build_agn_basis(wave: np.ndarray, slopes: Sequence[float]) -> np.ndarray:
    pivot = 5100.0
    basis = []
    for slope in slopes:
        vec = (np.asarray(wave, dtype=float) / pivot) ** float(slope)
        scale = np.nanmedian(np.abs(vec[np.isfinite(vec)]))
        basis.append(vec / (scale if scale > 0 else 1.0))
    return np.column_stack(basis) if basis else np.zeros((len(wave), 0))


def _resample_stellar_templates(prep: PreprocessedSpectrum, templates: PPXFTemplateLibrary):
    wave = prep.wave_log
    in_range = (wave >= templates.wavelength_coverage[0]) & (wave <= templates.wavelength_coverage[1])
    if np.sum(in_range) < len(wave):
        warnings.warn("Template wavelength coverage does not span the full pPXF fit range.", RuntimeWarning)
    matrix = np.empty((len(wave), templates.n_templates), dtype=float)
    for j in range(templates.n_templates):
        matrix[:, j] = np.interp(wave, templates.wave, templates.flux[:, j], left=0.0, right=0.0)
    scales = np.nanmedian(np.abs(matrix), axis=0)
    scales[~np.isfinite(scales) | (scales <= 0)] = 1.0
    return matrix / scales, scales, in_range


def run_ppxf_host_fit(
    preprocessed: PreprocessedSpectrum,
    templates: PPXFTemplateLibrary,
    agn_powerlaw_slopes: Sequence[float] = (-2.0, -1.5, -1.0, -0.5, 0.0),
    polynomial_degree: int = 4,
    multiplicative_polynomial_degree: int = 0,
    quiet: bool = True,
) -> PPXFHostFitResult:
    """Run the first pPXF stellar-host fit with masked emission regions."""

    ppxf = _require_ppxf()
    warnings_out = list(preprocessed.warnings) + list(templates.warnings)
    stellar_matrix, stellar_scales, in_template_range = _resample_stellar_templates(preprocessed, templates)
    agn_matrix = _build_agn_basis(preprocessed.wave_log, agn_powerlaw_slopes)
    fit_templates = np.column_stack([stellar_matrix, agn_matrix])

    good = (
        np.isfinite(preprocessed.flux_log)
        & np.isfinite(preprocessed.noise_log)
        & (preprocessed.noise_log > 0)
        & (~preprocessed.emission_mask_log)
        & in_template_range
    )
    goodpixels = np.flatnonzero(good)
    if goodpixels.size < max(20, fit_templates.shape[1] // 2):
        raise ValueError(f"Too few pPXF good pixels after masking: {goodpixels.size}")

    result = ppxf(
        fit_templates,
        preprocessed.flux_log,
        preprocessed.noise_log,
        preprocessed.velscale,
        start=[0.0, 150.0],
        goodpixels=goodpixels,
        degree=int(polynomial_degree),
        mdegree=int(multiplicative_polynomial_degree),
        quiet=quiet,
    )

    weights = np.asarray(getattr(result, "weights", np.zeros(fit_templates.shape[1])), dtype=float)
    n_stellar = stellar_matrix.shape[1]
    stellar_weights = weights[:n_stellar]
    agn_weights = weights[n_stellar:]
    host_log_norm = stellar_matrix @ stellar_weights
    agn_log_norm = agn_matrix @ agn_weights if agn_matrix.size else np.zeros_like(host_log_norm)
    total_log_norm = np.asarray(getattr(result, "bestfit", host_log_norm + agn_log_norm), dtype=float)
    residual_log_norm = preprocessed.flux_log - total_log_norm

    host_model = np.interp(preprocessed.wave_rest, preprocessed.wave_log, host_log_norm, left=np.nan, right=np.nan)
    agn_model = np.interp(preprocessed.wave_rest, preprocessed.wave_log, agn_log_norm, left=np.nan, right=np.nan)
    total_model = np.interp(preprocessed.wave_rest, preprocessed.wave_log, total_log_norm, left=np.nan, right=np.nan)

    host_model *= preprocessed.normalization
    agn_model *= preprocessed.normalization
    total_model *= preprocessed.normalization
    residual = preprocessed.flux - total_model

    sol = np.ravel(np.asarray(getattr(result, "sol", [np.nan, np.nan]), dtype=float))
    return PPXFHostFitResult(
        preprocessed=preprocessed,
        templates=templates,
        host_model_log=host_log_norm * preprocessed.normalization,
        agn_model_log=agn_log_norm * preprocessed.normalization,
        total_model_log=total_log_norm * preprocessed.normalization,
        residual_log=residual_log_norm * preprocessed.normalization,
        host_model=host_model,
        agn_model=agn_model,
        total_model=total_model,
        residual=residual,
        stellar_weights=stellar_weights,
        agn_weights=agn_weights,
        stellar_template_scales=stellar_scales,
        agn_slopes=np.asarray(agn_powerlaw_slopes, dtype=float),
        stellar_velocity=float(sol[0]) if sol.size else np.nan,
        stellar_sigma=float(sol[1]) if sol.size > 1 else np.nan,
        chi2=float(getattr(result, "chi2", np.nan)),
        reduced_chi2=float(getattr(result, "chi2", np.nan)),
        status="success",
        warnings=warnings_out,
        ppxf_result=result,
    )


def predict_host_sed(fit: PPXFHostFitResult) -> HostSED:
    """Evaluate the fitted host model on the full template wavelength grid."""

    templates = fit.templates
    scaled_templates = templates.flux / fit.stellar_template_scales[np.newaxis, :]
    host_flux = scaled_templates @ fit.stellar_weights * fit.preprocessed.normalization
    wave_min, wave_max = templates.wavelength_coverage
    samples: Dict[str, float] = {}
    warnings_out = list(fit.warnings)
    for name, wave in SAMPLE_WAVELENGTHS.items():
        if wave_min <= wave <= wave_max:
            samples[name] = float(np.interp(wave, templates.wave, host_flux))
        else:
            samples[name] = float("nan")
            warnings_out.append(f"{name}_outside_template_coverage")
    flags = {
        "template_covers_1um": bool(wave_min <= 10000.0 <= wave_max),
        "template_covers_1p6um": bool(wave_min <= 16000.0 <= wave_max),
        "template_covers_2p2um": bool(wave_min <= 22000.0 <= wave_max),
    }
    flags["nir_extrapolation_reliable"] = all(flags.values())
    flags["nir_extrapolation_not_available"] = not flags["nir_extrapolation_reliable"]
    if flags["nir_extrapolation_not_available"]:
        warnings_out.append("nir_extrapolation_not_available")
    return HostSED(
        wave_rest=templates.wave.copy(),
        host_flux=host_flux,
        samples=samples,
        flags=flags,
        warnings=warnings_out,
    )


def _write_csv(path: Path, data: Dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(path, index=False)
    return str(path)


def _interp_no_extrapolate(wave: float, grid: np.ndarray, values: np.ndarray) -> float:
    grid = np.asarray(grid, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(grid) & np.isfinite(values)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    finite_grid = grid[finite]
    finite_values = values[finite]
    order = np.argsort(finite_grid)
    finite_grid = finite_grid[order]
    finite_values = finite_values[order]
    if wave < finite_grid[0] or wave > finite_grid[-1]:
        return float("nan")
    return float(np.interp(wave, finite_grid, finite_values))


def _fitted_host_fraction_samples(fit: PPXFHostFitResult) -> Dict[str, float]:
    samples: Dict[str, float] = {}
    for host_name, wave in SAMPLE_WAVELENGTHS.items():
        suffix = host_name.removeprefix("fHost_")
        host = _interp_no_extrapolate(wave, fit.preprocessed.wave_rest, fit.host_model)
        agn = _interp_no_extrapolate(wave, fit.preprocessed.wave_rest, fit.agn_model)
        total = _interp_no_extrapolate(wave, fit.preprocessed.wave_rest, fit.total_model)
        samples[f"fHostFit_{suffix}"] = host
        samples[f"fAGNFit_{suffix}"] = agn
        samples[f"fTotalFit_{suffix}"] = total
        if np.isfinite(host) and np.isfinite(total) and total != 0:
            samples[f"fracHost_{suffix}"] = float(host / total)
        else:
            samples[f"fracHost_{suffix}"] = float("nan")
    return samples


def _summary_dict(
    spectrum: SpectrumData,
    fit: PPXFHostFitResult,
    sed: HostSED,
    input_file: str,
    output_dir: Path,
    qsofitmore_status: str,
    qsofitmore_result_path: Optional[str],
) -> Dict[str, Any]:
    summary = {
        "object_id": spectrum.object_id or spectrum.targetid,
        "targetid": spectrum.targetid,
        "redshift": fit.preprocessed.redshift,
        "ra": spectrum.ra,
        "dec": spectrum.dec,
        "input_file": input_file,
        "flux_density_unit": spectrum.metadata.get("flux_density_unit", DEFAULT_FLUX_DENSITY_UNIT),
        "template_file_used": fit.templates.source_path,
        "template_wavelength_min": fit.templates.wavelength_coverage[0],
        "template_wavelength_max": fit.templates.wavelength_coverage[1],
        "desi_fit_range_min": float(np.nanmin(fit.preprocessed.wave_log)),
        "desi_fit_range_max": float(np.nanmax(fit.preprocessed.wave_log)),
        "ppxf_status": fit.status,
        "qsofitmore_status": qsofitmore_status,
        "qsofitmore_result_path": qsofitmore_result_path,
        "stellar_velocity": fit.stellar_velocity,
        "stellar_velocity_dispersion": fit.stellar_sigma,
        "ppxf_reduced_chi2": fit.reduced_chi2,
        "qsofitmore_reduced_chi2": np.nan,
        "fAGN_5100": _interp_no_extrapolate(5100.0, fit.preprocessed.wave_rest, fit.agn_model),
        "broad_Halpha_detected": False,
        "broad_Hbeta_detected": False,
        "host_model_reliability": "template_weighted_ppxf_fit",
        "nir_extrapolation_reliability": "template_limited" if sed.flags["nir_extrapolation_reliable"] else "unavailable",
        "over_subtraction_risk": "inspect_residuals",
        "warnings": ";".join(sorted(set(fit.warnings + sed.warnings))),
    }
    summary.update(sed.samples)
    summary.update(_fitted_host_fraction_samples(fit))
    summary.update(sed.flags)
    return summary


def write_host_decomp_outputs(
    output_dir: str,
    spectrum: SpectrumData,
    fit: PPXFHostFitResult,
    sed: HostSED,
    host_subtracted_flux: np.ndarray,
    qsofitmore_status: str = "not_run",
    qsofitmore_result_path: Optional[str] = None,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Write standard host-decomposition products."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files: Dict[str, str] = {}
    files["desi_total_spectrum"] = _write_csv(
        out / "desi_total_spectrum.csv",
        {"wave_obs": fit.preprocessed.wave_obs, "wave_rest": fit.preprocessed.wave_rest, "flux": fit.preprocessed.flux, "error": fit.preprocessed.error},
    )
    files["desi_ppxf_host_model"] = _write_csv(
        out / "desi_ppxf_host_model.csv",
        {"wave_rest": fit.preprocessed.wave_rest, "host_flux": fit.host_model},
    )
    files["desi_ppxf_agn_continuum_model"] = _write_csv(
        out / "desi_ppxf_agn_continuum_model.csv",
        {"wave_rest": fit.preprocessed.wave_rest, "agn_flux": fit.agn_model},
    )
    files["desi_host_subtracted"] = _write_csv(
        out / "desi_host_subtracted.csv",
        {"wave_obs": fit.preprocessed.wave_obs, "wave_rest": fit.preprocessed.wave_rest, "flux": host_subtracted_flux, "error": fit.preprocessed.error},
    )
    files["host_sed_prediction"] = _write_csv(
        out / "host_sed_prediction.csv",
        {"wave_rest": sed.wave_rest, "host_flux": sed.host_flux},
    )
    npz_path = out / "host_decomp_result.npz"
    np.savez(
        npz_path,
        wave_obs=fit.preprocessed.wave_obs,
        wave_rest=fit.preprocessed.wave_rest,
        flux=fit.preprocessed.flux,
        host_model=fit.host_model,
        agn_model=fit.agn_model,
        total_model=fit.total_model,
        host_subtracted_flux=host_subtracted_flux,
        host_sed_wave=sed.wave_rest,
        host_sed_flux=sed.host_flux,
        stellar_weights=fit.stellar_weights,
        agn_weights=fit.agn_weights,
    )
    files["host_decomp_result"] = str(npz_path)
    summary = _summary_dict(
        spectrum,
        fit,
        sed,
        spectrum.metadata.get("input_file", ""),
        out,
        qsofitmore_status,
        qsofitmore_result_path,
    )
    summary_json = out / "host_decomp_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    files["host_decomp_summary_json"] = str(summary_json)
    summary_csv = out / "host_decomp_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    files["host_decomp_summary_csv"] = str(summary_csv)
    qsofitmore_model = out / "qsofitmore_model.csv"
    if qsofitmore_model.exists():
        files["qsofitmore_model"] = str(qsofitmore_model)
    return files, summary


def _try_run_qsofitmore(
    spectrum: SpectrumData,
    fit: PPXFHostFitResult,
    host_subtracted_flux: np.ndarray,
    output_dir: Path,
    line_list_path: Optional[str] = None,
    qsofitmore_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Optional[str]]:
    try:
        from qsofitmore import QSOFitNew
    except Exception as exc:
        return f"not_run_import_failed:{exc}", None

    qsofitmore_kwargs = dict(qsofitmore_kwargs or {})
    linefit = bool(line_list_path)
    try:
        finite = (
            np.isfinite(fit.preprocessed.wave_obs)
            & np.isfinite(host_subtracted_flux)
            & np.isfinite(fit.preprocessed.error)
            & (fit.preprocessed.error > 0)
        )
        if np.sum(finite) < 20:
            return "failed:too_few_finite_host_subtracted_pixels", None
        q = QSOFitNew(
            lam=fit.preprocessed.wave_obs[finite],
            flux=host_subtracted_flux[finite],
            err=fit.preprocessed.error[finite],
            z=fit.preprocessed.redshift,
            ra=spectrum.ra or 0,
            dec=spectrum.dec or 0,
            name=str(spectrum.object_id or spectrum.targetid or "object"),
            is_sdss=False,
            path=str(output_dir) + "/",
        )
        q.Fit(
            deredden=False,
            decomposition_host=False,
            include_iron=False,
            poly=False,
            BC=False,
            MC=False,
            linefit=linefit,
            line_list_path=line_list_path,
            save_result=True,
            plot_fig=False,
            save_fig=False,
            save_fits_path=str(output_dir),
            **qsofitmore_kwargs,
        )
        result_path = output_dir / f"res_qsofitmore_{q.name}.fits"
        model_path = output_dir / "qsofitmore_model.csv"
        payload = {"wave_obs": getattr(q, "wave", fit.preprocessed.wave_obs[finite])}
        if hasattr(q, "f_conti_model"):
            payload["continuum_model"] = q.f_conti_model
        pd.DataFrame(payload).to_csv(model_path, index=False)
        return "success", str(result_path if result_path.exists() else model_path)
    except Exception as exc:
        return f"failed:{exc}", None


def run_ppxf_qsofitmore_decomposition(
    input_path: str,
    row_index: Optional[int] = None,
    redshift: Optional[float] = None,
    object_id: Optional[str] = None,
    template_root: str = "~/tools/ppxf_data",
    template_file: str = "spectra_emiles_9.0.npz",
    output_dir: Optional[str] = None,
    fit_range: Tuple[float, float] = (3600.0, 7000.0),
    run_qsofitmore: bool = False,
    n_iterations: int = 1,
    line_list_path: Optional[str] = None,
    config: Optional[HostDecompConfig] = None,
) -> HostDecompWorkflowResult:
    """Run one pPXF host pass plus an optional qsofitmore host-subtracted pass."""

    cfg = config or default_config()
    spectrum = read_sparcli_spectrum(input_path, row_index=row_index, redshift=redshift, object_id=object_id)
    obj = object_id or spectrum.object_id or spectrum.targetid or "object"
    out = Path(output_dir or Path(cfg.output_dir) / str(obj))
    templates = load_ppxf_npz_templates(template_root=template_root, template_file=template_file)
    prep = prepare_desi_for_host_decomp(
        spectrum,
        redshift=redshift,
        fit_range=fit_range,
        line_mask_widths=cfg.line_mask_widths,
        broad_line_mask_widths=cfg.broad_line_mask_widths,
    )
    fit = run_ppxf_host_fit(
        prep,
        templates,
        agn_powerlaw_slopes=cfg.agn_powerlaw_slopes,
        polynomial_degree=cfg.polynomial_degree,
        multiplicative_polynomial_degree=cfg.multiplicative_polynomial_degree,
    )
    sed = predict_host_sed(fit)
    host_subtracted_flux = prep.flux - fit.host_model
    qsofitmore_status = "not_run"
    qsofitmore_result_path = None
    if run_qsofitmore:
        if not line_list_path:
            fit.warnings.append("qsofitmore_linefit_skipped_no_line_table")
        qsofitmore_status, qsofitmore_result_path = _try_run_qsofitmore(
            spectrum,
            fit,
            host_subtracted_flux,
            out,
            line_list_path=line_list_path,
        )
    files, summary = write_host_decomp_outputs(
        str(out),
        spectrum,
        fit,
        sed,
        host_subtracted_flux,
        qsofitmore_status=qsofitmore_status,
        qsofitmore_result_path=qsofitmore_result_path,
    )
    try:
        from .plots import plot_desi_ppxf_fit, plot_host_sed_prediction

        files["plot_desi_ppxf_fit"] = plot_desi_ppxf_fit(fit, out / "diagnostic_desi_ppxf_fit.png", host_sed=sed)
        files["plot_host_sed_prediction"] = plot_host_sed_prediction(sed, out / "diagnostic_host_sed_prediction.png")
    except Exception as exc:
        summary["warnings"] = (summary.get("warnings", "") + f";plot_failed:{exc}").strip(";")

    return HostDecompWorkflowResult(
        spectrum=spectrum,
        ppxf_result=fit,
        host_sed=sed,
        host_subtracted_flux=host_subtracted_flux,
        qsofitmore_status=qsofitmore_status,
        qsofitmore_result_path=qsofitmore_result_path,
        output_files=files,
        summary=summary,
    )
