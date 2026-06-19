"""Global continuum and constrained H-beta/[O III] fitting."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import least_squares, lsq_linear

from .config import (
    GlobalContinuumConfig,
    HalphaComplexConfig,
    HbetaComplexConfig,
    MgIIComplexConfig,
    UncertaintyConfig,
)
from .complex_recipes import ComplexRecipe
from . import complex_recipes
from .generic_complex import fit_generic_complex, resolve_recipe_coverage
from .global_result import (
    EmissionComplexResult,
    GlobalContinuumResult,
    HbetaComplexResult,
    NeoFitWorkflowResult,
)
from .spectrum import Spectrum
from .templates import (
    evaluate_balmer_series,
    evaluate_balmer_series_with_derivative,
    load_balmer_template,
    load_iron_template,
)
from .templates.iron import evaluate_iron_basis, evaluate_iron_basis_with_derivative
from .variable_projection import (
    VariableProjectionError,
    evaluate_profile_chi2,
    optimizer_result_adapter,
    solve_variable_projection,
)
from .warnings import NeoFitWarning

C_KMS = 299792.458
FWHM_TO_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))
HBETA_WAVE = 4862.68
OIII_4959_WAVE = 4960.30
OIII_5007_WAVE = 5008.24
HEII_WAVE = 4687.02
MGII_WAVE = 2798.75
HALPHA_WAVE = 6564.61
NII_6549_WAVE = 6549.85
NII_6585_WAVE = 6585.28
SII_6718_WAVE = 6718.29
SII_6733_WAVE = 6732.67


def balmer_continuum_basis(
    wave_rest: np.ndarray,
    temperature_k: float = 15000.0,
    tau_edge: float = 1.0,
    edge: float = 3646.0,
    min_wave: float = 2000.0,
) -> np.ndarray:
    """Return a unit-edge Dietrich-style Balmer-continuum basis."""

    wave = np.asarray(wave_rest, dtype=float)
    if temperature_k <= 0 or tau_edge <= 0:
        raise ValueError("Balmer-continuum temperature and tau_edge must be positive.")
    h = 6.62607015e-27
    c = 2.99792458e18
    k = 1.380649e-16

    def planck_lambda(lam):
        x = h * c / (lam * k * temperature_k)
        return lam**-5 / np.expm1(np.clip(x, 1.0e-8, 700.0))

    edge_value = planck_lambda(edge) * (1.0 - np.exp(-tau_edge))
    out = np.zeros_like(wave)
    active = (wave >= min_wave) & (wave <= edge)
    tau = tau_edge * (wave[active] / edge) ** 3
    out[active] = planck_lambda(wave[active]) * (1.0 - np.exp(-tau)) / edge_value
    return out


def _window_mask(wave: np.ndarray, windows: Sequence[Tuple[float, float]]) -> np.ndarray:
    mask = np.zeros_like(wave, dtype=bool)
    for lo, hi in windows:
        mask |= (wave >= float(lo)) & (wave <= float(hi))
    return mask


def _bounds(bounds, default_lo=-np.inf, default_hi=np.inf):
    lo, hi = bounds
    return default_lo if lo is None else float(lo), default_hi if hi is None else float(hi)


def _covariance_from_jacobian(
    jacobian: Optional[np.ndarray],
    reduced_chi2: float,
    names: Sequence[str],
) -> Tuple[Optional[np.ndarray], Dict[str, float], List[NeoFitWarning]]:
    warnings: List[NeoFitWarning] = []
    if jacobian is None or np.size(jacobian) == 0:
        return None, {name: np.nan for name in names}, warnings
    jac = np.asarray(jacobian, dtype=float)
    info = jac.T @ jac
    rank = int(np.linalg.matrix_rank(info))
    if rank < info.shape[0]:
        warnings.append(
            NeoFitWarning(
                code="covariance_rank_deficient",
                message="The fitted Jacobian is rank deficient; pseudoinverse uncertainties may be unstable.",
                context={"rank": rank, "n_parameters": int(info.shape[0])},
            )
        )
    covariance = np.linalg.pinv(info) * (float(reduced_chi2) if np.isfinite(reduced_chi2) else 1.0)
    errors = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))
    return covariance, {name: float(errors[i]) for i, name in enumerate(names)}, warnings


def _active_bound_warnings(result, names: Sequence[str]) -> List[NeoFitWarning]:
    warnings = []
    if result is None or not hasattr(result, "active_mask"):
        return warnings
    for index in np.where(np.asarray(result.active_mask) != 0)[0]:
        warnings.append(
            NeoFitWarning(
                code="parameter_at_bound",
                message=f"Parameter {names[index]} finished on an optimizer bound.",
                context={"parameter": names[index], "bound_side": int(result.active_mask[index])},
            )
        )
    return warnings


class _ContinuumContext:
    def __init__(self, spectrum: Spectrum, config: GlobalContinuumConfig):
        self.spectrum = spectrum
        self.config = config
        self.wave = spectrum.wave_rest
        self.warnings: List[NeoFitWarning] = []
        self.names: List[str] = []
        self.initial: List[float] = []
        self.lower: List[float] = []
        self.upper: List[float] = []
        self.uv_template = None
        self.opt_template = None
        self.balmer_template = None

        valid = spectrum.valid_mask
        self.base_fit_mask = valid & _window_mask(self.wave, config.continuum_windows)
        self.base_fit_mask &= ~_window_mask(self.wave, config.mask_windows)
        self._configure_parameters()

    def _add(self, name: str, value: float, bounds) -> None:
        lo, hi = _bounds(bounds)
        self.names.append(name)
        self.initial.append(float(np.clip(value, lo, hi)))
        self.lower.append(lo)
        self.upper.append(hi)

    def _overlap_count(self, coverage: Tuple[float, float]) -> int:
        lo, hi = coverage
        return int(np.count_nonzero(self.base_fit_mask & (self.wave >= lo) & (self.wave <= hi)))

    def _configure_parameters(self) -> None:
        cfg = self.config
        if cfg.power_law.enabled:
            valid_flux = self.spectrum.flux[self.base_fit_mask]
            norm = cfg.power_law.norm
            if norm is None:
                norm = max(float(np.nanmedian(valid_flux)) if valid_flux.size else 1.0, 1.0e-6)
            self._add("power_law.norm", norm, cfg.power_law.norm_bounds)
            self._add("power_law.slope", cfg.power_law.slope, cfg.power_law.slope_bounds)

        for label, iron_cfg in (("uv_iron", cfg.uv_iron), ("optical_iron", cfg.optical_iron)):
            if iron_cfg is None or not iron_cfg.enabled:
                continue
            template = load_iron_template(
                iron_cfg.template,
                template_path=iron_cfg.template_path,
                normalization=iron_cfg.normalization,
            )
            coverage = template.coverage or (float(template.wave_rest.min()), float(template.wave_rest.max()))
            if self._overlap_count(coverage) < cfg.min_component_pixels:
                self.warnings.append(
                    NeoFitWarning(
                        code="global_component_disabled_no_coverage",
                        message=f"{label} was disabled because too few continuum pixels overlap its template.",
                        context={"component": label, "coverage": coverage},
                    )
                )
                continue
            if label == "uv_iron":
                self.uv_template = template
            else:
                self.opt_template = template
            self._add(f"{label}.amp", iron_cfg.amp, iron_cfg.amp_bounds)
            self._add(f"{label}.fwhm_kms", iron_cfg.fwhm_kms, iron_cfg.fwhm_bounds)

        bc = cfg.balmer_continuum
        bc_pixels = self.base_fit_mask & (self.wave >= bc.min_wave) & (self.wave <= bc.edge)
        if bc.enabled and np.count_nonzero(bc_pixels) >= cfg.min_component_pixels:
            self._add("balmer_continuum.amp", bc.amplitude, bc.amplitude_bounds)
        elif bc.enabled:
            self.warnings.append(
                NeoFitWarning(
                    code="global_component_disabled_no_coverage",
                    message="Balmer continuum was disabled because its wavelength range is not sufficiently covered.",
                    context={"component": "balmer_continuum"},
                )
            )

        bs = cfg.balmer_series
        bs_pixels = self.base_fit_mask & (self.wave >= 3500.0) & (self.wave <= 4260.0)
        if bs.enabled and np.count_nonzero(bs_pixels) >= cfg.min_component_pixels:
            self.balmer_template = load_balmer_template(
                log10_ne=bs.log10_ne, n_min=bs.n_min, provenance=bs.provenance
            )
            self.warnings.extend(self.balmer_template.warnings)
            self._add("balmer_series.amp", bs.amplitude, bs.amplitude_bounds)
            if self._balmer_fixed_fwhm() is None:
                self._add("balmer_series.fwhm_kms", bs.fwhm_kms, bs.fwhm_bounds)
        elif bs.enabled:
            self.warnings.append(
                NeoFitWarning(
                    code="global_component_disabled_no_coverage",
                    message="High-order Balmer series was disabled because 3500-4260 Angstrom is not sufficiently covered.",
                    context={"component": "balmer_series"},
                )
            )

        self.initial = np.asarray(self.initial, dtype=float)
        self.lower = np.asarray(self.lower, dtype=float)
        self.upper = np.asarray(self.upper, dtype=float)
        self.index = {name: i for i, name in enumerate(self.names)}
        self._initialize_linear_amplitudes()

    def _balmer_fixed_fwhm(self) -> Optional[float]:
        config = self.config.balmer_series
        if config.fixed_fwhm_kms is not None:
            return float(config.fixed_fwhm_kms)
        if not config.fit_fwhm:
            return float(config.fwhm_kms)
        return None

    def _get(self, theta: np.ndarray, name: str, default: float = 0.0) -> float:
        return float(theta[self.index[name]]) if name in self.index else float(default)

    def _initialize_linear_amplitudes(self) -> None:
        if not self.names or not np.any(self.base_fit_mask):
            return
        wave = self.wave[self.base_fit_mask]
        columns = []
        names = []
        if "power_law.norm" in self.index:
            slope = self._get(self.initial, "power_law.slope")
            columns.append((wave / self.config.power_law.pivot) ** slope)
            names.append("power_law.norm")
        if self.uv_template is not None:
            fwhm = self._get(self.initial, "uv_iron.fwhm_kms")
            columns.append(evaluate_iron_basis(self.uv_template, wave, fwhm))
            names.append("uv_iron.amp")
        if self.opt_template is not None:
            fwhm = self._get(self.initial, "optical_iron.fwhm_kms")
            columns.append(evaluate_iron_basis(self.opt_template, wave, fwhm))
            names.append("optical_iron.amp")
        if "balmer_continuum.amp" in self.index:
            bc = self.config.balmer_continuum
            columns.append(balmer_continuum_basis(wave, bc.temperature_k, bc.tau_edge, bc.edge, bc.min_wave))
            names.append("balmer_continuum.amp")
        if "balmer_series.amp" in self.index:
            fwhm = self._balmer_fixed_fwhm()
            if fwhm is None:
                fwhm = self._get(self.initial, "balmer_series.fwhm_kms")
            columns.append(evaluate_balmer_series(self.balmer_template, wave, fwhm))
            names.append("balmer_series.amp")
        if not columns:
            return
        design = np.column_stack(columns)
        err = self.spectrum.err[self.base_fit_mask]
        weighted_design = design / err[:, None]
        weighted_flux = self.spectrum.flux[self.base_fit_mask] / err
        try:
            solution = lsq_linear(weighted_design, weighted_flux, bounds=(0.0, np.inf)).x
            for name, value in zip(names, solution):
                idx = self.index[name]
                self.initial[idx] = np.clip(value, self.lower[idx], self.upper[idx])
        except Exception:
            pass

    @property
    def linear_names(self) -> List[str]:
        return [
            name
            for name in self.names
            if name == "power_law.norm" or name.endswith(".amp")
        ]

    @property
    def nonlinear_names(self) -> List[str]:
        linear = set(self.linear_names)
        return [name for name in self.names if name not in linear]

    def _named_values(self, names: Sequence[str], values: np.ndarray) -> Dict[str, float]:
        return {name: float(values[index]) for index, name in enumerate(names)}

    def separable_design(
        self,
        nonlinear: np.ndarray,
        wave: np.ndarray,
        need_derivatives: bool,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        nonlinear_values = self._named_values(self.nonlinear_names, nonlinear)
        columns: List[np.ndarray] = []
        derivative_columns = [
            [] for _ in self.nonlinear_names
        ] if need_derivatives else None

        def append_column(
            basis: np.ndarray,
            derivatives: Optional[Dict[str, np.ndarray]] = None,
        ) -> None:
            columns.append(np.asarray(basis, dtype=float))
            if derivative_columns is None:
                return
            derivatives = derivatives or {}
            for index, name in enumerate(self.nonlinear_names):
                derivative_columns[index].append(
                    np.asarray(derivatives.get(name, np.zeros_like(wave)), dtype=float)
                )

        if "power_law.norm" in self.index:
            slope = nonlinear_values["power_law.slope"]
            basis = (wave / self.config.power_law.pivot) ** slope
            append_column(
                basis,
                {"power_law.slope": basis * np.log(wave / self.config.power_law.pivot)},
            )
        if self.uv_template is not None:
            fwhm = nonlinear_values["uv_iron.fwhm_kms"]
            if need_derivatives:
                basis, derivative = evaluate_iron_basis_with_derivative(
                    self.uv_template, wave, fwhm
                )
            else:
                basis = evaluate_iron_basis(self.uv_template, wave, fwhm)
                derivative = None
            append_column(
                basis,
                {"uv_iron.fwhm_kms": derivative} if derivative is not None else None,
            )
        if self.opt_template is not None:
            fwhm = nonlinear_values["optical_iron.fwhm_kms"]
            if need_derivatives:
                basis, derivative = evaluate_iron_basis_with_derivative(
                    self.opt_template, wave, fwhm
                )
            else:
                basis = evaluate_iron_basis(self.opt_template, wave, fwhm)
                derivative = None
            append_column(
                basis,
                {"optical_iron.fwhm_kms": derivative} if derivative is not None else None,
            )
        if "balmer_continuum.amp" in self.index:
            bc = self.config.balmer_continuum
            append_column(
                balmer_continuum_basis(
                    wave, bc.temperature_k, bc.tau_edge, bc.edge, bc.min_wave
                )
            )
        if "balmer_series.amp" in self.index:
            fwhm = self._balmer_fixed_fwhm()
            if fwhm is None:
                fwhm = nonlinear_values["balmer_series.fwhm_kms"]
            if need_derivatives and "balmer_series.fwhm_kms" in nonlinear_values:
                basis, derivative = evaluate_balmer_series_with_derivative(
                    self.balmer_template, wave, fwhm
                )
                derivatives = {"balmer_series.fwhm_kms": derivative}
            else:
                basis = evaluate_balmer_series(self.balmer_template, wave, fwhm)
                derivatives = None
            append_column(basis, derivatives)

        design = np.column_stack(columns)
        if derivative_columns is None:
            return design, None
        return design, tuple(np.column_stack(items) for items in derivative_columns)

    def assemble_full_parameters(
        self,
        linear: np.ndarray,
        nonlinear: np.ndarray,
    ) -> np.ndarray:
        values = np.empty(len(self.names), dtype=float)
        for name, value in zip(self.linear_names, linear):
            values[self.index[name]] = value
        for name, value in zip(self.nonlinear_names, nonlinear):
            values[self.index[name]] = value
        return values

    def separable_initial_and_bounds(self):
        linear_indices = np.array([self.index[name] for name in self.linear_names], dtype=int)
        nonlinear_indices = np.array([self.index[name] for name in self.nonlinear_names], dtype=int)
        return (
            self.initial[linear_indices],
            (self.lower[linear_indices], self.upper[linear_indices]),
            self.initial[nonlinear_indices],
            (self.lower[nonlinear_indices], self.upper[nonlinear_indices]),
        )

    def components(self, theta: np.ndarray, wave: np.ndarray) -> Dict[str, np.ndarray]:
        components: Dict[str, np.ndarray] = {}
        if "power_law.norm" in self.index:
            norm = self._get(theta, "power_law.norm")
            slope = self._get(theta, "power_law.slope")
            components["power_law"] = norm * (wave / self.config.power_law.pivot) ** slope
        if self.uv_template is not None:
            components["uv_iron"] = self._get(theta, "uv_iron.amp") * evaluate_iron_basis(
                self.uv_template, wave, self._get(theta, "uv_iron.fwhm_kms")
            )
        if self.opt_template is not None:
            components["optical_iron"] = self._get(theta, "optical_iron.amp") * evaluate_iron_basis(
                self.opt_template, wave, self._get(theta, "optical_iron.fwhm_kms")
            )
        if "balmer_continuum.amp" in self.index:
            bc = self.config.balmer_continuum
            components["balmer_continuum"] = self._get(theta, "balmer_continuum.amp") * balmer_continuum_basis(
                wave, bc.temperature_k, bc.tau_edge, bc.edge, bc.min_wave
            )
        if self.balmer_template is not None:
            fwhm = self._balmer_fixed_fwhm()
            if fwhm is None:
                fwhm = self._get(theta, "balmer_series.fwhm_kms")
            components["balmer_series"] = self._get(theta, "balmer_series.amp") * evaluate_balmer_series(
                self.balmer_template, wave, fwhm
            )
        return components

    def model(self, theta: np.ndarray, wave: np.ndarray) -> np.ndarray:
        components = self.components(theta, wave)
        return sum(components.values(), np.zeros_like(wave, dtype=float))


def _full_separable_jacobian(
    context,
    design: np.ndarray,
    design_derivatives: Sequence[np.ndarray],
    linear: np.ndarray,
    err: np.ndarray,
) -> np.ndarray:
    jacobian = np.zeros((design.shape[0], len(context.names)), dtype=float)
    for column, name in enumerate(context.linear_names):
        jacobian[:, context.index[name]] = -design[:, column] / err
    for derivative, name in zip(design_derivatives, context.nonlinear_names):
        jacobian[:, context.index[name]] = -(derivative @ linear) / err
    return jacobian


def _solve_separable_once(context, wave, flux, err, start, max_nfev, jacobian_method):
    _, linear_bounds, _, nonlinear_bounds = context.separable_initial_and_bounds()
    nonlinear_initial = np.array(
        [start[context.index[name]] for name in context.nonlinear_names], dtype=float
    )
    evaluator = lambda nonlinear, need_derivatives: context.separable_design(
        nonlinear, wave, need_derivatives
    )
    result = solve_variable_projection(
        flux,
        err,
        nonlinear_initial,
        nonlinear_bounds,
        linear_bounds,
        evaluator,
        jacobian_method=jacobian_method,
        max_nfev=max_nfev,
    )
    primary_result = result
    best_start = None
    best_chi2 = float(np.sum(result.residual**2))
    probe_count = 0
    for index, name in enumerate(context.nonlinear_names):
        if not name.endswith(".fwhm_kms"):
            continue
        for boundary in (nonlinear_bounds[0][index], nonlinear_bounds[1][index]):
            if not np.isfinite(boundary) or np.isclose(result.nonlinear[index], boundary):
                continue
            candidate = result.nonlinear.copy()
            candidate[index] = boundary
            try:
                candidate_chi2 = evaluate_profile_chi2(
                    flux, err, candidate, linear_bounds, evaluator
                )
            except VariableProjectionError:
                continue
            probe_count += 1
            if candidate_chi2 < best_chi2 - max(1.0e-8, 1.0e-10 * best_chi2):
                best_chi2 = candidate_chi2
                best_start = candidate
    if best_start is not None:
        try:
            restarted = solve_variable_projection(
                flux,
                err,
                best_start,
                nonlinear_bounds,
                linear_bounds,
                evaluator,
                jacobian_method=jacobian_method,
                max_nfev=max_nfev,
            )
            selected = (
                restarted
                if np.sum(restarted.residual**2) < np.sum(primary_result.residual**2)
                else primary_result
            )
            total_nfev = primary_result.nfev + restarted.nfev
            total_njev = primary_result.njev + restarted.njev
            total_linear_solves = (
                primary_result.linear_solve_count + restarted.linear_solve_count
            )
            result = selected
            result.nfev = total_nfev
            result.njev = total_njev
            result.linear_solve_count = total_linear_solves
        except VariableProjectionError:
            result = primary_result
    result.linear_solve_count += probe_count
    full_x = context.assemble_full_parameters(result.linear, result.nonlinear)
    full_jacobian = _full_separable_jacobian(
        context,
        result.design,
        result.design_derivatives,
        result.linear,
        err,
    )
    full_active_mask = np.zeros(len(context.names), dtype=int)
    for value, name in zip(result.linear_active_mask, context.linear_names):
        full_active_mask[context.index[name]] = value
    for value, name in zip(result.nonlinear_active_mask, context.nonlinear_names):
        full_active_mask[context.index[name]] = value
    return optimizer_result_adapter(
        full_x=full_x,
        full_jacobian=full_jacobian,
        full_active_mask=full_active_mask,
        result=result,
    )


def _solve_legacy_once(context, wave, flux, err, start, max_nfev):
    return least_squares(
        lambda theta: (flux - context.model(theta, wave)) / err,
        start,
        bounds=(context.lower, context.upper),
        jac="2-point",
        max_nfev=max_nfev,
    )


def _solve_once_with_fallback(context, wave, flux, err, start, config):
    requested = config.optimizer_method
    if requested == "legacy_joint":
        return (
            _solve_legacy_once(context, wave, flux, err, start, config.max_nfev),
            "legacy_joint",
            None,
        )
    try:
        result = _solve_separable_once(
            context,
            wave,
            flux,
            err,
            start,
            config.max_nfev,
            config.jacobian_method,
        )
        if not result.success:
            raise VariableProjectionError(result.message)
        return result, "variable_projection", None
    except Exception as exc:
        if requested == "variable_projection":
            raise
        result = _solve_legacy_once(context, wave, flux, err, start, config.max_nfev)
        return result, "legacy_joint", str(exc)


def fit_global_continuum(
    spectrum: Spectrum,
    config: Optional[GlobalContinuumConfig] = None,
    *,
    compute_covariance: bool = True,
) -> GlobalContinuumResult:
    """Fit the global AGN continuum on legacy line-free windows."""

    cfg = config or GlobalContinuumConfig()
    context = _ContinuumContext(spectrum, cfg)
    if np.count_nonzero(context.base_fit_mask) <= len(context.names):
        raise ValueError("Too few valid continuum-window pixels for the active global model.")

    clip_mask = context.base_fit_mask.copy()
    result = None
    optimizer_used = "variable_projection"
    fallback_reasons: List[str] = []
    total_nfev = 0
    total_njev = 0
    total_linear_solves = 0
    for _ in range(max(int(cfg.clip_passes), 0)):
        wave = context.wave[clip_mask]
        flux = spectrum.flux[clip_mask]
        err = spectrum.err[clip_mask]
        start = context.initial if result is None else result.x
        result, method, fallback_reason = _solve_once_with_fallback(
            context, wave, flux, err, start, cfg
        )
        optimizer_used = method if method == "legacy_joint" else optimizer_used
        if fallback_reason is not None:
            fallback_reasons.append(fallback_reason)
        total_nfev += int(getattr(result, "nfev", 0) or 0)
        total_njev += int(getattr(result, "njev", 0) or 0)
        total_linear_solves += int(getattr(result, "linear_solve_count", 0) or 0)
        standardized = (spectrum.flux[context.base_fit_mask] - context.model(result.x, context.wave[context.base_fit_mask]))
        standardized /= spectrum.err[context.base_fit_mask]
        keep = (standardized >= -cfg.clip_low_sigma) & (standardized <= cfg.clip_high_sigma)
        updated = context.base_fit_mask.copy()
        updated[context.base_fit_mask] = keep
        if np.array_equal(updated, clip_mask):
            break
        clip_mask = updated

    wave_fit = context.wave[clip_mask]
    flux_fit = spectrum.flux[clip_mask]
    err_fit = spectrum.err[clip_mask]
    start = context.initial if result is None else result.x
    result, method, fallback_reason = _solve_once_with_fallback(
        context, wave_fit, flux_fit, err_fit, start, cfg
    )
    optimizer_used = method if method == "legacy_joint" else optimizer_used
    if fallback_reason is not None:
        fallback_reasons.append(fallback_reason)
    total_nfev += int(getattr(result, "nfev", 0) or 0)
    total_njev += int(getattr(result, "njev", 0) or 0)
    total_linear_solves += int(getattr(result, "linear_solve_count", 0) or 0)
    residual = (flux_fit - context.model(result.x, wave_fit)) / err_fit
    chi2 = float(np.sum(residual**2))
    dof = max(int(wave_fit.size - result.x.size), 0)
    reduced = float(chi2 / dof) if dof else np.nan
    if compute_covariance:
        covariance, errors, cov_warnings = _covariance_from_jacobian(result.jac, reduced, context.names)
    else:
        covariance = None
        errors = {name: np.nan for name in context.names}
        cov_warnings = []
    warnings = list(context.warnings) + cov_warnings + _active_bound_warnings(result, context.names)
    if fallback_reasons:
        warnings.append(
            NeoFitWarning(
                code="optimizer_fallback_legacy",
                message="Variable projection failed; the legacy joint optimizer was used.",
                context={"reasons": fallback_reasons},
            )
        )
    if not result.success:
        warnings.append(NeoFitWarning(code="fit_failed", message=str(result.message), severity="error"))
    components = context.components(result.x, context.wave)
    metadata = spectrum.metadata.to_dict()
    metadata.update(
        {
            "continuum_windows": list(cfg.continuum_windows),
            "mask_windows": list(cfg.mask_windows),
            "balmer_template": context.balmer_template.name if context.balmer_template is not None else None,
            "balmer_template_source": (
                context.balmer_template.source_path if context.balmer_template is not None else None
            ),
            "balmer_series_fwhm_fixed": context._balmer_fixed_fwhm() is not None,
            "optimizer_requested": cfg.optimizer_method,
            "optimizer_used": optimizer_used,
            "jacobian_method": (
                cfg.jacobian_method if optimizer_used == "variable_projection" else "2-point"
            ),
            "optimizer_fallback": bool(fallback_reasons),
            "n_linear_parameters": len(context.linear_names),
            "n_nonlinear_parameters": len(context.nonlinear_names),
            "nonlinear_nfev": total_nfev,
            "nonlinear_njev": total_njev,
            "linear_solve_count": total_linear_solves,
        }
    )
    if "balmer_series.amp" in context.index:
        balmer_amp = float(result.x[context.index["balmer_series.amp"]])
        balmer_fwhm = context._balmer_fixed_fwhm()
        if balmer_fwhm is None:
            balmer_fwhm = float(result.x[context.index["balmer_series.fwhm_kms"]])
        scale = spectrum.flux_density_scale_to_cgs
        metadata.update(
            {
                "balmer_series_implied_hbeta_flux_input": balmer_amp,
                "balmer_series_implied_hbeta_flux_cgs": (
                    balmer_amp * (1.0 + spectrum.z) * float(scale) if scale is not None else np.nan
                ),
                "balmer_series_fwhm_kms": float(balmer_fwhm),
                "balmer_series_amplitude_definition": (
                    "Integrated Hbeta flux implied by the template's Hbeta-relative line ratios"
                ),
            }
        )
    return GlobalContinuumResult(
        success=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        param_values={name: float(result.x[i]) for i, name in enumerate(context.names)},
        param_errors=errors,
        covariance=covariance,
        chi2=chi2,
        dof=dof,
        reduced_chi2=reduced,
        wave_rest=context.wave.copy(),
        model=context.model(result.x, context.wave),
        component_models=components,
        fit_mask=context.base_fit_mask.copy(),
        clip_mask=clip_mask.copy(),
        warnings=warnings,
        metadata=metadata,
        optimizer_result=result,
    )


def _gaussian_area_profile(wave: np.ndarray, flux: float, center: float, fwhm_kms: float) -> np.ndarray:
    sigma = (float(fwhm_kms) / C_KMS) * float(center) / FWHM_TO_SIGMA
    if sigma <= 0:
        return np.zeros_like(wave)
    return float(flux) * np.exp(-0.5 * ((wave - center) / sigma) ** 2) / (np.sqrt(2.0 * np.pi) * sigma)


def _gaussian_unit_profile_with_derivatives(
    wave: np.ndarray,
    rest_center: float,
    velocity_kms: float,
    fwhm_kms: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = float(rest_center) * np.exp(float(velocity_kms) / C_KMS)
    sigma = (float(fwhm_kms) / C_KMS) * center / FWHM_TO_SIGMA
    if sigma <= 0:
        zeros = np.zeros_like(wave, dtype=float)
        return zeros, zeros, zeros
    u = (wave - center) / sigma
    profile = np.exp(-0.5 * u * u) / (np.sqrt(2.0 * np.pi) * sigma)
    derivative_velocity = profile * (u * u - 1.0 + u * center / sigma) / C_KMS
    derivative_fwhm = profile * (u * u - 1.0) / float(fwhm_kms)
    return profile, derivative_velocity, derivative_fwhm


class _HbetaContext:
    def __init__(self, config: HbetaComplexConfig, include_wing: bool, flux_scale: float):
        self.config = config
        self.include_wing = include_wing
        self.names: List[str] = []
        self.initial: List[float] = []
        self.lower: List[float] = []
        self.upper: List[float] = []
        self._configure(max(float(flux_scale), 1.0e-6))

    def _add(self, name, value, lo, hi):
        self.names.append(name)
        self.initial.append(float(np.clip(value, lo, hi)))
        self.lower.append(float(lo))
        self.upper.append(float(hi))

    def _configure(self, scale):
        fractions = (0.55, 0.30, 0.15)
        for i, ((fwhm_lo, fwhm_hi), fraction) in enumerate(
            zip(self.config.broad_fwhm_bands_kms, fractions), start=1
        ):
            prefix = f"Hb_broad{i}"
            self._add(f"{prefix}.flux", scale * fraction, 0.0, np.inf)
            self._add(
                f"{prefix}.velocity_kms",
                0.0,
                self.config.broad_velocity_bounds_kms[0],
                self.config.broad_velocity_bounds_kms[1],
            )
            self._add(f"{prefix}.fwhm_kms", 0.5 * (fwhm_lo + fwhm_hi), fwhm_lo, fwhm_hi)
        self._add("Hb_narrow.flux", scale * 0.05, 0.0, np.inf)
        self._add(
            "narrow.velocity_kms",
            0.0,
            self.config.narrow_velocity_bounds_kms[0],
            self.config.narrow_velocity_bounds_kms[1],
        )
        self._add(
            "narrow.fwhm_kms",
            350.0,
            self.config.narrow_fwhm_bounds_kms[0],
            self.config.narrow_fwhm_bounds_kms[1],
        )
        self._add("OIII5007_core.flux", scale * 0.2, 0.0, np.inf)
        if self.include_wing:
            self._add("OIII5007_wing.flux", scale * 0.1, 0.0, np.inf)
            self._add("wing.velocity_kms", -250.0, -2000.0, 1000.0)
            self._add("wing.fwhm_kms", 900.0, 300.0, 3500.0)
        if self.config.heii_enabled:
            self._add("HeII_broad.flux", scale * 0.05, 0.0, np.inf)
            self._add("HeII_broad.velocity_kms", 0.0, -2000.0, 2000.0)
            self._add("HeII_broad.fwhm_kms", 3000.0, 900.0, 10000.0)
        self.initial = np.asarray(self.initial, dtype=float)
        self.lower = np.asarray(self.lower, dtype=float)
        self.upper = np.asarray(self.upper, dtype=float)
        self.index = {name: i for i, name in enumerate(self.names)}

    def get(self, theta, name):
        return float(theta[self.index[name]])

    @staticmethod
    def shifted(center, velocity):
        return center * np.exp(float(velocity) / C_KMS)

    def components(self, theta, wave):
        out = {}
        for i in range(1, 4):
            prefix = f"Hb_broad{i}"
            center = self.shifted(HBETA_WAVE, self.get(theta, f"{prefix}.velocity_kms"))
            out[prefix] = _gaussian_area_profile(
                wave, self.get(theta, f"{prefix}.flux"), center, self.get(theta, f"{prefix}.fwhm_kms")
            )
        narrow_v = self.get(theta, "narrow.velocity_kms")
        narrow_width = self.get(theta, "narrow.fwhm_kms")
        out["Hb_narrow"] = _gaussian_area_profile(
            wave,
            self.get(theta, "Hb_narrow.flux"),
            self.shifted(HBETA_WAVE, narrow_v),
            narrow_width,
        )
        core_flux = self.get(theta, "OIII5007_core.flux")
        out["OIII5007_core"] = _gaussian_area_profile(
            wave, core_flux, self.shifted(OIII_5007_WAVE, narrow_v), narrow_width
        )
        out["OIII4959_core"] = _gaussian_area_profile(
            wave,
            core_flux / self.config.oiii_ratio_5007_4959,
            self.shifted(OIII_4959_WAVE, narrow_v),
            narrow_width,
        )
        if self.include_wing:
            wing_flux = self.get(theta, "OIII5007_wing.flux")
            wing_v = self.get(theta, "wing.velocity_kms")
            wing_width = self.get(theta, "wing.fwhm_kms")
            out["OIII5007_wing"] = _gaussian_area_profile(
                wave, wing_flux, self.shifted(OIII_5007_WAVE, wing_v), wing_width
            )
            out["OIII4959_wing"] = _gaussian_area_profile(
                wave,
                wing_flux / self.config.oiii_ratio_5007_4959,
                self.shifted(OIII_4959_WAVE, wing_v),
                wing_width,
            )
        if self.config.heii_enabled:
            out["HeII_broad"] = _gaussian_area_profile(
                wave,
                self.get(theta, "HeII_broad.flux"),
                self.shifted(HEII_WAVE, self.get(theta, "HeII_broad.velocity_kms")),
                self.get(theta, "HeII_broad.fwhm_kms"),
            )
        return out

    def model(self, theta, wave):
        return sum(self.components(theta, wave).values(), np.zeros_like(wave))

    def broad_profile(self, theta, wave):
        components = self.components(theta, wave)
        return components["Hb_broad1"] + components["Hb_broad2"] + components["Hb_broad3"]

    @property
    def linear_names(self) -> List[str]:
        return [name for name in self.names if name.endswith(".flux")]

    @property
    def nonlinear_names(self) -> List[str]:
        linear = set(self.linear_names)
        return [name for name in self.names if name not in linear]

    def _named_values(self, names: Sequence[str], values: np.ndarray) -> Dict[str, float]:
        return {name: float(values[index]) for index, name in enumerate(names)}

    def separable_design(
        self,
        nonlinear: np.ndarray,
        wave: np.ndarray,
        need_derivatives: bool,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        nonlinear_values = self._named_values(self.nonlinear_names, nonlinear)
        columns: List[np.ndarray] = []
        derivative_columns = [
            [] for _ in self.nonlinear_names
        ] if need_derivatives else None

        def append_column(basis: np.ndarray, derivatives: Dict[str, np.ndarray]) -> None:
            columns.append(np.asarray(basis, dtype=float))
            if derivative_columns is None:
                return
            for index, name in enumerate(self.nonlinear_names):
                derivative_columns[index].append(
                    np.asarray(derivatives.get(name, np.zeros_like(wave)), dtype=float)
                )

        for index in range(1, 4):
            prefix = f"Hb_broad{index}"
            velocity_name = f"{prefix}.velocity_kms"
            width_name = f"{prefix}.fwhm_kms"
            basis, derivative_velocity, derivative_width = (
                _gaussian_unit_profile_with_derivatives(
                    wave,
                    HBETA_WAVE,
                    nonlinear_values[velocity_name],
                    nonlinear_values[width_name],
                )
            )
            append_column(
                basis,
                {
                    velocity_name: derivative_velocity,
                    width_name: derivative_width,
                },
            )

        narrow_velocity = nonlinear_values["narrow.velocity_kms"]
        narrow_width = nonlinear_values["narrow.fwhm_kms"]
        hb_basis, hb_velocity_derivative, hb_width_derivative = (
            _gaussian_unit_profile_with_derivatives(
                wave, HBETA_WAVE, narrow_velocity, narrow_width
            )
        )
        append_column(
            hb_basis,
            {
                "narrow.velocity_kms": hb_velocity_derivative,
                "narrow.fwhm_kms": hb_width_derivative,
            },
        )
        oiii5007, oiii5007_velocity, oiii5007_width = (
            _gaussian_unit_profile_with_derivatives(
                wave, OIII_5007_WAVE, narrow_velocity, narrow_width
            )
        )
        oiii4959, oiii4959_velocity, oiii4959_width = (
            _gaussian_unit_profile_with_derivatives(
                wave, OIII_4959_WAVE, narrow_velocity, narrow_width
            )
        )
        ratio = self.config.oiii_ratio_5007_4959
        append_column(
            oiii5007 + oiii4959 / ratio,
            {
                "narrow.velocity_kms": oiii5007_velocity + oiii4959_velocity / ratio,
                "narrow.fwhm_kms": oiii5007_width + oiii4959_width / ratio,
            },
        )

        if self.include_wing:
            wing_velocity = nonlinear_values["wing.velocity_kms"]
            wing_width = nonlinear_values["wing.fwhm_kms"]
            wing5007, wing5007_velocity, wing5007_width = (
                _gaussian_unit_profile_with_derivatives(
                    wave, OIII_5007_WAVE, wing_velocity, wing_width
                )
            )
            wing4959, wing4959_velocity, wing4959_width = (
                _gaussian_unit_profile_with_derivatives(
                    wave, OIII_4959_WAVE, wing_velocity, wing_width
                )
            )
            append_column(
                wing5007 + wing4959 / ratio,
                {
                    "wing.velocity_kms": wing5007_velocity + wing4959_velocity / ratio,
                    "wing.fwhm_kms": wing5007_width + wing4959_width / ratio,
                },
            )

        if self.config.heii_enabled:
            velocity_name = "HeII_broad.velocity_kms"
            width_name = "HeII_broad.fwhm_kms"
            basis, derivative_velocity, derivative_width = (
                _gaussian_unit_profile_with_derivatives(
                    wave,
                    HEII_WAVE,
                    nonlinear_values[velocity_name],
                    nonlinear_values[width_name],
                )
            )
            append_column(
                basis,
                {
                    velocity_name: derivative_velocity,
                    width_name: derivative_width,
                },
            )

        design = np.column_stack(columns)
        if derivative_columns is None:
            return design, None
        return design, tuple(np.column_stack(items) for items in derivative_columns)

    def assemble_full_parameters(
        self,
        linear: np.ndarray,
        nonlinear: np.ndarray,
    ) -> np.ndarray:
        values = np.empty(len(self.names), dtype=float)
        for name, value in zip(self.linear_names, linear):
            values[self.index[name]] = value
        for name, value in zip(self.nonlinear_names, nonlinear):
            values[self.index[name]] = value
        return values

    def separable_initial_and_bounds(self):
        linear_indices = np.array([self.index[name] for name in self.linear_names], dtype=int)
        nonlinear_indices = np.array([self.index[name] for name in self.nonlinear_names], dtype=int)
        return (
            self.initial[linear_indices],
            (self.lower[linear_indices], self.upper[linear_indices]),
            self.initial[nonlinear_indices],
            (self.lower[nonlinear_indices], self.upper[nonlinear_indices]),
        )


class _SeparableLineContext:
    def __init__(self):
        self.names: List[str] = []
        self.initial: List[float] = []
        self.lower: List[float] = []
        self.upper: List[float] = []

    def _add(self, name, value, lo, hi):
        self.names.append(name)
        self.initial.append(float(np.clip(value, lo, hi)))
        self.lower.append(float(lo))
        self.upper.append(float(hi))

    def _finalize(self):
        self.initial = np.asarray(self.initial, dtype=float)
        self.lower = np.asarray(self.lower, dtype=float)
        self.upper = np.asarray(self.upper, dtype=float)
        self.index = {name: i for i, name in enumerate(self.names)}

    def get(self, theta, name):
        return float(theta[self.index[name]])

    @staticmethod
    def shifted(center, velocity):
        return center * np.exp(float(velocity) / C_KMS)

    @property
    def linear_names(self) -> List[str]:
        return [name for name in self.names if name.endswith(".flux")]

    @property
    def nonlinear_names(self) -> List[str]:
        linear = set(self.linear_names)
        return [name for name in self.names if name not in linear]

    def _named_values(self, names: Sequence[str], values: np.ndarray) -> Dict[str, float]:
        return {name: float(values[index]) for index, name in enumerate(names)}

    def assemble_full_parameters(self, linear: np.ndarray, nonlinear: np.ndarray) -> np.ndarray:
        values = np.empty(len(self.names), dtype=float)
        for name, value in zip(self.linear_names, linear):
            values[self.index[name]] = value
        for name, value in zip(self.nonlinear_names, nonlinear):
            values[self.index[name]] = value
        return values

    def separable_initial_and_bounds(self):
        linear_indices = np.array([self.index[name] for name in self.linear_names], dtype=int)
        nonlinear_indices = np.array([self.index[name] for name in self.nonlinear_names], dtype=int)
        return (
            self.initial[linear_indices],
            (self.lower[linear_indices], self.upper[linear_indices]),
            self.initial[nonlinear_indices],
            (self.lower[nonlinear_indices], self.upper[nonlinear_indices]),
        )

    def model(self, theta, wave):
        return sum(self.components(theta, wave).values(), np.zeros_like(wave))


class _MgIIContext(_SeparableLineContext):
    def __init__(self, config: MgIIComplexConfig, flux_scale: float):
        super().__init__()
        self.config = config
        scale = max(float(flux_scale), 1.0e-6)
        for index, ((fwhm_lo, fwhm_hi), fraction) in enumerate(
            zip(config.broad_fwhm_bands_kms, (0.65, 0.35)), start=1
        ):
            prefix = f"MgII_broad{index}"
            self._add(f"{prefix}.flux", scale * fraction, 0.0, np.inf)
            self._add(
                f"{prefix}.velocity_kms",
                0.0,
                config.broad_velocity_bounds_kms[0],
                config.broad_velocity_bounds_kms[1],
            )
            self._add(f"{prefix}.fwhm_kms", 0.5 * (fwhm_lo + fwhm_hi), fwhm_lo, fwhm_hi)
        self._finalize()

    def components(self, theta, wave):
        out = {}
        for index in range(1, 3):
            prefix = f"MgII_broad{index}"
            out[prefix] = _gaussian_area_profile(
                wave,
                self.get(theta, f"{prefix}.flux"),
                self.shifted(MGII_WAVE, self.get(theta, f"{prefix}.velocity_kms")),
                self.get(theta, f"{prefix}.fwhm_kms"),
            )
        return out

    def broad_profile(self, theta, wave):
        components = self.components(theta, wave)
        return components["MgII_broad1"] + components["MgII_broad2"]

    def separable_design(self, nonlinear, wave, need_derivatives):
        values = self._named_values(self.nonlinear_names, nonlinear)
        columns = []
        derivative_columns = [[] for _ in self.nonlinear_names] if need_derivatives else None
        for index in range(1, 3):
            prefix = f"MgII_broad{index}"
            velocity_name = f"{prefix}.velocity_kms"
            width_name = f"{prefix}.fwhm_kms"
            basis, velocity_derivative, width_derivative = _gaussian_unit_profile_with_derivatives(
                wave, MGII_WAVE, values[velocity_name], values[width_name]
            )
            columns.append(basis)
            if derivative_columns is not None:
                derivatives = {
                    velocity_name: velocity_derivative,
                    width_name: width_derivative,
                }
                for derivative_index, name in enumerate(self.nonlinear_names):
                    derivative_columns[derivative_index].append(
                        derivatives.get(name, np.zeros_like(wave))
                    )
        design = np.column_stack(columns)
        if derivative_columns is None:
            return design, None
        return design, tuple(np.column_stack(items) for items in derivative_columns)


class _HalphaContext(_SeparableLineContext):
    def __init__(self, config: HalphaComplexConfig, flux_scale: float):
        super().__init__()
        self.config = config
        scale = max(float(flux_scale), 1.0e-6)
        for index, ((fwhm_lo, fwhm_hi), fraction) in enumerate(
            zip(config.broad_fwhm_bands_kms, (0.55, 0.30, 0.15)), start=1
        ):
            prefix = f"Ha_broad{index}"
            self._add(f"{prefix}.flux", scale * fraction, 0.0, np.inf)
            self._add(
                f"{prefix}.velocity_kms",
                0.0,
                config.broad_velocity_bounds_kms[0],
                config.broad_velocity_bounds_kms[1],
            )
            self._add(f"{prefix}.fwhm_kms", 0.5 * (fwhm_lo + fwhm_hi), fwhm_lo, fwhm_hi)
        self._add("Ha_narrow.flux", scale * 0.05, 0.0, np.inf)
        self._add(
            "narrow.velocity_kms",
            0.0,
            config.narrow_velocity_bounds_kms[0],
            config.narrow_velocity_bounds_kms[1],
        )
        self._add(
            "narrow.fwhm_kms",
            350.0,
            config.narrow_fwhm_bounds_kms[0],
            config.narrow_fwhm_bounds_kms[1],
        )
        self._add("NII6585.flux", scale * 0.05, 0.0, np.inf)
        self._add("SII6718.flux", scale * 0.03, 0.0, np.inf)
        self._add("SII6733.flux", scale * 0.03, 0.0, np.inf)
        self._finalize()

    def components(self, theta, wave):
        out = {}
        for index in range(1, 4):
            prefix = f"Ha_broad{index}"
            out[prefix] = _gaussian_area_profile(
                wave,
                self.get(theta, f"{prefix}.flux"),
                self.shifted(HALPHA_WAVE, self.get(theta, f"{prefix}.velocity_kms")),
                self.get(theta, f"{prefix}.fwhm_kms"),
            )
        narrow_velocity = self.get(theta, "narrow.velocity_kms")
        narrow_width = self.get(theta, "narrow.fwhm_kms")
        out["Ha_narrow"] = _gaussian_area_profile(
            wave,
            self.get(theta, "Ha_narrow.flux"),
            self.shifted(HALPHA_WAVE, narrow_velocity),
            narrow_width,
        )
        nii_flux = self.get(theta, "NII6585.flux")
        out["NII6585"] = _gaussian_area_profile(
            wave, nii_flux, self.shifted(NII_6585_WAVE, narrow_velocity), narrow_width
        )
        out["NII6549"] = _gaussian_area_profile(
            wave,
            nii_flux / self.config.nii_ratio_6585_6549,
            self.shifted(NII_6549_WAVE, narrow_velocity),
            narrow_width,
        )
        out["SII6718"] = _gaussian_area_profile(
            wave,
            self.get(theta, "SII6718.flux"),
            self.shifted(SII_6718_WAVE, narrow_velocity),
            narrow_width,
        )
        out["SII6733"] = _gaussian_area_profile(
            wave,
            self.get(theta, "SII6733.flux"),
            self.shifted(SII_6733_WAVE, narrow_velocity),
            narrow_width,
        )
        return out

    def broad_profile(self, theta, wave):
        components = self.components(theta, wave)
        return components["Ha_broad1"] + components["Ha_broad2"] + components["Ha_broad3"]

    def separable_design(self, nonlinear, wave, need_derivatives):
        values = self._named_values(self.nonlinear_names, nonlinear)
        columns = []
        derivative_columns = [[] for _ in self.nonlinear_names] if need_derivatives else None

        def append_column(basis, derivatives):
            columns.append(basis)
            if derivative_columns is not None:
                for derivative_index, name in enumerate(self.nonlinear_names):
                    derivative_columns[derivative_index].append(
                        derivatives.get(name, np.zeros_like(wave))
                    )

        for index in range(1, 4):
            prefix = f"Ha_broad{index}"
            velocity_name = f"{prefix}.velocity_kms"
            width_name = f"{prefix}.fwhm_kms"
            basis, velocity_derivative, width_derivative = _gaussian_unit_profile_with_derivatives(
                wave, HALPHA_WAVE, values[velocity_name], values[width_name]
            )
            append_column(
                basis,
                {velocity_name: velocity_derivative, width_name: width_derivative},
            )
        narrow_velocity = values["narrow.velocity_kms"]
        narrow_width = values["narrow.fwhm_kms"]
        ha, ha_velocity, ha_width = _gaussian_unit_profile_with_derivatives(
            wave, HALPHA_WAVE, narrow_velocity, narrow_width
        )
        append_column(
            ha,
            {"narrow.velocity_kms": ha_velocity, "narrow.fwhm_kms": ha_width},
        )
        nii6585, nii6585_velocity, nii6585_width = _gaussian_unit_profile_with_derivatives(
            wave, NII_6585_WAVE, narrow_velocity, narrow_width
        )
        nii6549, nii6549_velocity, nii6549_width = _gaussian_unit_profile_with_derivatives(
            wave, NII_6549_WAVE, narrow_velocity, narrow_width
        )
        ratio = self.config.nii_ratio_6585_6549
        append_column(
            nii6585 + nii6549 / ratio,
            {
                "narrow.velocity_kms": nii6585_velocity + nii6549_velocity / ratio,
                "narrow.fwhm_kms": nii6585_width + nii6549_width / ratio,
            },
        )
        for center in (SII_6718_WAVE, SII_6733_WAVE):
            basis, velocity_derivative, width_derivative = _gaussian_unit_profile_with_derivatives(
                wave, center, narrow_velocity, narrow_width
            )
            append_column(
                basis,
                {
                    "narrow.velocity_kms": velocity_derivative,
                    "narrow.fwhm_kms": width_derivative,
                },
            )
        design = np.column_stack(columns)
        if derivative_columns is None:
            return design, None
        return design, tuple(np.column_stack(items) for items in derivative_columns)


def _profile_fwhm(
    wave: np.ndarray,
    profile: np.ndarray,
    reference_wave: float = HBETA_WAVE,
) -> float:
    if wave.size < 3 or not np.any(profile > 0):
        return np.nan
    peak_index = int(np.argmax(profile))
    half = 0.5 * float(profile[peak_index])
    above = profile >= half
    left = peak_index
    right = peak_index
    while left > 0 and above[left - 1]:
        left -= 1
    while right < wave.size - 1 and above[right + 1]:
        right += 1
    if left == 0 or right == wave.size - 1:
        return np.nan
    left_wave = np.interp(half, [profile[left - 1], profile[left]], [wave[left - 1], wave[left]])
    right_wave = np.interp(
        half, [profile[right + 1], profile[right]], [wave[right + 1], wave[right]]
    )
    return float(abs(right_wave - left_wave) / float(reference_wave) * C_KMS)


def _hbeta_metrics(
    theta: np.ndarray,
    context: _HbetaContext,
    continuum_at_hbeta: float,
    z: float,
    flux_scale_to_cgs: Optional[float],
) -> Dict[str, float]:
    grid = np.linspace(4500.0, 5220.0, 7201)
    profile = context.broad_profile(theta, grid)
    area = float(np.trapezoid(profile, grid))
    centroid = float(np.trapezoid(grid * profile, grid) / area) if area > 0 else np.nan
    variance = float(np.trapezoid((grid - centroid) ** 2 * profile, grid) / area) if area > 0 else np.nan
    sigma_kms = np.sqrt(max(variance, 0.0)) / HBETA_WAVE * C_KMS if np.isfinite(variance) else np.nan
    physical = area * (1.0 + z) * flux_scale_to_cgs if flux_scale_to_cgs is not None else np.nan
    return {
        "Hb_broad_flux_input": area,
        "Hb_broad_flux_cgs": float(physical),
        "Hb_broad_centroid": centroid,
        "Hb_broad_velocity_kms": float(np.log(centroid / HBETA_WAVE) * C_KMS) if centroid > 0 else np.nan,
        "Hb_broad_sigma_kms": float(sigma_kms),
        "Hb_broad_fwhm_kms": _profile_fwhm(grid, profile),
        "Hb_broad_ew_rest": float(area / continuum_at_hbeta) if continuum_at_hbeta > 0 else np.nan,
    }


def _metric_errors(theta, covariance, metric_function):
    base = metric_function(theta)
    if covariance is None:
        return {name: np.nan for name in base}
    names = list(base)
    jac = np.zeros((len(names), theta.size), dtype=float)
    for j in range(theta.size):
        step = max(abs(theta[j]) * 1.0e-5, 1.0e-5)
        plus = theta.copy()
        minus = theta.copy()
        plus[j] += step
        minus[j] -= step
        pval = metric_function(plus)
        mval = metric_function(minus)
        for i, name in enumerate(names):
            if np.isfinite(pval[name]) and np.isfinite(mval[name]):
                jac[i, j] = (pval[name] - mval[name]) / (2.0 * step)
    metric_cov = jac @ covariance @ jac.T
    diag = np.sqrt(np.clip(np.diag(metric_cov), 0.0, np.inf))
    return {name: float(diag[i]) for i, name in enumerate(names)}


def _complex_coverage(
    spectrum: Spectrum,
    window: Tuple[float, float],
    line_centers: Sequence[float],
    min_coverage_fraction: float,
    min_valid_pixels: int,
    edge_margin_kms: float,
) -> Tuple[bool, Dict[str, Any]]:
    valid_wave = spectrum.wave_rest[spectrum.valid_mask]
    lo, hi = map(float, window)
    if valid_wave.size == 0:
        return False, {
            "coverage_fraction": 0.0,
            "n_valid_pixels": 0,
            "reason": "no_valid_pixels",
        }
    valid_min = float(valid_wave.min())
    valid_max = float(valid_wave.max())
    overlap = max(0.0, min(hi, valid_max) - max(lo, valid_min))
    coverage_fraction = overlap / (hi - lo)
    window_mask = spectrum.valid_mask & (spectrum.wave_rest >= lo) & (spectrum.wave_rest <= hi)
    n_valid_pixels = int(np.count_nonzero(window_mask))
    centers_covered = all(
        valid_min <= center * np.exp(-edge_margin_kms / C_KMS)
        and valid_max >= center * np.exp(edge_margin_kms / C_KMS)
        for center in line_centers
    )
    covered = (
        coverage_fraction >= min_coverage_fraction
        and n_valid_pixels >= min_valid_pixels
        and centers_covered
    )
    return covered, {
        "coverage_fraction": float(coverage_fraction),
        "n_valid_pixels": n_valid_pixels,
        "centers_covered_with_margin": bool(centers_covered),
        "valid_wave_min": valid_min,
        "valid_wave_max": valid_max,
        "window": (lo, hi),
    }


def _failed_complex_result(
    spectrum: Spectrum,
    continuum_result: GlobalContinuumResult,
    complex_name: str,
    window: Tuple[float, float],
    warning: NeoFitWarning,
    coverage: Dict[str, Any],
) -> EmissionComplexResult:
    wave = spectrum.wave_rest
    mask = spectrum.valid_mask & (wave >= window[0]) & (wave <= window[1])
    metadata = spectrum.metadata.to_dict()
    metadata.update({"complex_name": complex_name, "coverage": coverage})
    return EmissionComplexResult(
        success=False,
        status=-1,
        message=warning.message,
        selected_model="not_fit",
        param_values={},
        param_errors={},
        covariance=None,
        metrics={},
        metric_errors={},
        chi2=np.nan,
        dof=0,
        reduced_chi2=np.nan,
        bic=np.nan,
        wave_rest=wave.copy(),
        flux_continuum_subtracted=spectrum.flux - continuum_result.model,
        err=spectrum.err.copy(),
        model=np.zeros_like(wave),
        component_models={},
        fit_mask=mask,
        warnings=[warning],
        metadata=metadata,
    )


def _broad_complex_metrics(
    theta: np.ndarray,
    context,
    *,
    metric_prefix: str,
    reference_wave: float,
    grid: np.ndarray,
    continuum_at_line: float,
    z: float,
    flux_scale_to_cgs: Optional[float],
) -> Dict[str, float]:
    profile = context.broad_profile(theta, grid)
    area = float(np.trapezoid(profile, grid))
    centroid = float(np.trapezoid(grid * profile, grid) / area) if area > 0 else np.nan
    variance = (
        float(np.trapezoid((grid - centroid) ** 2 * profile, grid) / area)
        if area > 0
        else np.nan
    )
    sigma_kms = (
        np.sqrt(max(variance, 0.0)) / reference_wave * C_KMS
        if np.isfinite(variance)
        else np.nan
    )
    physical = (
        area * (1.0 + z) * flux_scale_to_cgs
        if flux_scale_to_cgs is not None
        else np.nan
    )
    metrics = {
        f"{metric_prefix}_broad_flux_input": area,
        f"{metric_prefix}_broad_flux_cgs": float(physical),
        f"{metric_prefix}_broad_centroid": centroid,
        f"{metric_prefix}_broad_velocity_kms": (
            float(np.log(centroid / reference_wave) * C_KMS) if centroid > 0 else np.nan
        ),
        f"{metric_prefix}_broad_sigma_kms": float(sigma_kms),
        f"{metric_prefix}_broad_fwhm_kms": _profile_fwhm(
            grid, profile, reference_wave=reference_wave
        ),
        f"{metric_prefix}_broad_ew_rest": (
            float(area / continuum_at_line) if continuum_at_line > 0 else np.nan
        ),
    }
    if metric_prefix == "Ha":
        ratio = context.config.nii_ratio_6585_6549
        for name in ("Ha_narrow", "NII6585", "SII6718", "SII6733"):
            value = context.get(theta, f"{name}.flux")
            metrics[f"{name}_flux_input"] = value
            metrics[f"{name}_flux_cgs"] = (
                value * (1.0 + z) * flux_scale_to_cgs
                if flux_scale_to_cgs is not None
                else np.nan
            )
        nii6549 = context.get(theta, "NII6585.flux") / ratio
        metrics["NII6549_flux_input"] = nii6549
        metrics["NII6549_flux_cgs"] = (
            nii6549 * (1.0 + z) * flux_scale_to_cgs
            if flux_scale_to_cgs is not None
            else np.nan
        )
    return metrics


def _fit_separable_emission_complex(
    spectrum: Spectrum,
    continuum_result: GlobalContinuumResult,
    *,
    config,
    context_class,
    complex_name: str,
    selected_model: str,
    reference_wave: float,
    line_centers: Sequence[float],
    metric_prefix: str,
    metric_grid: Tuple[float, float],
    compute_covariance: bool,
) -> EmissionComplexResult:
    covered, coverage = _complex_coverage(
        spectrum,
        config.window,
        line_centers,
        config.min_coverage_fraction,
        config.min_valid_pixels,
        config.edge_margin_kms,
    )
    if not covered:
        warning = NeoFitWarning(
            code="line_complex_not_covered",
            message=f"{complex_name} was skipped because its fitting window is not covered.",
            severity="info",
            context={"complex": complex_name, **coverage},
        )
        return _failed_complex_result(
            spectrum, continuum_result, complex_name, config.window, warning, coverage
        )

    wave = spectrum.wave_rest
    mask = (
        spectrum.valid_mask
        & (wave >= config.window[0])
        & (wave <= config.window[1])
    )
    line_flux = spectrum.flux - continuum_result.model
    positive = np.clip(line_flux[mask], 0.0, np.inf)
    flux_scale = float(np.trapezoid(positive, wave[mask]))
    context = context_class(config, flux_scale)
    result, optimizer_used, fallback_reason = _solve_once_with_fallback(
        context,
        wave[mask],
        line_flux[mask],
        spectrum.err[mask],
        context.initial,
        config,
    )
    residual = (line_flux[mask] - context.model(result.x, wave[mask])) / spectrum.err[mask]
    chi2 = float(np.sum(residual**2))
    dof = max(int(np.count_nonzero(mask) - result.x.size), 0)
    reduced = float(chi2 / dof) if dof else np.nan
    bic = float(chi2 + result.x.size * np.log(np.count_nonzero(mask)))
    if compute_covariance:
        covariance, errors, warnings = _covariance_from_jacobian(
            result.jac, reduced, context.names
        )
    else:
        covariance = None
        errors = {name: np.nan for name in context.names}
        warnings = []
    if fallback_reason is not None:
        warnings.append(
            NeoFitWarning(
                code="optimizer_fallback_legacy",
                message="Variable projection failed; the legacy joint optimizer was used.",
                context={"reason": fallback_reason},
            )
        )
    warnings.extend(_active_bound_warnings(result, context.names))
    if compute_covariance:
        warnings.append(
            NeoFitWarning(
                code="statistical_uncertainty_excludes_continuum_host",
                message="Covariance errors condition on the fitted host and continuum models.",
                severity="info",
            )
        )
    continuum_at_line = float(
        np.interp(reference_wave, continuum_result.wave_rest, continuum_result.model)
    )
    grid = np.linspace(metric_grid[0], metric_grid[1], 7201)
    metric_function = lambda theta: _broad_complex_metrics(
        theta,
        context,
        metric_prefix=metric_prefix,
        reference_wave=reference_wave,
        grid=grid,
        continuum_at_line=continuum_at_line,
        z=spectrum.z,
        flux_scale_to_cgs=spectrum.flux_density_scale_to_cgs,
    )
    metrics = metric_function(result.x)
    metric_errors = _metric_errors(result.x, covariance, metric_function)
    metadata = spectrum.metadata.to_dict()
    metadata.update(
        {
            "complex_name": complex_name,
            "coverage": coverage,
            "continuum_at_line": continuum_at_line,
            "line_flux_cgs_conversion": "(1+z)*flux_density_scale_to_cgs",
            "optimizer_requested": config.optimizer_method,
            "optimizer_used": optimizer_used,
            "jacobian_method": (
                config.jacobian_method if optimizer_used == "variable_projection" else "2-point"
            ),
            "optimizer_fallback": fallback_reason is not None,
            "n_linear_parameters": len(context.linear_names),
            "n_nonlinear_parameters": len(context.nonlinear_names),
            "nonlinear_nfev": int(getattr(result, "nfev", 0) or 0),
            "nonlinear_njev": int(getattr(result, "njev", 0) or 0),
            "linear_solve_count": int(getattr(result, "linear_solve_count", 0) or 0),
        }
    )
    return EmissionComplexResult(
        success=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        selected_model=selected_model,
        param_values={name: float(result.x[index]) for index, name in enumerate(context.names)},
        param_errors=errors,
        covariance=covariance,
        metrics=metrics,
        metric_errors=metric_errors,
        chi2=chi2,
        dof=dof,
        reduced_chi2=reduced,
        bic=bic,
        wave_rest=wave.copy(),
        flux_continuum_subtracted=line_flux,
        err=spectrum.err.copy(),
        model=context.model(result.x, wave),
        component_models=context.components(result.x, wave),
        fit_mask=mask,
        warnings=warnings,
        metadata=metadata,
        optimizer_result=result,
    )


def fit_mgii_complex(
    spectrum: Spectrum,
    continuum_result: GlobalContinuumResult,
    config: Optional[MgIIComplexConfig] = None,
    *,
    compute_covariance: bool = True,
) -> EmissionComplexResult:
    """Fit the two-component broad Mg II profile when covered."""

    cfg = config or MgIIComplexConfig()
    return _fit_separable_emission_complex(
        spectrum,
        continuum_result,
        config=cfg,
        context_class=_MgIIContext,
        complex_name="MgII",
        selected_model="two_broad_gaussians",
        reference_wave=MGII_WAVE,
        line_centers=(MGII_WAVE,),
        metric_prefix="MgII",
        metric_grid=(2550.0, 3050.0),
        compute_covariance=compute_covariance,
    )


def fit_halpha_complex(
    spectrum: Spectrum,
    continuum_result: GlobalContinuumResult,
    config: Optional[HalphaComplexConfig] = None,
    *,
    compute_covariance: bool = True,
) -> EmissionComplexResult:
    """Fit broad H-alpha plus tied narrow H-alpha/[N II]/[S II]."""

    cfg = config or HalphaComplexConfig()
    result = _fit_separable_emission_complex(
        spectrum,
        continuum_result,
        config=cfg,
        context_class=_HalphaContext,
        complex_name="Halpha_NII_SII",
        selected_model="three_broad_plus_tied_narrow",
        reference_wave=HALPHA_WAVE,
        line_centers=(
            HALPHA_WAVE,
            NII_6549_WAVE,
            NII_6585_WAVE,
            SII_6718_WAVE,
            SII_6733_WAVE,
        ),
        metric_prefix="Ha",
        metric_grid=(6200.0, 6900.0),
        compute_covariance=compute_covariance,
    )
    result.metadata["nii_ratio_6585_6549"] = cfg.nii_ratio_6585_6549
    return result


def _fit_hbeta_candidate(
    spectrum: Spectrum,
    continuum_result: GlobalContinuumResult,
    config: HbetaComplexConfig,
    include_wing: bool,
    compute_covariance: bool,
) -> HbetaComplexResult:
    wave = spectrum.wave_rest
    lo, hi = config.window
    mask = spectrum.valid_mask & (wave >= lo) & (wave <= hi)
    if not config.heii_enabled:
        mask &= ~_window_mask(wave, (config.heii_mask,))
    if np.count_nonzero(mask) < 30:
        warning = NeoFitWarning(
            code="window_too_few_pixels",
            message="Too few valid pixels cover the H-beta/[O III] complex.",
            severity="error",
        )
        return HbetaComplexResult(
            False, -1, warning.message, "wing" if include_wing else "core", {}, {}, None, {}, {},
            np.nan, 0, np.nan, np.nan, wave.copy(), spectrum.flux - continuum_result.model,
            spectrum.err.copy(), np.zeros_like(wave), {}, mask, [warning], spectrum.metadata.to_dict()
        )
    line_flux = spectrum.flux - continuum_result.model
    positive = np.clip(line_flux[mask], 0.0, np.inf)
    flux_scale = float(np.trapezoid(positive, wave[mask]))
    context = _HbetaContext(config, include_wing, flux_scale)
    result, optimizer_used, fallback_reason = _solve_once_with_fallback(
        context,
        wave[mask],
        line_flux[mask],
        spectrum.err[mask],
        context.initial,
        config,
    )
    residual = (line_flux[mask] - context.model(result.x, wave[mask])) / spectrum.err[mask]
    chi2 = float(np.sum(residual**2))
    dof = max(int(np.count_nonzero(mask) - result.x.size), 0)
    reduced = float(chi2 / dof) if dof else np.nan
    bic = float(chi2 + result.x.size * np.log(np.count_nonzero(mask)))
    if compute_covariance:
        covariance, errors, warnings = _covariance_from_jacobian(result.jac, reduced, context.names)
    else:
        covariance = None
        errors = {name: np.nan for name in context.names}
        warnings = []
    if fallback_reason is not None:
        warnings.append(
            NeoFitWarning(
                code="optimizer_fallback_legacy",
                message="Variable projection failed; the legacy joint optimizer was used.",
                context={"reason": fallback_reason},
            )
        )
    warnings.extend(_active_bound_warnings(result, context.names))
    if compute_covariance:
        warnings.append(
            NeoFitWarning(
                code="statistical_uncertainty_excludes_continuum_host",
                message="Covariance errors condition on the fitted host and continuum models.",
                severity="info",
            )
        )
    continuum_at_hbeta = float(np.interp(HBETA_WAVE, continuum_result.wave_rest, continuum_result.model))
    metric_function = lambda theta: _hbeta_metrics(
        theta, context, continuum_at_hbeta, spectrum.z, spectrum.flux_density_scale_to_cgs
    )
    metrics = metric_function(result.x)
    metric_errors = _metric_errors(result.x, covariance, metric_function)
    metadata = spectrum.metadata.to_dict()
    metadata.update(
        {
            "oiii_ratio_5007_4959": config.oiii_ratio_5007_4959,
            "continuum_at_hbeta": continuum_at_hbeta,
            "line_flux_cgs_conversion": "(1+z)*flux_density_scale_to_cgs",
            "optimizer_requested": config.optimizer_method,
            "optimizer_used": optimizer_used,
            "jacobian_method": (
                config.jacobian_method if optimizer_used == "variable_projection" else "2-point"
            ),
            "optimizer_fallback": fallback_reason is not None,
            "n_linear_parameters": len(context.linear_names),
            "n_nonlinear_parameters": len(context.nonlinear_names),
            "nonlinear_nfev": int(getattr(result, "nfev", 0) or 0),
            "nonlinear_njev": int(getattr(result, "njev", 0) or 0),
            "linear_solve_count": int(getattr(result, "linear_solve_count", 0) or 0),
        }
    )
    return HbetaComplexResult(
        success=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        selected_model="wing" if include_wing else "core",
        param_values={name: float(result.x[i]) for i, name in enumerate(context.names)},
        param_errors=errors,
        covariance=covariance,
        metrics=metrics,
        metric_errors=metric_errors,
        chi2=chi2,
        dof=dof,
        reduced_chi2=reduced,
        bic=bic,
        wave_rest=wave.copy(),
        flux_continuum_subtracted=line_flux,
        err=spectrum.err.copy(),
        model=context.model(result.x, wave),
        component_models=context.components(result.x, wave),
        fit_mask=mask,
        warnings=warnings,
        metadata=metadata,
        optimizer_result=result,
    )


def fit_hbeta_complex(
    spectrum: Spectrum,
    continuum_result: GlobalContinuumResult,
    config: Optional[HbetaComplexConfig] = None,
    *,
    compute_covariance: bool = True,
) -> HbetaComplexResult:
    """Fit core-only and optional wing H-beta/[O III] candidates."""

    cfg = config or HbetaComplexConfig()
    candidate_covariance = compute_covariance or cfg.fit_oiii_wings
    core = _fit_hbeta_candidate(
        spectrum, continuum_result, cfg, include_wing=False, compute_covariance=candidate_covariance
    )
    if not cfg.fit_oiii_wings or not core.success:
        if not compute_covariance:
            core.covariance = None
            core.param_errors = {name: np.nan for name in core.param_values}
            core.metric_errors = {name: np.nan for name in core.metrics}
        return core
    wing = _fit_hbeta_candidate(
        spectrum, continuum_result, cfg, include_wing=True, compute_covariance=candidate_covariance
    )
    wing_flux = wing.param_values.get("OIII5007_wing.flux", 0.0)
    wing_error = wing.param_errors.get("OIII5007_wing.flux", np.nan)
    wing_snr = wing_flux / wing_error if np.isfinite(wing_error) and wing_error > 0 else 0.0
    wing_width = wing.param_values.get("wing.fwhm_kms", 0.0)
    core_width = wing.param_values.get("narrow.fwhm_kms", np.inf)
    bic_improvement = core.bic - wing.bic
    accepted = (
        wing.success
        and bic_improvement >= cfg.wing_bic_delta
        and wing_snr >= cfg.wing_min_snr
        and wing_width > core_width
    )
    selected = wing if accepted else core
    selected.metadata["wing_candidate"] = {
        "accepted": bool(accepted),
        "bic_improvement": float(bic_improvement),
        "wing_snr": float(wing_snr),
        "wing_fwhm_kms": float(wing_width),
        "core_fwhm_kms": float(core_width),
    }
    if not accepted:
        selected.warnings.append(
            NeoFitWarning(
                code="oiii_wing_rejected",
                message="The [O III] wing candidate did not pass the BIC, S/N, and width criteria.",
                severity="info",
                context=selected.metadata["wing_candidate"],
            )
        )
    if not compute_covariance:
        selected.covariance = None
        selected.param_errors = {name: np.nan for name in selected.param_values}
        selected.metric_errors = {name: np.nan for name in selected.metrics}
    return selected


def _continuum_sample(
    spectrum: Spectrum,
    continuum: GlobalContinuumResult,
    wavelength: float,
    host_model: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    valid_wave = spectrum.wave_rest[spectrum.valid_mask]
    if valid_wave.size == 0 or wavelength < valid_wave.min() or wavelength > valid_wave.max():
        return {}
    power_law = continuum.component_models.get("power_law", np.zeros_like(continuum.model))
    values = {
        f"f_powerlaw_{int(wavelength)}": float(np.interp(wavelength, continuum.wave_rest, power_law)),
        f"fAGN_{int(wavelength)}": float(np.interp(wavelength, continuum.wave_rest, continuum.model)),
    }
    if host_model is not None:
        host = float(np.interp(wavelength, spectrum.wave_rest, host_model))
        agn = values[f"fAGN_{int(wavelength)}"]
        values[f"fHost_{int(wavelength)}"] = host
        values[f"fracHost_{int(wavelength)}"] = host / (host + agn) if host + agn > 0 else np.nan
    return values


def _optional_complex_for_workflow(
    fitter,
    spectrum,
    continuum,
    config,
    uncertainty,
    complex_name,
):
    try:
        result = fitter(
            spectrum,
            continuum,
            config,
            compute_covariance=uncertainty.covariance,
        )
    except Exception as exc:
        covered, coverage = _complex_coverage(
            spectrum,
            config.window,
            (
                (MGII_WAVE,)
                if complex_name == "mgii"
                else (
                    HALPHA_WAVE,
                    NII_6549_WAVE,
                    NII_6585_WAVE,
                    SII_6718_WAVE,
                    SII_6733_WAVE,
                )
            ),
            config.min_coverage_fraction,
            config.min_valid_pixels,
            config.edge_margin_kms,
        )
        if not covered:
            warning = NeoFitWarning(
                code="line_complex_not_covered",
                message=f"{complex_name} was skipped because its fitting window is not covered.",
                severity="info",
                context={"complex": complex_name, **coverage},
            )
        else:
            warning = NeoFitWarning(
                code="optional_line_fit_failed",
                message=f"{complex_name} fitting failed: {exc}",
                context={"complex": complex_name},
            )
        result = _failed_complex_result(
            spectrum,
            continuum,
            complex_name,
            config.window,
            warning,
            coverage,
        )
    if "line_complex_not_covered" in result.warning_codes():
        return None, result.warnings
    warnings = []
    if not result.success:
        warnings.append(
            NeoFitWarning(
                code="optional_line_fit_failed",
                message=f"{complex_name} fitting did not converge successfully.",
                context={"complex": complex_name, "message": result.message},
            )
        )
    return result, warnings


def fit_global_lines(
    spectrum: Spectrum,
    global_config: Optional[GlobalContinuumConfig] = None,
    hbeta_config: Optional[HbetaComplexConfig] = None,
    mgii_config: Optional[MgIIComplexConfig] = None,
    halpha_config: Optional[HalphaComplexConfig] = None,
    uncertainty_config: Optional[UncertaintyConfig] = None,
    *,
    host_model_on_grid: Optional[np.ndarray] = None,
    complexes: Optional[Sequence[Union[str, ComplexRecipe]]] = None,
) -> NeoFitWorkflowResult:
    """Fit the global continuum and adaptively selected emission recipes."""

    global_cfg = global_config or GlobalContinuumConfig()
    hbeta_cfg = hbeta_config or HbetaComplexConfig()
    mgii_cfg = mgii_config or MgIIComplexConfig()
    halpha_cfg = halpha_config or HalphaComplexConfig()
    uncertainty_cfg = uncertainty_config or UncertaintyConfig()
    continuum_initial = fit_global_continuum(
        spectrum, global_cfg, compute_covariance=uncertainty_cfg.covariance
    )
    requested_recipes = _resolve_requested_recipes(complexes)
    selected_recipes: List[ComplexRecipe] = []
    complex_statuses: Dict[str, str] = {}
    coverage_by_recipe = {}
    for recipe in requested_recipes:
        coverage = resolve_recipe_coverage(spectrum, recipe)
        coverage_by_recipe[recipe.id] = coverage
        if coverage.status == "not_covered":
            complex_statuses[recipe.id] = "not_covered"
            continue
        selected_recipes.append(recipe)
        complex_statuses[recipe.id] = coverage.status

    warnings: List[NeoFitWarning] = []
    hbeta_recipe = next(
        (recipe for recipe in selected_recipes if recipe.id == "hbeta_oiii"), None
    )
    hbeta_was_requested = any(
        recipe.id == "hbeta_oiii" for recipe in requested_recipes
    )
    hbeta_initial = None
    if hbeta_recipe is not None:
        try:
            hbeta_initial = fit_hbeta_complex(
                spectrum,
                continuum_initial,
                hbeta_cfg,
                compute_covariance=uncertainty_cfg.covariance,
            )
        except Exception as exc:
            coverage = coverage_by_recipe[hbeta_recipe.id]
            warning = NeoFitWarning(
                code="optional_line_fit_failed",
                message=f"Hβ fitting failed: {exc}",
                context={"recipe": hbeta_recipe.id},
            )
            hbeta_initial = _failed_complex_result(
                spectrum,
                continuum_initial,
                hbeta_recipe.id,
                hbeta_recipe.fit_window,
                warning,
                {
                    "coverage_fraction": coverage.coverage_fraction,
                    "n_valid_pixels": coverage.n_valid_pixels,
                    "status": coverage.status,
                },
            )
        complex_statuses[hbeta_recipe.id] = (
            "fit" if hbeta_initial.success else "failed"
        )
        hbeta_initial.metadata.update(
            {
                "recipe_id": hbeta_recipe.id,
                "recipe_label": hbeta_recipe.label,
                "recipe_backend": hbeta_recipe.backend,
                "coverage_status": coverage_by_recipe[hbeta_recipe.id].status,
            }
        )

    bs_config = global_cfg.balmer_series
    balmer_available = continuum_initial.metadata.get("balmer_template") is not None
    initial_width = continuum_initial.metadata.get("balmer_series_fwhm_kms", np.nan)
    width_source = (
        "disabled"
        if not bs_config.enabled
        else "fixed_config"
        if bs_config.fixed_fwhm_kms is not None or not bs_config.fit_fwhm
        else "free_global_fit"
        if balmer_available and np.isfinite(initial_width)
        else "failed_or_unconstrained"
    )
    warning_codes: List[str] = []
    sync_policy = bs_config.sync_with_hbeta
    width_is_free = bs_config.fit_fwhm and bs_config.fixed_fwhm_kms is None
    sync_requested = (
        sync_policy in ("auto", "require")
        and balmer_available
        and width_is_free
        and hbeta_was_requested
    )
    hbeta_reliable, hbeta_snr, reliability_reason = _hbeta_sync_reliability(
        hbeta_initial,
        uncertainty_cfg,
        bs_config.sync_min_fwhm_snr,
    )
    refinement_iterations = 0
    width_difference = np.nan
    width_converged = False
    sync_attempted = False
    continuum = continuum_initial
    hbeta = hbeta_initial
    continuum_width_snr = np.nan
    if balmer_available and width_is_free:
        width_index = list(continuum_initial.param_values).index(
            "balmer_series.fwhm_kms"
        )
        active_mask = np.asarray(
            getattr(
                continuum_initial.optimizer_result,
                "active_mask",
                np.zeros(len(continuum_initial.param_values)),
            ),
            dtype=int,
        )
        free_width = continuum_initial.param_values.get(
            "balmer_series.fwhm_kms", np.nan
        )
        free_width_error = continuum_initial.param_errors.get(
            "balmer_series.fwhm_kms", np.nan
        )
        if np.isfinite(free_width_error) and free_width_error > 0:
            continuum_width_snr = float(free_width / free_width_error)
        if not np.isfinite(continuum_width_snr) or continuum_width_snr < 3.0:
            _append_workflow_warning(
                warnings,
                warning_codes,
                "balmer_series_fwhm_weakly_constrained",
                "The freely fitted Balmer-series width is weakly constrained.",
                {"fwhm_snr": continuum_width_snr},
                severity="info",
            )
        if active_mask[width_index] != 0:
            width_source = "failed_or_unconstrained"
            _append_workflow_warning(
                warnings,
                warning_codes,
                "balmer_series_fwhm_at_bound",
                "The freely fitted Balmer-series width is active on an optimizer bound.",
                {"bound_side": int(active_mask[width_index])},
            )
    if sync_requested and hbeta_reliable and hbeta_initial is not None:
        sync_attempted = True
        measured_width = hbeta_initial.metrics.get("Hb_broad_fwhm_kms", np.nan)
        continuum = continuum_initial
        hbeta = hbeta_initial
        series_width = float(measured_width)
        for iteration in range(max(int(global_cfg.balmer_width_sync_max_iterations), 1)):
            refined_series = replace(
                global_cfg.balmer_series,
                fit_fwhm=False,
                fwhm_kms=series_width,
                fixed_fwhm_kms=None,
            )
            refined_config = replace(global_cfg, balmer_series=refined_series)
            candidate_continuum = fit_global_continuum(
                spectrum, refined_config, compute_covariance=uncertainty_cfg.covariance
            )
            candidate_hbeta = fit_hbeta_complex(
                spectrum,
                candidate_continuum,
                hbeta_cfg,
                compute_covariance=uncertainty_cfg.covariance,
            )
            refinement_iterations = iteration + 1
            fitted_width = candidate_hbeta.metrics.get("Hb_broad_fwhm_kms", np.nan)
            if (
                not candidate_continuum.success
                or not candidate_hbeta.success
                or not np.isfinite(fitted_width)
                or fitted_width <= 0
            ):
                _append_workflow_warning(
                    warnings,
                    warning_codes,
                    "hbeta_sync_failed",
                    "Hβ synchronization failed; the initial free-width solution was restored.",
                )
                break
            width_difference = float(fitted_width - series_width)
            if abs(width_difference) <= float(global_cfg.balmer_width_sync_tolerance_kms):
                continuum = candidate_continuum
                hbeta = candidate_hbeta
                width_converged = True
                width_source = "synced_to_hbeta"
                break
            series_width = float(fitted_width)
        if not width_converged:
            continuum = continuum_initial
            hbeta = hbeta_initial
            width_source = (
                "fixed_config"
                if bs_config.fixed_fwhm_kms is not None or not bs_config.fit_fwhm
                else "free_global_fit"
            )
            if "hbeta_sync_failed" not in warning_codes:
                _append_workflow_warning(
                    warnings,
                    warning_codes,
                    "hbeta_sync_not_converged",
                    "Balmer-series and Hβ widths did not converge; the initial solution was restored.",
                    {
                        "difference_kms": width_difference,
                        "tolerance_kms": global_cfg.balmer_width_sync_tolerance_kms,
                        "iterations": refinement_iterations,
                    },
                )
    else:
        if bs_config.enabled and not balmer_available:
            _append_workflow_warning(
                warnings,
                warning_codes,
                "balmer_series_region_not_covered",
                "The high-order Balmer-series region is not sufficiently covered.",
            )
        elif (
            sync_policy in ("auto", "require")
            and hbeta_was_requested
            and hbeta_recipe is None
        ):
            _append_workflow_warning(
                warnings,
                warning_codes,
                "hbeta_sync_skipped_not_covered",
                "Hβ synchronization was skipped because its recipe is not covered.",
                severity="info",
            )
            if bs_config.fit_fwhm and bs_config.fixed_fwhm_kms is None:
                _append_workflow_warning(
                    warnings,
                    warning_codes,
                    "balmer_series_fwhm_free_no_hbeta_anchor",
                    "The Balmer-series width remains the freely fitted continuum value.",
                    severity="info",
                )
        elif sync_policy in ("auto", "require") and not hbeta_reliable:
            _append_workflow_warning(
                warnings,
                warning_codes,
                "hbeta_sync_skipped_unreliable",
                "Hβ synchronization was skipped because its width is unreliable.",
                {"reason": reliability_reason, "fwhm_snr": hbeta_snr},
            )
        if sync_policy == "require" and not width_converged:
            _append_workflow_warning(
                warnings,
                warning_codes,
                "hbeta_sync_required_unmet",
                "Required Hβ synchronization was unavailable; continuing with the free width.",
            )

    line_complexes: Dict[str, EmissionComplexResult] = {}
    if hbeta is not None:
        hbeta.metadata.update(
            {
                "recipe_id": "hbeta_oiii",
                "recipe_label": complex_recipes.get("hbeta_oiii").label,
                "recipe_backend": "hbeta_adapter",
            }
        )
        line_complexes["hbeta_oiii"] = hbeta
        complex_statuses["hbeta_oiii"] = "fit" if hbeta.success else "failed"
    for recipe in selected_recipes:
        if recipe.id == "hbeta_oiii":
            continue
        fit = _fit_selected_recipe(
            recipe,
            spectrum,
            continuum,
            uncertainty_cfg,
            mgii_cfg,
            halpha_cfg,
        )
        if fit is None:
            complex_statuses[recipe.id] = "not_covered"
            continue
        line_complexes[recipe.id] = fit
        complex_statuses[recipe.id] = "fit" if fit.success else "failed"
        warnings.extend(
            warning for warning in fit.warnings
            if warning.code in ("optional_line_fit_failed", "recipe_backend_not_implemented")
        )
    mgii = line_complexes.get("mgii")
    halpha = line_complexes.get("halpha_nii_sii")

    samples = {}
    for wavelength in (3000.0, 5100.0):
        samples.update(_continuum_sample(spectrum, continuum, wavelength, host_model_on_grid))
    final_width = continuum.metadata.get("balmer_series_fwhm_kms", np.nan)
    metadata = {
        "refinement_performed": continuum is not continuum_initial,
        "balmer_series_fwhm_kms": float(final_width),
        "balmer_series_fwhm_source": width_source,
        "balmer_series_fwhm_synced_to_hbeta": bool(width_converged),
        "balmer_series_fwhm_warning_codes": tuple(warning_codes),
        "balmer_series_fwhm_snr": float(hbeta_snr),
        "balmer_series_free_fwhm_snr": float(continuum_width_snr),
        "hbeta_sync_requested": bool(sync_requested),
        "hbeta_sync_attempted": bool(sync_attempted),
        "hbeta_sync_converged": bool(width_converged),
        "hbeta_sync_iterations": int(refinement_iterations),
        "hbeta_sync_difference_kms": float(width_difference),
        "balmer_width_sync_tolerance_kms": float(global_cfg.balmer_width_sync_tolerance_kms),
        "continuum_samples": samples,
        "continuum_sample_flux_density_unit": spectrum.flux_density_unit,
        "uncertainty_mode": "covariance",
        "complex_statuses": dict(complex_statuses),
        "line_complex_status": dict(complex_statuses),
        "requested_complex_recipes": tuple(recipe.id for recipe in requested_recipes),
        "selected_complex_recipes": tuple(recipe.id for recipe in selected_recipes),
    }
    continuum.metadata.update(
        {
            key: metadata[key]
            for key in (
                "balmer_series_fwhm_kms",
                "balmer_series_fwhm_source",
                "balmer_series_fwhm_synced_to_hbeta",
                "balmer_series_fwhm_warning_codes",
                "balmer_series_fwhm_snr",
                "hbeta_sync_requested",
                "hbeta_sync_attempted",
                "hbeta_sync_converged",
                "hbeta_sync_iterations",
            )
        }
    )
    workflow = NeoFitWorkflowResult(
        spectrum=spectrum,
        total_spectrum=spectrum,
        continuum_initial=continuum_initial,
        continuum=continuum,
        hbeta_initial=hbeta_initial,
        hbeta=hbeta,
        mgii=mgii,
        halpha=halpha,
        line_complexes=line_complexes,
        complex_statuses=complex_statuses,
        warnings=warnings,
        metadata=metadata,
    )
    if uncertainty_cfg.monte_carlo_trials > 0:
        workflow.monte_carlo = _run_workflow_mc(
            spectrum,
            global_cfg,
            hbeta_cfg,
            mgii_cfg,
            halpha_cfg,
            int(uncertainty_cfg.monte_carlo_trials),
            uncertainty_cfg.random_seed,
            requested_recipes,
        )
        workflow.metadata["uncertainty_mode"] = "covariance+monte_carlo"
    return workflow


def _append_workflow_warning(
    warnings: List[NeoFitWarning],
    codes: List[str],
    code: str,
    message: str,
    context: Optional[Dict[str, Any]] = None,
    *,
    severity: str = "warning",
) -> None:
    if code in codes:
        return
    codes.append(code)
    warnings.append(
        NeoFitWarning(code=code, message=message, severity=severity, context=context or {})
    )


def _resolve_requested_recipes(
    requested: Optional[Sequence[Union[str, ComplexRecipe]]],
) -> List[ComplexRecipe]:
    candidates = (
        [recipe for recipe in complex_recipes.list_complexes() if recipe.auto_enabled]
        if requested is None
        else [
            complex_recipes.get(item) if isinstance(item, str) else item
            for item in requested
        ]
    )
    if any(not isinstance(recipe, ComplexRecipe) for recipe in candidates):
        raise TypeError("complexes entries must be recipe IDs or ComplexRecipe objects.")
    selected: List[ComplexRecipe] = []
    groups: Dict[str, ComplexRecipe] = {}
    for recipe in candidates:
        group = recipe.exclusive_group
        if group is None:
            selected.append(recipe)
            continue
        previous = groups.get(group)
        if previous is None:
            groups[group] = recipe
            selected.append(recipe)
            continue
        if requested is not None:
            raise ValueError(
                f"overlapping_complex_recipes: {previous.id!r} and {recipe.id!r}"
            )
        if recipe.priority > previous.priority:
            selected.remove(previous)
            groups[group] = recipe
            selected.append(recipe)
    return selected


def _hbeta_sync_reliability(
    fit: Optional[EmissionComplexResult],
    uncertainty: UncertaintyConfig,
    minimum_snr: Optional[float],
) -> Tuple[bool, float, str]:
    if fit is None:
        return False, np.nan, "not_covered"
    if not fit.success:
        return False, np.nan, "fit_failed"
    width = fit.metrics.get("Hb_broad_fwhm_kms", np.nan)
    error = fit.metric_errors.get("Hb_broad_fwhm_kms", np.nan)
    if not np.isfinite(width) or width <= 0:
        return False, np.nan, "nonfinite_width"
    if minimum_snr is not None:
        if not uncertainty.covariance:
            return False, np.nan, "covariance_disabled"
        if not np.isfinite(error) or error <= 0:
            return False, np.nan, "nonfinite_width_uncertainty"
        snr = float(width / error)
        if snr < minimum_snr:
            return False, snr, "low_width_snr"
    else:
        snr = float(width / error) if np.isfinite(error) and error > 0 else np.nan
    active = np.asarray(
        getattr(fit.optimizer_result, "active_mask", np.zeros(len(fit.param_values))),
        dtype=int,
    )
    names = list(fit.param_values)
    if any(
        active[index] != 0
        for index, name in enumerate(names)
        if name.startswith("Hb_broad") and name.endswith(".fwhm_kms")
    ):
        return False, snr, "broad_width_at_bound"
    return True, snr, "reliable"


def _fit_selected_recipe(
    recipe: ComplexRecipe,
    spectrum: Spectrum,
    continuum: GlobalContinuumResult,
    uncertainty: UncertaintyConfig,
    mgii_config: MgIIComplexConfig,
    halpha_config: HalphaComplexConfig,
) -> Optional[EmissionComplexResult]:
    try:
        if recipe.backend == "mgii_adapter":
            result = fit_mgii_complex(
                spectrum, continuum, mgii_config, compute_covariance=uncertainty.covariance
            )
        elif recipe.backend == "halpha_adapter":
            result = fit_halpha_complex(
                spectrum, continuum, halpha_config, compute_covariance=uncertainty.covariance
            )
        elif recipe.backend == "generic":
            result = fit_generic_complex(
                spectrum, continuum, recipe, compute_covariance=uncertainty.covariance
            )
        else:
            warning = NeoFitWarning(
                code="recipe_backend_not_implemented",
                message=f"The backend for {recipe.id} is not implemented.",
                context={"recipe": recipe.id, "backend": recipe.backend},
            )
            return _failed_complex_result(
                spectrum, continuum, recipe.id, recipe.fit_window, warning, {}
            )
    except Exception as exc:
        coverage = resolve_recipe_coverage(spectrum, recipe)
        warning = NeoFitWarning(
            code="optional_line_fit_failed",
            message=f"{recipe.id} fitting failed: {exc}",
            context={"recipe": recipe.id},
        )
        return _failed_complex_result(
            spectrum,
            continuum,
            recipe.id,
            recipe.fit_window,
            warning,
            {
                "coverage_fraction": coverage.coverage_fraction,
                "n_valid_pixels": coverage.n_valid_pixels,
                "status": coverage.status,
            },
        )
    if result is not None:
        result.metadata.setdefault("recipe_id", recipe.id)
        result.metadata.setdefault("recipe_label", recipe.label)
        result.metadata.setdefault("recipe_backend", recipe.backend)
    return result


def fit_global_hbeta(
    spectrum: Spectrum,
    global_config: Optional[GlobalContinuumConfig] = None,
    hbeta_config: Optional[HbetaComplexConfig] = None,
    uncertainty_config: Optional[UncertaintyConfig] = None,
    *,
    host_model_on_grid: Optional[np.ndarray] = None,
) -> NeoFitWorkflowResult:
    """Compatibility wrapper for :func:`fit_global_lines`."""

    result = fit_global_lines(
        spectrum,
        global_config,
        hbeta_config,
        None,
        None,
        uncertainty_config,
        host_model_on_grid=host_model_on_grid,
        complexes=("hbeta_oiii",),
    )
    result.metadata["compatibility_hbeta_mode"] = True
    return result


def _run_workflow_mc(
    spectrum,
    global_config,
    hbeta_config,
    mgii_config,
    halpha_config,
    n_trials,
    seed,
    recipes,
):
    rng = np.random.default_rng(seed)
    samples: Dict[str, List[float]] = {}
    continuum_successes = 0
    complex_successes: Dict[str, int] = {recipe.id: 0 for recipe in recipes}
    for _ in range(n_trials):
        noisy = Spectrum.from_arrays(
            spectrum.wave_obs,
            spectrum.flux + rng.normal(0.0, spectrum.err),
            err=spectrum.err,
            z=spectrum.z,
            mask=spectrum.mask,
            metadata=spectrum.metadata,
        )
        try:
            result = fit_global_lines(
                noisy,
                global_config,
                hbeta_config,
                mgii_config,
                halpha_config,
                UncertaintyConfig(covariance=True, monte_carlo_trials=0),
                complexes=recipes,
            )
            values = {}
            if result.continuum_success:
                continuum_successes += 1
                values.update(result.continuum.param_values)
            for recipe_key, complex_result in result.line_complexes.items():
                recipe_id = complex_result.metadata.get("recipe_id", recipe_key)
                if complex_result.success:
                    if recipe_id in complex_successes:
                        complex_successes[recipe_id] += 1
                    values.update(complex_result.metrics)
            for name, value in values.items():
                if np.isfinite(value):
                    samples.setdefault(name, []).append(float(value))
        except Exception:
            continue
    percentiles = {}
    for name, values in samples.items():
        if values:
            p16, p50, p84 = np.percentile(values, [16.0, 50.0, 84.0])
            percentiles[name] = {"p16": float(p16), "p50": float(p50), "p84": float(p84)}
    return {
        "n_requested": int(n_trials),
        "continuum_success_count": int(continuum_successes),
        "complex_success_counts": complex_successes,
        "percentiles": percentiles,
    }
