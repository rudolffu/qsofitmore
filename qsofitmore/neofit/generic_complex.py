"""Recipe compiler and generic bounded variable-projection line fitter."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import lines
from .complex_recipes import ComponentRecipe, ComplexRecipe
from .global_result import EmissionComplexResult, GlobalContinuumResult
from .spectrum import Spectrum
from .warnings import NeoFitWarning

C_KMS = 299792.458
FWHM_TO_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


@dataclass(frozen=True)
class RecipeCoverage:
    status: str
    recipe: ComplexRecipe
    active_component_ids: Tuple[str, ...]
    disabled_component_ids: Tuple[str, ...]
    missing_required_line_ids: Tuple[str, ...]
    coverage_fraction: float
    n_valid_pixels: int
    fit_windows: Tuple[Tuple[float, float], ...]
    warnings: Tuple[NeoFitWarning, ...]

    @property
    def covered(self) -> bool:
        return self.status in ("covered", "partially_covered")


def _center_covered(center: float, valid_min: float, valid_max: float, margin: float) -> bool:
    return (
        valid_min <= center * np.exp(-margin / C_KMS)
        and valid_max >= center * np.exp(margin / C_KMS)
    )


def resolve_recipe_coverage(spectrum: Spectrum, recipe: ComplexRecipe) -> RecipeCoverage:
    """Resolve component-adaptive coverage before constructing a model."""

    valid_wave = spectrum.wave_rest[spectrum.valid_mask]
    if valid_wave.size == 0:
        warning = NeoFitWarning(
            code="complex_not_covered",
            message=f"{recipe.id} has no valid wavelength coverage.",
            severity="info",
            context={"recipe": recipe.id},
        )
        return RecipeCoverage(
            "not_covered", recipe, (), tuple(item.id for item in recipe.components),
            recipe.required_line_ids, 0.0, 0, (), (warning,)
        )
    valid_min, valid_max = float(valid_wave.min()), float(valid_wave.max())
    lo, hi = recipe.fit_window
    overlap = max(0.0, min(hi, valid_max) - max(lo, valid_min))
    coverage_fraction = overlap / (hi - lo) if hi > lo else 0.0
    fit_windows = tuple(
        window
        for window in (recipe.fit_windows or (recipe.fit_window,))
        if max(window[0], valid_min) < min(window[1], valid_max)
    )
    fit_mask = np.zeros_like(spectrum.valid_mask)
    for window in fit_windows:
        fit_mask |= (spectrum.wave_rest >= window[0]) & (spectrum.wave_rest <= window[1])
    fit_mask &= spectrum.valid_mask
    for window in recipe.mask_windows:
        fit_mask &= ~((spectrum.wave_rest >= window[0]) & (spectrum.wave_rest <= window[1]))
    n_valid_pixels = int(np.count_nonzero(fit_mask))

    covered_lines = {
        line_id
        for component in recipe.components
        for line_id in component.line_ids
        if _center_covered(
            lines.get(line_id).vacuum_wavelength,
            valid_min,
            valid_max,
            recipe.edge_margin_kms,
        )
    }
    covered_lines.update(
        line_id
        for line_id in recipe.required_line_ids
        if _center_covered(
            lines.get(line_id).vacuum_wavelength,
            valid_min,
            valid_max,
            recipe.edge_margin_kms,
        )
    )
    missing_required = tuple(
        line_id for line_id in recipe.required_line_ids if line_id not in covered_lines
    )
    if recipe.coverage_mode == "full":
        active = tuple(item.id for item in recipe.components if item.enabled)
        disabled = tuple(item.id for item in recipe.components if not item.enabled)
        covered = (
            coverage_fraction >= recipe.min_coverage_fraction
            and n_valid_pixels >= recipe.min_valid_pixels
            and not missing_required
        )
        status = "covered" if covered else (
            "missing_required" if missing_required and overlap > 0 else "not_covered"
        )
    else:
        active = tuple(
            item.id
            for item in recipe.components
            if item.enabled and all(line_id in covered_lines for line_id in item.line_ids)
        )
        active_set = set(active)
        active = tuple(
            item.id
            for item in recipe.components
            if item.id in active_set
            and (
                item.fixed_ratio_to is None
                or item.fixed_ratio_to in active_set
            )
        )
        disabled = tuple(item.id for item in recipe.components if item.id not in active)
        if missing_required:
            status = "missing_required"
        elif active and n_valid_pixels >= recipe.min_valid_pixels:
            status = "partially_covered" if disabled else "covered"
        else:
            status = "not_covered"

    warnings: List[NeoFitWarning] = []
    if status == "not_covered":
        warnings.append(
            NeoFitWarning(
                code="complex_not_covered",
                message=f"{recipe.id} is outside usable wavelength coverage.",
                severity="info",
                context={"recipe": recipe.id},
            )
        )
    elif status == "missing_required":
        warnings.append(
            NeoFitWarning(
                code="required_line_not_covered",
                message=f"{recipe.id} is missing one or more required lines.",
                context={"recipe": recipe.id, "line_ids": missing_required},
            )
        )
    if disabled:
        warnings.append(
            NeoFitWarning(
                code="complex_partially_covered",
                message=f"{recipe.id} will fit only covered component groups.",
                severity="info",
                context={"recipe": recipe.id, "disabled_components": disabled},
            )
        )
        for component_id in disabled:
            warnings.append(
                NeoFitWarning(
                    code="recipe_component_disabled_by_coverage",
                    message=f"{component_id} was disabled by wavelength coverage.",
                    severity="info",
                    context={"recipe": recipe.id, "component": component_id},
                )
            )
    return RecipeCoverage(
        status, recipe, active, disabled, missing_required, float(coverage_fraction),
        n_valid_pixels, fit_windows, tuple(warnings)
    )


def _profile(
    wave: np.ndarray,
    rest_center: float,
    velocity: float,
    fwhm: float,
    profile: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = float(rest_center) * np.exp(float(velocity) / C_KMS)
    if profile == "gaussian":
        sigma = float(fwhm) * center / C_KMS / FWHM_TO_SIGMA
        u = (wave - center) / sigma
        basis = np.exp(-0.5 * u * u) / (np.sqrt(2.0 * np.pi) * sigma)
        d_velocity = basis * (u * u - 1.0 + u * center / sigma) / C_KMS
        d_fwhm = basis * (u * u - 1.0) / float(fwhm)
        return basis, d_velocity, d_fwhm
    gamma = float(fwhm) * center / C_KMS / 2.0
    delta = wave - center
    denominator = delta * delta + gamma * gamma
    basis = gamma / (np.pi * denominator)
    dlog_dcenter = 1.0 / center + (
        2.0 * delta - 2.0 * gamma * gamma / center
    ) / denominator
    d_velocity = basis * dlog_dcenter * center / C_KMS
    d_fwhm = basis * (1.0 - 2.0 * gamma * gamma / denominator) / float(fwhm)
    return basis, d_velocity, d_fwhm


def _numerical_profile_fwhm(
    wave: np.ndarray,
    profile: np.ndarray,
    reference_wave: float,
) -> float:
    if wave.size < 3 or not np.any(profile > 0):
        return np.nan
    peak = int(np.argmax(profile))
    half = 0.5 * float(profile[peak])
    above = profile >= half
    left = peak
    right = peak
    while left > 0 and above[left - 1]:
        left -= 1
    while right < wave.size - 1 and above[right + 1]:
        right += 1
    if left == 0 or right == wave.size - 1:
        return np.nan
    left_wave = np.interp(
        half,
        [profile[left - 1], profile[left]],
        [wave[left - 1], wave[left]],
    )
    right_wave = np.interp(
        half,
        [profile[right + 1], profile[right]],
        [wave[right + 1], wave[right]],
    )
    return float(abs(right_wave - left_wave) / reference_wave * C_KMS)


class GenericComplexContext:
    """Compiled separable model for one resolved recipe."""

    def __init__(self, recipe: ComplexRecipe, component_ids: Sequence[str], flux_scale: float):
        self.recipe = recipe
        selected = set(component_ids)
        self.components_config = tuple(
            item for item in recipe.components if item.enabled and item.id in selected
        )
        self.names: List[str] = []
        self.initial: List[float] = []
        self.lower: List[float] = []
        self.upper: List[float] = []
        self.instances: List[Tuple[str, ComponentRecipe, Tuple[str, ...], str]] = []
        scale = max(float(flux_scale), 1.0e-8)

        for component in self.components_config:
            bands = component.fwhm_bands_kms
            for index in range(component.multiplicity):
                suffix = f"{index + 1}" if component.multiplicity > 1 else ""
                instance_id = f"{component.id}{suffix}"
                group = component.kinematic_group or instance_id
                if component.multiplicity > 1:
                    group = f"{group}{index + 1}"
                flux_name = (
                    f"{component.fixed_ratio_to}.flux"
                    if component.fixed_ratio_to is not None
                    else f"{instance_id}.flux"
                )
                self.instances.append((instance_id, component, component.line_ids, group))
                if component.fixed_ratio_to is None and flux_name not in self.names:
                    lo = -np.inf if component.flux_bounds[0] is None else component.flux_bounds[0]
                    hi = np.inf if component.flux_bounds[1] is None else component.flux_bounds[1]
                    self._add(flux_name, scale / max(len(self.components_config), 1), lo, hi)
                velocity_name = f"{group}.velocity_kms"
                width_name = f"{group}.fwhm_kms"
                if velocity_name not in self.names:
                    self._add(
                        velocity_name, 0.0,
                        component.velocity_bounds_kms[0], component.velocity_bounds_kms[1],
                    )
                if width_name not in self.names:
                    band = bands[min(index, len(bands) - 1)]
                    self._add(width_name, 0.5 * (band[0] + band[1]), band[0], band[1])
        if recipe.continuum_mode in ("constant", "linear"):
            self._add("continuum.constant", 0.0, -np.inf, np.inf)
        if recipe.continuum_mode == "linear":
            self._add("continuum.slope", 0.0, -np.inf, np.inf)
        self.initial = np.asarray(self.initial, dtype=float)
        self.lower = np.asarray(self.lower, dtype=float)
        self.upper = np.asarray(self.upper, dtype=float)
        self.index = {name: index for index, name in enumerate(self.names)}
        self.pivot = 0.5 * sum(recipe.fit_window)

    def _add(self, name, value, lower, upper):
        self.names.append(name)
        self.initial.append(float(np.clip(value, lower, upper)))
        self.lower.append(float(lower))
        self.upper.append(float(upper))

    @property
    def linear_names(self):
        return [
            name for name in self.names
            if name.endswith(".flux") or name.startswith("continuum.")
        ]

    @property
    def nonlinear_names(self):
        linear = set(self.linear_names)
        return [name for name in self.names if name not in linear]

    def separable_initial_and_bounds(self):
        li = np.asarray([self.index[name] for name in self.linear_names], dtype=int)
        ni = np.asarray([self.index[name] for name in self.nonlinear_names], dtype=int)
        return (
            self.initial[li], (self.lower[li], self.upper[li]),
            self.initial[ni], (self.lower[ni], self.upper[ni]),
        )

    def assemble_full_parameters(self, linear, nonlinear):
        values = np.empty(len(self.names), dtype=float)
        for name, value in zip(self.linear_names, linear):
            values[self.index[name]] = value
        for name, value in zip(self.nonlinear_names, nonlinear):
            values[self.index[name]] = value
        return values

    def _value(self, theta, name):
        return float(theta[self.index[name]])

    def _instance_basis(self, instance, nonlinear_values, wave):
        _, component, line_ids, group = instance
        velocity_name = f"{group}.velocity_kms"
        width_name = f"{group}.fwhm_kms"
        basis = np.zeros_like(wave)
        d_velocity = np.zeros_like(wave)
        d_width = np.zeros_like(wave)
        for line_id in line_ids:
            line = lines.get(line_id)
            values = _profile(
                wave, line.vacuum_wavelength,
                nonlinear_values[velocity_name], nonlinear_values[width_name],
                component.profile,
            )
            basis += values[0]
            d_velocity += values[1]
            d_width += values[2]
        ratio = component.fixed_ratio if component.fixed_ratio_to is not None else 1.0
        return basis / ratio, d_velocity / ratio, d_width / ratio

    def separable_design(self, nonlinear, wave, need_derivatives):
        nonlinear_values = dict(zip(self.nonlinear_names, map(float, nonlinear)))
        columns = []
        derivative_columns = [[] for _ in self.nonlinear_names] if need_derivatives else None
        by_flux: Dict[str, List[Tuple[Any, ...]]] = {}
        for instance in self.instances:
            instance_id, component, _, _ = instance
            flux_name = (
                f"{component.fixed_ratio_to}.flux"
                if component.fixed_ratio_to is not None
                else f"{instance_id}.flux"
            )
            by_flux.setdefault(flux_name, []).append(instance)
        for linear_name in self.linear_names:
            if linear_name == "continuum.constant":
                basis = np.ones_like(wave)
                derivatives = {}
            elif linear_name == "continuum.slope":
                basis = wave - self.pivot
                derivatives = {}
            else:
                basis = np.zeros_like(wave)
                derivatives: Dict[str, np.ndarray] = {}
                for instance in by_flux.get(linear_name, ()):
                    values = self._instance_basis(instance, nonlinear_values, wave)
                    basis += values[0]
                    group = instance[3]
                    velocity_name = f"{group}.velocity_kms"
                    width_name = f"{group}.fwhm_kms"
                    derivatives[velocity_name] = derivatives.get(
                        velocity_name, np.zeros_like(wave)
                    ) + values[1]
                    derivatives[width_name] = derivatives.get(
                        width_name, np.zeros_like(wave)
                    ) + values[2]
            columns.append(basis)
            if derivative_columns is not None:
                for index, name in enumerate(self.nonlinear_names):
                    derivative_columns[index].append(
                        derivatives.get(name, np.zeros_like(wave))
                    )
        design = np.column_stack(columns)
        if derivative_columns is None:
            return design, None
        return design, tuple(np.column_stack(items) for items in derivative_columns)

    def components(self, theta, wave):
        out: Dict[str, np.ndarray] = {}
        nonlinear_values = {
            name: self._value(theta, name) for name in self.nonlinear_names
        }
        for instance in self.instances:
            instance_id, component, _, _ = instance
            flux_name = (
                f"{component.fixed_ratio_to}.flux"
                if component.fixed_ratio_to is not None
                else f"{instance_id}.flux"
            )
            basis, _, _ = self._instance_basis(instance, nonlinear_values, wave)
            # _instance_basis already applies the ratio.
            out[instance_id] = self._value(theta, flux_name) * basis
        if "continuum.constant" in self.index:
            out["local_continuum_constant"] = np.full_like(
                wave, self._value(theta, "continuum.constant")
            )
        if "continuum.slope" in self.index:
            out["local_continuum_slope"] = (
                self._value(theta, "continuum.slope") * (wave - self.pivot)
            )
        return out

    def model(self, theta, wave):
        return sum(self.components(theta, wave).values(), np.zeros_like(wave))


def _failed_result(
    spectrum: Spectrum,
    continuum: GlobalContinuumResult,
    recipe: ComplexRecipe,
    coverage: RecipeCoverage,
) -> EmissionComplexResult:
    mask = np.zeros_like(spectrum.valid_mask)
    for window in coverage.fit_windows:
        mask |= (spectrum.wave_rest >= window[0]) & (spectrum.wave_rest <= window[1])
    mask &= spectrum.valid_mask
    warning = coverage.warnings[0] if coverage.warnings else NeoFitWarning(
        code="complex_not_covered", message=f"{recipe.id} is not covered."
    )
    return EmissionComplexResult(
        False, -1, warning.message, "not_fit", {}, {}, None, {}, {},
        np.nan, 0, np.nan, np.nan, spectrum.wave_rest.copy(),
        spectrum.flux - continuum.model, spectrum.err.copy(), np.zeros_like(spectrum.flux),
        {}, mask, list(coverage.warnings), {
            **spectrum.metadata.to_dict(), "recipe_id": recipe.id,
            "coverage_status": coverage.status,
        }
    )


def fit_generic_complex(
    spectrum: Spectrum,
    continuum: GlobalContinuumResult,
    recipe: ComplexRecipe,
    *,
    compute_covariance: bool = True,
) -> Optional[EmissionComplexResult]:
    """Fit one generic recipe; return ``None`` only when it is not covered."""

    from .global_fit import (
        _active_bound_warnings,
        _covariance_from_jacobian,
        _metric_errors,
        _solve_once_with_fallback,
    )

    coverage = resolve_recipe_coverage(spectrum, recipe)
    if coverage.status == "not_covered":
        return None
    if coverage.status == "missing_required":
        return _failed_result(spectrum, continuum, recipe, coverage)
    mask = np.zeros_like(spectrum.valid_mask)
    for window in coverage.fit_windows:
        mask |= (spectrum.wave_rest >= window[0]) & (spectrum.wave_rest <= window[1])
    for window in recipe.mask_windows:
        mask &= ~((spectrum.wave_rest >= window[0]) & (spectrum.wave_rest <= window[1]))
    mask &= spectrum.valid_mask
    line_flux = spectrum.flux - continuum.model
    if recipe.continuum_mode != "fixed_global":
        fit_flux = spectrum.flux if recipe.continuum_mode != "absent" else line_flux
    else:
        fit_flux = line_flux
    positive = np.clip(line_flux[mask], 0.0, np.inf)
    scale = float(np.trapezoid(positive, spectrum.wave_rest[mask]))
    context = GenericComplexContext(recipe, coverage.active_component_ids, scale)
    optimizer_config = SimpleNamespace(
        optimizer_method="auto", jacobian_method="semi_analytic", max_nfev=1500
    )
    result, optimizer_used, fallback_reason = _solve_once_with_fallback(
        context,
        spectrum.wave_rest[mask],
        fit_flux[mask],
        spectrum.err[mask],
        context.initial,
        optimizer_config,
    )
    residual = (
        fit_flux[mask] - context.model(result.x, spectrum.wave_rest[mask])
    ) / spectrum.err[mask]
    chi2 = float(np.sum(residual**2))
    dof = max(int(np.count_nonzero(mask) - result.x.size), 0)
    reduced = chi2 / dof if dof else np.nan
    bic = chi2 + result.x.size * np.log(max(np.count_nonzero(mask), 1))
    if compute_covariance:
        covariance, errors, fit_warnings = _covariance_from_jacobian(
            result.jac, reduced, context.names
        )
    else:
        covariance = None
        errors = {name: np.nan for name in context.names}
        fit_warnings = []
    selection_metadata: Dict[str, Any] = {}
    effective_component_ids = list(coverage.active_component_ids)
    candidate_components = [
        component
        for component in context.components_config
        if component.selection_rule is not None
        and component.selection_rule != "specialized_outflow_deferred"
    ]
    if candidate_components:
        candidate_ids = {component.id for component in candidate_components}
        reduced_ids = [
            component_id
            for component_id in effective_component_ids
            if component_id not in candidate_ids
        ]
        if reduced_ids:
            reduced_context = GenericComplexContext(recipe, reduced_ids, scale)
            reduced_result, _, _ = _solve_once_with_fallback(
                reduced_context,
                spectrum.wave_rest[mask],
                fit_flux[mask],
                spectrum.err[mask],
                reduced_context.initial,
                optimizer_config,
            )
            reduced_residual = (
                fit_flux[mask]
                - reduced_context.model(reduced_result.x, spectrum.wave_rest[mask])
            ) / spectrum.err[mask]
            reduced_chi2_value = float(np.sum(reduced_residual**2))
            reduced_bic_value = reduced_chi2_value + reduced_result.x.size * np.log(
                max(np.count_nonzero(mask), 1)
            )
            candidate_snrs = []
            for component in candidate_components:
                for instance_id, instance_component, _, _ in context.instances:
                    if instance_component.id != component.id:
                        continue
                    flux_name = (
                        f"{component.fixed_ratio_to}.flux"
                        if component.fixed_ratio_to is not None
                        else f"{instance_id}.flux"
                    )
                    flux_value = float(result.x[context.index[flux_name]])
                    flux_error = errors.get(flux_name, np.nan)
                    if np.isfinite(flux_error) and flux_error > 0:
                        candidate_snrs.append(flux_value / flux_error)
            maximum_snr = max(candidate_snrs, default=0.0)
            bic_improvement = float(reduced_bic_value - bic)
            accepted = bic_improvement >= 10.0 and maximum_snr >= 3.0
            selection_metadata = {
                "candidate_components": tuple(sorted(candidate_ids)),
                "accepted": bool(accepted),
                "bic_improvement": bic_improvement,
                "maximum_flux_snr": float(maximum_snr),
            }
            if not accepted:
                context = reduced_context
                result = reduced_result
                chi2 = reduced_chi2_value
                dof = max(int(np.count_nonzero(mask) - result.x.size), 0)
                reduced = chi2 / dof if dof else np.nan
                bic = float(reduced_bic_value)
                effective_component_ids = reduced_ids
                if compute_covariance:
                    covariance, errors, fit_warnings = _covariance_from_jacobian(
                        result.jac, reduced, context.names
                    )
                else:
                    covariance = None
                    errors = {name: np.nan for name in context.names}
                    fit_warnings = []
                fit_warnings.append(
                    NeoFitWarning(
                        code="recipe_optional_component_rejected",
                        message="Optional recipe components failed the BIC/S/N rule.",
                        severity="info",
                        context=selection_metadata,
                    )
                )
    fit_warnings = list(coverage.warnings) + fit_warnings
    fit_warnings.extend(_active_bound_warnings(result, context.names))
    active_components = [
        component
        for component in recipe.components
        if component.id in coverage.active_component_ids
    ]
    if any(
        component.selection_rule == "specialized_outflow_deferred"
        for component in active_components
    ):
        fit_warnings.append(
            NeoFitWarning(
                code="recipe_backend_not_implemented",
                message="Specialized outflow selection is not implemented; the enabled component was fitted generically.",
                context={"recipe": recipe.id},
            )
        )
    if any(component.fixed_ratio_to is not None for component in active_components):
        fit_warnings.append(
            NeoFitWarning(
                code="recipe_fixed_ratio_applied",
                message="One or more recipe flux ratios were held fixed.",
                severity="info",
                context={
                    "components": tuple(
                        component.id
                        for component in active_components
                        if component.fixed_ratio_to is not None
                    )
                },
            )
        )
    blend_line_ids = tuple(
        line_id
        for component in active_components
        for line_id in component.line_ids
        if lines.get(line_id).transition_type == "blend"
    )
    if blend_line_ids:
        fit_warnings.append(
            NeoFitWarning(
                code="recipe_blend_warning",
                message="Blend measurements depend on the configured decomposition.",
                severity="info",
                context={"line_ids": blend_line_ids},
            )
        )
    if fallback_reason:
        fit_warnings.append(
            NeoFitWarning(
                code="optimizer_fallback_legacy",
                message="Variable projection failed; the legacy joint optimizer was used.",
                context={"reason": fallback_reason},
            )
        )
    if recipe.id == "paschen_nir" and {
        "HeI10833_broad", "Pagamma_broad"
    }.issubset(coverage.active_component_ids):
        fit_warnings.append(
            NeoFitWarning(
                code="nir_he10833_pgamma_blend",
                message="He I 10833 and Paγ are a decomposition-dependent blend.",
                severity="info",
            )
        )

    def metrics(theta):
        values: Dict[str, float] = {}
        grouped: Dict[
            Tuple[str, str],
            List[Tuple[float, float, float, str]],
        ] = {}
        for instance_id, component, line_ids, group in context.instances:
            flux_name = (
                f"{component.fixed_ratio_to}.flux"
                if component.fixed_ratio_to is not None
                else f"{instance_id}.flux"
            )
            ratio = component.fixed_ratio if component.fixed_ratio_to is not None else 1.0
            flux = context._value(theta, flux_name) / ratio
            velocity = context._value(theta, f"{group}.velocity_kms")
            width = context._value(theta, f"{group}.fwhm_kms")
            for feature in line_ids:
                grouped.setdefault((feature, component.role), []).append(
                    (flux, velocity, width, component.profile)
                )

        for (feature, role), entries in grouped.items():
            definition = lines.get(feature)
            reference_wave = definition.vacuum_wavelength
            maximum_width = max(entry[2] for entry in entries)
            half_span = max(
                50.0,
                5.0 * maximum_width * reference_wave / C_KMS,
            )
            grid = np.linspace(
                reference_wave - half_span,
                reference_wave + half_span,
                2401,
            )
            profile = np.zeros_like(grid)
            for flux, velocity, width, profile_name in entries:
                basis, _, _ = _profile(
                    grid,
                    reference_wave,
                    velocity,
                    width,
                    profile_name,
                )
                profile += flux * basis
            integrated_flux = float(np.trapezoid(profile, grid))
            centroid = (
                float(np.trapezoid(grid * profile, grid) / integrated_flux)
                if integrated_flux > 0
                else np.nan
            )
            variance = (
                float(
                    np.trapezoid(
                        (grid - centroid) ** 2 * profile,
                        grid,
                    )
                    / integrated_flux
                )
                if integrated_flux > 0
                else np.nan
            )
            sigma_kms = (
                np.sqrt(max(variance, 0.0)) / reference_wave * C_KMS
                if np.isfinite(variance)
                else np.nan
            )
            fwhm_kms = _numerical_profile_fwhm(
                grid, profile, reference_wave
            )
            continuum_at_line = float(
                np.interp(centroid, continuum.wave_rest, continuum.model)
            )
            prefix = f"{feature}_{role}"
            values[f"{prefix}_flux_input"] = integrated_flux
            values[f"{prefix}_flux_cgs"] = (
                integrated_flux
                * (1.0 + spectrum.z)
                * spectrum.flux_density_scale_to_cgs
                if spectrum.flux_density_scale_to_cgs is not None else np.nan
            )
            values[f"{prefix}_fwhm_kms"] = fwhm_kms
            values[f"{prefix}_sigma_kms"] = float(sigma_kms)
            values[f"{prefix}_centroid"] = centroid
            values[f"{prefix}_ew_rest"] = (
                integrated_flux / continuum_at_line
                if continuum_at_line > 0 else np.nan
            )
        return values

    metric_values = metrics(result.x)
    metric_errors = _metric_errors(result.x, covariance, metrics)
    metadata = spectrum.metadata.to_dict()
    metadata.update({
        "recipe_id": recipe.id,
        "recipe_label": recipe.label,
        "recipe_backend": recipe.backend,
        "coverage_status": coverage.status,
        "coverage_fraction": coverage.coverage_fraction,
        "n_valid_pixels": coverage.n_valid_pixels,
        "active_components": tuple(effective_component_ids),
        "disabled_components": coverage.disabled_component_ids,
        "fit_windows": coverage.fit_windows,
        "optimizer_requested": "auto",
        "optimizer_used": optimizer_used,
        "jacobian_method": "semi_analytic" if optimizer_used == "variable_projection" else "2-point",
        "optimizer_fallback": fallback_reason is not None,
        "n_linear_parameters": len(context.linear_names),
        "n_nonlinear_parameters": len(context.nonlinear_names),
        "nonlinear_nfev": int(getattr(result, "nfev", 0) or 0),
        "nonlinear_njev": int(getattr(result, "njev", 0) or 0),
        "linear_solve_count": int(getattr(result, "linear_solve_count", 0) or 0),
        "decomposition_dependent": recipe.id == "paschen_nir",
        "optional_component_selection": selection_metadata,
    })
    return EmissionComplexResult(
        bool(result.success), int(result.status), str(result.message), recipe.id,
        {name: float(result.x[index]) for index, name in enumerate(context.names)},
        errors, covariance, metric_values, metric_errors, chi2, dof, float(reduced),
        float(bic), spectrum.wave_rest.copy(), line_flux, spectrum.err.copy(),
        context.model(result.x, spectrum.wave_rest),
        context.components(result.x, spectrum.wave_rest), mask, fit_warnings, metadata,
        result,
    )
