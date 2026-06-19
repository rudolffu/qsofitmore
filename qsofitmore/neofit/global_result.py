"""Result containers for the neofit global continuum and H-beta workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .spectrum import Spectrum
from .warnings import NeoFitWarning


@dataclass
class GlobalContinuumResult:
    """Global continuum fit evaluated on the full input grid."""

    success: bool
    status: int
    message: str
    param_values: Dict[str, float]
    param_errors: Dict[str, float]
    covariance: Optional[np.ndarray]
    chi2: float
    dof: int
    reduced_chi2: float
    wave_rest: np.ndarray
    model: np.ndarray
    component_models: Dict[str, np.ndarray]
    fit_mask: np.ndarray
    clip_mask: np.ndarray
    warnings: List[NeoFitWarning] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    optimizer_result: Optional[Any] = None

    def warning_codes(self) -> List[str]:
        return [warning.code for warning in self.warnings]

    def summary(self) -> Dict[str, Any]:
        return {
            "success": bool(self.success),
            "status": int(self.status),
            "message": self.message,
            "param_values": dict(self.param_values),
            "param_errors": dict(self.param_errors),
            "chi2": float(self.chi2),
            "dof": int(self.dof),
            "reduced_chi2": float(self.reduced_chi2),
            "n_fit_pixels": int(np.count_nonzero(self.fit_mask)),
            "n_clipped_pixels": int(np.count_nonzero(self.fit_mask & ~self.clip_mask)),
            "warning_codes": self.warning_codes(),
            "metadata": dict(self.metadata),
        }


@dataclass
class EmissionComplexResult:
    """One continuum-subtracted emission-line complex fit."""

    success: bool
    status: int
    message: str
    selected_model: str
    param_values: Dict[str, float]
    param_errors: Dict[str, float]
    covariance: Optional[np.ndarray]
    metrics: Dict[str, float]
    metric_errors: Dict[str, float]
    chi2: float
    dof: int
    reduced_chi2: float
    bic: float
    wave_rest: np.ndarray
    flux_continuum_subtracted: np.ndarray
    err: np.ndarray
    model: np.ndarray
    component_models: Dict[str, np.ndarray]
    fit_mask: np.ndarray
    warnings: List[NeoFitWarning] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    optimizer_result: Optional[Any] = None

    def warning_codes(self) -> List[str]:
        return [warning.code for warning in self.warnings]

    def summary(self) -> Dict[str, Any]:
        return {
            "success": bool(self.success),
            "status": int(self.status),
            "message": self.message,
            "selected_model": self.selected_model,
            "param_values": dict(self.param_values),
            "param_errors": dict(self.param_errors),
            "metrics": dict(self.metrics),
            "metric_errors": dict(self.metric_errors),
            "chi2": float(self.chi2),
            "dof": int(self.dof),
            "reduced_chi2": float(self.reduced_chi2),
            "bic": float(self.bic),
            "n_fit_pixels": int(np.count_nonzero(self.fit_mask)),
            "warning_codes": self.warning_codes(),
            "metadata": dict(self.metadata),
        }


HbetaComplexResult = EmissionComplexResult


@dataclass
class NeoFitWorkflowResult:
    """One optional-host, global-continuum, multi-complex workflow."""

    spectrum: Spectrum
    continuum_initial: GlobalContinuumResult
    continuum: GlobalContinuumResult
    hbeta_initial: Optional[HbetaComplexResult] = None
    hbeta: Optional[HbetaComplexResult] = None
    mgii: Optional[EmissionComplexResult] = None
    halpha: Optional[EmissionComplexResult] = None
    line_complexes: Dict[str, EmissionComplexResult] = field(default_factory=dict)
    complex_statuses: Dict[str, str] = field(default_factory=dict)
    host_decomp_enabled: bool = False
    total_spectrum: Optional[Spectrum] = None
    host_fit: Optional[Any] = None
    host_sed: Optional[Any] = None
    host_model_on_quasar_grid: Optional[np.ndarray] = None
    host_warnings: List[str] = field(default_factory=list)
    monte_carlo: Dict[str, Any] = field(default_factory=dict)
    warnings: List[NeoFitWarning] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    output_files: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.line_complexes:
            self.line_complexes = {}
            if self.hbeta is not None:
                self.line_complexes["hbeta_oiii"] = self.hbeta
            if self.mgii is not None:
                self.line_complexes["mgii"] = self.mgii
            if self.halpha is not None:
                self.line_complexes["halpha_nii_sii"] = self.halpha

    @property
    def continuum_success(self) -> bool:
        return bool(self.continuum.success)

    @property
    def legacy_hbeta_success(self) -> bool:
        """Deprecated Hβ-oriented success verdict."""

        return bool(
            self.continuum.success
            and self.hbeta is not None
            and self.hbeta.success
        )

    def warning_codes(self) -> List[str]:
        codes = [warning.code for warning in self.warnings]
        codes.extend(self.continuum.warning_codes())
        for result in self.line_complexes.values():
            codes.extend(result.warning_codes())
        return codes

    def summary(self) -> Dict[str, Any]:
        payload = {
            "continuum_success": self.continuum_success,
            "host_decomp_enabled": bool(self.host_decomp_enabled),
            "continuum": self.continuum.summary(),
            "hbeta": self.hbeta.summary() if self.hbeta is not None else None,
            "mgii": self.mgii.summary() if self.mgii is not None else None,
            "halpha": self.halpha.summary() if self.halpha is not None else None,
            "complex_statuses": dict(self.complex_statuses),
            "line_complexes": {
                name: result.summary() for name, result in self.line_complexes.items()
            },
            "monte_carlo": dict(self.monte_carlo),
            "host_warnings": list(self.host_warnings),
            "warning_codes": self.warning_codes(),
            "metadata": dict(self.metadata),
            "output_files": dict(self.output_files),
        }
        if self.metadata.get("compatibility_hbeta_mode", False):
            payload["legacy_hbeta_success"] = self.legacy_hbeta_success
        return payload
