"""Result containers for neofit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .warnings import NeoFitWarning


_GAUSSIAN_FLUX_FACTOR = float(np.sqrt(2.0 * np.pi))
_GAUSSIAN_FWHM_FACTOR = float(np.sqrt(8.0 * np.log(2.0)))


@dataclass
class FitResult:
    """Result of a neofit optimization."""

    success: bool
    status: int
    message: str
    theta: np.ndarray
    param_names: List[str]
    param_values: Dict[str, float]
    chi2: float
    dof: int
    reduced_chi2: float
    model: np.ndarray
    residual: np.ndarray
    wave_rest_fit: np.ndarray
    flux_fit: np.ndarray
    err_fit: np.ndarray
    wave_rest_window: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    flux_window: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    err_window: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    model_window: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    residual_window: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    fit_used_window: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    component_models: Dict[str, np.ndarray] = field(default_factory=dict)
    component_models_window: Dict[str, np.ndarray] = field(default_factory=dict)
    warnings: List[NeoFitWarning] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    optimizer_result: Optional[Any] = None

    @classmethod
    def failed(
        cls,
        message: str,
        warnings: Optional[List[NeoFitWarning]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "FitResult":
        """Create a failed result for validation-level failures."""

        return cls(
            success=False,
            status=-1,
            message=message,
            theta=np.array([], dtype=float),
            param_names=[],
            param_values={},
            chi2=float("nan"),
            dof=0,
            reduced_chi2=float("nan"),
            model=np.array([], dtype=float),
            residual=np.array([], dtype=float),
            wave_rest_fit=np.array([], dtype=float),
            flux_fit=np.array([], dtype=float),
            err_fit=np.array([], dtype=float),
            component_models={},
            warnings=list(warnings or []),
            metadata=dict(metadata or {}),
            optimizer_result=None,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly summary dictionary."""

        return {
            "success": bool(self.success),
            "status": int(self.status),
            "message": str(self.message),
            "param_values": dict(self.param_values),
            "chi2": float(self.chi2),
            "dof": int(self.dof),
            "reduced_chi2": float(self.reduced_chi2),
            "n_pixels": int(self.wave_rest_fit.size),
            "param_names": list(self.param_names),
            "metadata": dict(self.metadata),
            "warnings": [warning.to_dict() for warning in self.warnings],
        }

    def to_table(self):
        """Return Gaussian line measurements as a pandas DataFrame when available."""

        scale = self.metadata.get("flux_density_scale_to_cgs")
        flux_unit = self.metadata.get("flux_density_unit", "input")
        rows = []
        component_names = sorted(
            {
                name.rsplit(".", 1)[0]
                for name in self.param_values
                if name.endswith((".amp", ".center", ".sigma"))
            }
        )
        for component in component_names:
            keys = {
                "amp": f"{component}.amp",
                "center": f"{component}.center",
                "sigma": f"{component}.sigma",
            }
            if not all(key in self.param_values for key in keys.values()):
                continue
            amp = float(self.param_values[keys["amp"]])
            center = float(self.param_values[keys["center"]])
            sigma = float(self.param_values[keys["sigma"]])
            line_flux_input = amp * sigma * _GAUSSIAN_FLUX_FACTOR
            line_flux_cgs = line_flux_input * float(scale) if scale is not None else np.nan
            rows.append(
                {
                    "name": component,
                    "component_type": "gaussian",
                    "amp": amp,
                    "center": center,
                    "sigma": sigma,
                    "fwhm": sigma * _GAUSSIAN_FWHM_FACTOR,
                    "line_flux_input": line_flux_input,
                    "line_flux_cgs": line_flux_cgs,
                    "flux_density_unit": flux_unit,
                    "flux_density_scale_to_cgs": scale,
                    "success": bool(self.success),
                }
            )
        iron = self.metadata.get("iron")
        if isinstance(iron, dict) and "iron.amp" in self.param_values:
            iron_flux_input = float(iron.get("iron_flux_input", np.nan))
            iron_flux_cgs = float(iron.get("iron_flux_cgs", np.nan))
            rows.append(
                {
                    "name": "iron",
                    "component_type": "iron",
                    "amp": np.nan,
                    "center": np.nan,
                    "sigma": np.nan,
                    "fwhm": np.nan,
                    "line_flux_input": iron_flux_input,
                    "line_flux_cgs": iron_flux_cgs,
                    "flux_density_unit": flux_unit,
                    "flux_density_scale_to_cgs": scale,
                    "success": bool(self.success),
                    "iron_template": iron.get("template"),
                    "iron_amp": float(self.param_values["iron.amp"]),
                    "iron_fwhm_kms": float(iron.get("fwhm_kms", np.nan)),
                    "iron_flux_input": iron_flux_input,
                    "iron_flux_cgs": iron_flux_cgs,
                    "iron_template_coverage_min": float(iron.get("template_coverage_min", np.nan)),
                    "iron_template_coverage_max": float(iron.get("template_coverage_max", np.nan)),
                    "iron_template_reference": iron.get("template_reference"),
                }
            )
        try:
            import pandas as pd

            return pd.DataFrame(rows)
        except Exception:
            return rows

    def warning_codes(self) -> List[str]:
        """Return warning codes attached to this result."""

        return [warning.code for warning in self.warnings]

    def summary(self) -> Dict[str, Any]:
        """Return a compact result summary."""

        return self.to_dict()


@dataclass
class LocalFitResult:
    """Result of fitting one or more independent local windows."""

    success: bool
    window_results: Dict[str, FitResult]
    warnings: List[NeoFitWarning] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def warning_codes(self) -> List[str]:
        """Return warning codes from the local result and all window results."""

        codes = [warning.code for warning in self.warnings]
        for result in self.window_results.values():
            codes.extend(result.warning_codes())
        return codes

    def to_table(self):
        """Combine successful window line tables and add a ``window`` column."""

        rows = []
        for window, result in self.window_results.items():
            table = result.to_table()
            if hasattr(table, "to_dict"):
                records = table.to_dict("records")
            else:
                records = list(table)
            for row in records:
                payload = dict(row)
                payload["window"] = window
                rows.append(payload)
        try:
            import pandas as pd

            return pd.DataFrame(rows)
        except Exception:
            return rows

    def summary(self) -> Dict[str, Any]:
        """Return a compact local-fit summary."""

        return {
            "success": bool(self.success),
            "n_windows": int(len(self.window_results)),
            "n_success": int(sum(result.success for result in self.window_results.values())),
            "warning_codes": self.warning_codes(),
            "metadata": dict(self.metadata),
        }
