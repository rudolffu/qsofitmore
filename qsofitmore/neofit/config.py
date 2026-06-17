"""Configuration dataclasses for neofit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

Bounds = Tuple[Optional[float], Optional[float]]
Window = Tuple[float, float]


@dataclass(frozen=True)
class GaussianComponent:
    """Initial values and bounds for one Gaussian component."""

    name: str
    center: float
    amp: float
    sigma: float
    bounds: Dict[str, Bounds] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("GaussianComponent.name must be non-empty.")
        for field_name in ("center", "amp", "sigma"):
            if not np.isfinite(getattr(self, field_name)):
                raise ValueError(f"GaussianComponent.{field_name} must be finite.")
        if self.sigma <= 0:
            raise ValueError("GaussianComponent.sigma must be positive.")


@dataclass(frozen=True)
class IronTemplateConfig:
    """Configuration for one iron-template component with fitted FWHM."""

    template: str
    template_path: Optional[str] = None
    enabled: bool = True
    amp: float = 1.0
    amp_bounds: Bounds = (0.0, None)
    fwhm_kms: float = 3000.0
    fwhm_bounds: Bounds = (500.0, 10000.0)
    normalization: str = "area"

    def __post_init__(self) -> None:
        if not self.template:
            raise ValueError("IronTemplateConfig.template must be non-empty.")
        if not np.isfinite(self.amp):
            raise ValueError("IronTemplateConfig.amp must be finite.")
        if not np.isfinite(self.fwhm_kms) or self.fwhm_kms <= 0:
            raise ValueError("IronTemplateConfig.fwhm_kms must be positive and finite.")
        fwhm_lo, fwhm_hi = self.fwhm_bounds
        if fwhm_lo is not None and (not np.isfinite(fwhm_lo) or fwhm_lo <= 0):
            raise ValueError("IronTemplateConfig.fwhm_bounds lower bound must be positive and finite.")
        if fwhm_hi is not None and (not np.isfinite(fwhm_hi) or fwhm_hi <= 0):
            raise ValueError("IronTemplateConfig.fwhm_bounds upper bound must be positive and finite.")
        if fwhm_lo is not None and fwhm_hi is not None and fwhm_hi <= fwhm_lo:
            raise ValueError("IronTemplateConfig.fwhm_bounds upper bound must be greater than lower bound.")
        if self.normalization != "area":
            raise ValueError("Only IronTemplateConfig.normalization='area' is supported.")

    @classmethod
    def bg92(cls, fwhm_kms: float = 1500.0, **kwargs) -> "IronTemplateConfig":
        return cls(template="bg92", fwhm_kms=fwhm_kms, **kwargs)

    @classmethod
    def park22(cls, path: Optional[str] = None, fwhm_kms: float = 4000.0, **kwargs) -> "IronTemplateConfig":
        return cls(template="park22", template_path=path, fwhm_kms=fwhm_kms, **kwargs)

    @classmethod
    def veron04(cls, path: Optional[str] = None, fwhm_kms: float = 2500.0, **kwargs) -> "IronTemplateConfig":
        return cls(template="veron04", template_path=path, fwhm_kms=fwhm_kms, **kwargs)

    @classmethod
    def vw01(cls, fwhm_kms: float = 3000.0, **kwargs) -> "IronTemplateConfig":
        return cls(template="vw01", fwhm_kms=fwhm_kms, **kwargs)


@dataclass(frozen=True)
class LineComplexConfig:
    """Recipe for an MVP local emission-line complex fit."""

    center: float
    window: Window
    components: List[GaussianComponent]
    name: Optional[str] = None
    local_continuum: Optional[str] = "linear"
    iron: Optional[IronTemplateConfig] = None
    fit_windows: Optional[List[Window]] = None
    mask_windows: List[Window] = field(default_factory=list)
    plot_window: Optional[Window] = None
    jacobian: str = "analytic_dense"
    max_nfev: Optional[int] = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.center):
            raise ValueError("LineComplexConfig.center must be finite.")
        if len(self.window) != 2:
            raise ValueError("LineComplexConfig.window must contain two values.")
        lo, hi = map(float, self.window)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            raise ValueError("LineComplexConfig.window must be finite and increasing.")
        if self.fit_windows is not None:
            for subwindow in self.fit_windows:
                self._validate_subwindow(subwindow, "fit_windows")
        for subwindow in self.mask_windows:
            self._validate_subwindow(subwindow, "mask_windows")
        if self.plot_window is not None:
            self._validate_subwindow(self.plot_window, "plot_window")
        if not self.components:
            raise ValueError("LineComplexConfig.components must not be empty.")
        mode = self.local_continuum
        if mode not in (None, "constant", "linear"):
            raise ValueError("local_continuum must be None, 'constant', or 'linear'.")
        if self.jacobian not in ("analytic_dense", "analytic_sparse", "finite_difference"):
            raise ValueError("jacobian must be 'analytic_dense', 'analytic_sparse', or 'finite_difference'.")

    @staticmethod
    def _validate_subwindow(window: Window, label: str) -> None:
        if len(window) != 2:
            raise ValueError(f"LineComplexConfig.{label} entries must contain two values.")
        lo, hi = map(float, window)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            raise ValueError(f"LineComplexConfig.{label} entries must be finite and increasing.")


@dataclass(frozen=True)
class LocalFitConfig:
    """Configuration for fitting one or more independent local windows."""

    windows: List[LineComplexConfig]
    mode: str = "independent"
    require_min_pixels: int = 8
    edge_buffer: float = 0.0

    def __post_init__(self) -> None:
        if not self.windows:
            raise ValueError("LocalFitConfig.windows must not be empty.")
        if self.mode != "independent":
            raise ValueError("Only LocalFitConfig.mode='independent' is implemented.")
        if self.require_min_pixels < 1:
            raise ValueError("require_min_pixels must be positive.")
        if self.edge_buffer < 0:
            raise ValueError("edge_buffer must be non-negative.")
