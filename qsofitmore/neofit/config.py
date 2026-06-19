"""Configuration dataclasses for neofit."""

from __future__ import annotations

from dataclasses import dataclass, field
import warnings as _warnings
from typing import Dict, List, Optional, Tuple, Union

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
class LorentzianComponent:
    """Initial values and bounds for one Lorentzian component."""

    name: str
    center: float
    amp: float
    gamma: float
    bounds: Dict[str, Bounds] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("LorentzianComponent.name must be non-empty.")
        for field_name in ("center", "amp", "gamma"):
            if not np.isfinite(getattr(self, field_name)):
                raise ValueError(f"LorentzianComponent.{field_name} must be finite.")
        if self.gamma <= 0:
            raise ValueError("LorentzianComponent.gamma must be positive.")


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
    components: List[Union[GaussianComponent, LorentzianComponent]]
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


LEGACY_CONTINUUM_WINDOWS: Tuple[Window, ...] = (
    (1150.0, 1170.0),
    (1275.0, 1290.0),
    (1350.0, 1360.0),
    (1445.0, 1465.0),
    (1690.0, 1705.0),
    (1770.0, 1810.0),
    (1970.0, 2400.0),
    (2480.0, 2675.0),
    (2925.0, 3400.0),
    (3500.0, 3600.0),
    (3600.0, 4260.0),
    (4435.0, 4640.0),
    (5100.0, 5535.0),
    (6005.0, 6035.0),
    (6110.0, 6250.0),
    (6800.0, 7000.0),
    (7180.0, 7250.0),
    (7600.0, 7700.0),
    (7950.0, 8050.0),
    (8600.0, 8800.0),
    (9350.0, 9400.0),
    (9650.0, 9800.0),
    (10200.0, 10600.0),
    (11400.0, 12400.0),
)


@dataclass(frozen=True)
class PowerLawConfig:
    """Pivoted global ``f_lambda`` power-law configuration."""

    enabled: bool = True
    pivot: float = 3000.0
    norm: Optional[float] = None
    norm_bounds: Bounds = (0.0, None)
    slope: float = -1.5
    slope_bounds: Bounds = (-5.0, 3.0)


@dataclass(frozen=True)
class BalmerContinuumConfig:
    """Dietrich-style Balmer continuum with fixed physical shape."""

    enabled: bool = True
    edge: float = 3646.0
    min_wave: float = 2000.0
    temperature_k: float = 15000.0
    tau_edge: float = 1.0
    amplitude: float = 0.1
    amplitude_bounds: Bounds = (0.0, None)


@dataclass(frozen=True)
class BalmerSeriesConfig:
    """High-order Balmer-series template selection and broadening."""

    enabled: bool = True
    log10_ne: int = 9
    n_min: int = 6
    provenance: str = "sh95"
    amplitude: float = 1.0
    amplitude_bounds: Bounds = (0.0, None)
    fit_fwhm: bool = True
    fwhm_kms: float = 5000.0
    fwhm_bounds: Bounds = (500.0, 15000.0)
    sync_with_hbeta: str = "auto"
    sync_min_fwhm_snr: Optional[float] = 3.0
    fixed_fwhm_kms: Optional[float] = None

    def __post_init__(self) -> None:
        if self.log10_ne not in (9, 10):
            raise ValueError("BalmerSeriesConfig.log10_ne must be 9 or 10.")
        if self.n_min not in (6, 7):
            raise ValueError("BalmerSeriesConfig.n_min must be 6 or 7.")
        if self.provenance not in ("sh95", "sh95_k13full_ext", "sh95_asymptotic_ext"):
            raise ValueError("Unsupported Balmer-series provenance.")
        if not np.isfinite(self.fwhm_kms) or self.fwhm_kms <= 0:
            raise ValueError("BalmerSeriesConfig.fwhm_kms must be positive and finite.")
        fwhm_lo, fwhm_hi = self.fwhm_bounds
        if fwhm_lo is None or fwhm_hi is None or fwhm_lo <= 0 or fwhm_hi <= fwhm_lo:
            raise ValueError("BalmerSeriesConfig.fwhm_bounds must be finite, positive, and increasing.")
        if self.sync_with_hbeta not in ("auto", "never", "require"):
            raise ValueError(
                "BalmerSeriesConfig.sync_with_hbeta must be 'auto', 'never', or 'require'."
            )
        if self.sync_min_fwhm_snr is not None and self.sync_min_fwhm_snr < 0:
            raise ValueError("BalmerSeriesConfig.sync_min_fwhm_snr must be non-negative or None.")
        if self.fixed_fwhm_kms is not None and self.fixed_fwhm_kms <= 0:
            raise ValueError("fixed_fwhm_kms must be positive.")
        if self.fixed_fwhm_kms is not None:
            _warnings.warn(
                "BalmerSeriesConfig.fixed_fwhm_kms is deprecated; use "
                "fit_fwhm=False with fwhm_kms instead.",
                DeprecationWarning,
                stacklevel=2,
            )


@dataclass(frozen=True)
class GlobalContinuumConfig:
    """Configuration for the first neofit global AGN continuum."""

    power_law: PowerLawConfig = field(default_factory=PowerLawConfig)
    uv_iron: Optional[IronTemplateConfig] = field(
        default_factory=lambda: IronTemplateConfig.vw01(fwhm_kms=3000.0)
    )
    optical_iron: Optional[IronTemplateConfig] = field(
        default_factory=lambda: IronTemplateConfig.park22(fwhm_kms=3000.0)
    )
    balmer_continuum: BalmerContinuumConfig = field(default_factory=BalmerContinuumConfig)
    balmer_series: BalmerSeriesConfig = field(default_factory=BalmerSeriesConfig)
    continuum_windows: Tuple[Window, ...] = LEGACY_CONTINUUM_WINDOWS
    mask_windows: Tuple[Window, ...] = ((3710.0, 3745.0), (3855.0, 3880.0))
    min_component_pixels: int = 20
    clip_passes: int = 2
    clip_low_sigma: float = 3.0
    clip_high_sigma: float = 5.0
    balmer_width_sync_tolerance_kms: float = 5.0
    balmer_width_sync_max_iterations: int = 5
    optimizer_method: str = "auto"
    jacobian_method: str = "semi_analytic"
    max_nfev: Optional[int] = 1000

    def __post_init__(self) -> None:
        if self.optimizer_method not in ("auto", "variable_projection", "legacy_joint"):
            raise ValueError(
                "optimizer_method must be 'auto', 'variable_projection', or 'legacy_joint'."
            )
        if self.jacobian_method not in ("semi_analytic", "2-point"):
            raise ValueError("jacobian_method must be 'semi_analytic' or '2-point'.")
        if self.balmer_width_sync_tolerance_kms <= 0:
            raise ValueError("balmer_width_sync_tolerance_kms must be positive.")
        if self.balmer_width_sync_max_iterations < 1:
            raise ValueError("balmer_width_sync_max_iterations must be at least one.")


@dataclass(frozen=True)
class HbetaComplexConfig:
    """Configuration for the constrained H-beta/[O III] model."""

    window: Window = (4640.0, 5100.0)
    broad_fwhm_bands_kms: Tuple[Tuple[float, float], ...] = (
        (900.0, 2500.0),
        (2500.0, 6000.0),
        (6000.0, 20000.0),
    )
    broad_velocity_bounds_kms: Tuple[float, float] = (-2000.0, 2000.0)
    narrow_fwhm_bounds_kms: Tuple[float, float] = (70.0, 1200.0)
    narrow_velocity_bounds_kms: Tuple[float, float] = (-1000.0, 1000.0)
    oiii_ratio_5007_4959: float = 2.98
    fit_oiii_wings: bool = True
    wing_bic_delta: float = 10.0
    wing_min_snr: float = 3.0
    heii_enabled: bool = False
    heii_mask: Window = (4660.0, 4715.0)
    optimizer_method: str = "auto"
    jacobian_method: str = "semi_analytic"
    max_nfev: Optional[int] = 1500

    def __post_init__(self) -> None:
        if self.optimizer_method not in ("auto", "variable_projection", "legacy_joint"):
            raise ValueError(
                "optimizer_method must be 'auto', 'variable_projection', or 'legacy_joint'."
            )
        if self.jacobian_method not in ("semi_analytic", "2-point"):
            raise ValueError("jacobian_method must be 'semi_analytic' or '2-point'.")


@dataclass(frozen=True)
class MgIIComplexConfig:
    """Configuration for broad and narrow Mg II emission."""

    window: Window = (2700.0, 2900.0)
    broad_fwhm_bands_kms: Tuple[Tuple[float, float], ...] = (
        (900.0, 3500.0),
        (3500.0, 15000.0),
    )
    broad_velocity_bounds_kms: Tuple[float, float] = (-2000.0, 2000.0)
    narrow_fwhm_bounds_kms: Tuple[float, float] = (70.0, 1200.0)
    narrow_velocity_bounds_kms: Tuple[float, float] = (-1000.0, 1000.0)
    min_coverage_fraction: float = 0.8
    min_valid_pixels: int = 30
    edge_margin_kms: float = 1000.0
    optimizer_method: str = "auto"
    jacobian_method: str = "semi_analytic"
    max_nfev: Optional[int] = 1500

    def __post_init__(self) -> None:
        if len(self.broad_fwhm_bands_kms) != 2:
            raise ValueError("MgIIComplexConfig requires two broad FWHM bands.")
        if self.narrow_fwhm_bounds_kms[0] <= 0:
            raise ValueError("Mg II narrow FWHM bounds must be positive.")
        if self.narrow_fwhm_bounds_kms[1] <= self.narrow_fwhm_bounds_kms[0]:
            raise ValueError("Mg II narrow FWHM bounds must be increasing.")
        _validate_complex_optimizer_config(self)


@dataclass(frozen=True)
class HalphaComplexConfig:
    """Configuration for the H-alpha/[N II]/[S II] complex."""

    window: Window = (6400.0, 6800.0)
    broad_fwhm_bands_kms: Tuple[Tuple[float, float], ...] = (
        (900.0, 2500.0),
        (2500.0, 6000.0),
        (6000.0, 20000.0),
    )
    broad_velocity_bounds_kms: Tuple[float, float] = (-2000.0, 2000.0)
    narrow_fwhm_bounds_kms: Tuple[float, float] = (70.0, 1200.0)
    narrow_velocity_bounds_kms: Tuple[float, float] = (-1000.0, 1000.0)
    nii_ratio_6585_6549: float = 2.96
    min_coverage_fraction: float = 0.8
    min_valid_pixels: int = 30
    edge_margin_kms: float = 1000.0
    optimizer_method: str = "auto"
    jacobian_method: str = "semi_analytic"
    max_nfev: Optional[int] = 1500

    def __post_init__(self) -> None:
        if len(self.broad_fwhm_bands_kms) != 3:
            raise ValueError("HalphaComplexConfig requires three broad FWHM bands.")
        if self.nii_ratio_6585_6549 <= 0:
            raise ValueError("nii_ratio_6585_6549 must be positive.")
        _validate_complex_optimizer_config(self)


def _validate_complex_optimizer_config(config) -> None:
    if config.optimizer_method not in ("auto", "variable_projection", "legacy_joint"):
        raise ValueError(
            "optimizer_method must be 'auto', 'variable_projection', or 'legacy_joint'."
        )
    if config.jacobian_method not in ("semi_analytic", "2-point"):
        raise ValueError("jacobian_method must be 'semi_analytic' or '2-point'.")
    if not 0 < config.min_coverage_fraction <= 1:
        raise ValueError("min_coverage_fraction must be in (0, 1].")
    if config.min_valid_pixels < 1:
        raise ValueError("min_valid_pixels must be positive.")
    if config.edge_margin_kms < 0:
        raise ValueError("edge_margin_kms must be non-negative.")


@dataclass(frozen=True)
class UncertaintyConfig:
    """Statistical uncertainty settings for the global workflow."""

    covariance: bool = True
    monte_carlo_trials: int = 0
    random_seed: Optional[int] = 12345
    refit_host_in_mc: bool = True
