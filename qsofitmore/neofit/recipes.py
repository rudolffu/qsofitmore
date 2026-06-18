"""Small convenience recipes for neofit local broad-line fitting."""

from __future__ import annotations

from typing import Iterable, List, Optional

from .config import GaussianComponent, IronTemplateConfig, LineComplexConfig, LocalFitConfig, LorentzianComponent


def _iron_config(
    iron_template: Optional[str],
    iron_template_path: Optional[str],
    iron_fwhm_kms: float,
) -> Optional[IronTemplateConfig]:
    if iron_template is None:
        return None
    return IronTemplateConfig(
        template=iron_template,
        template_path=iron_template_path,
        fwhm_kms=iron_fwhm_kms,
    )


def local_hbeta(
    iron_template: Optional[str] = None,
    iron_template_path: Optional[str] = None,
    iron_fwhm_kms: float = 3000.0,
    profile: str = "gaussian",
) -> LineComplexConfig:
    """Return a minimal broad H-beta local-window recipe."""

    profile_key = str(profile).strip().lower()
    if profile_key == "gaussian":
        component = GaussianComponent(
            name="Hb_broad",
            center=4861.33,
            amp=10.0,
            sigma=30.0,
            bounds={"amp": (0.0, None), "center": (4800.0, 4920.0), "sigma": (5.0, 200.0)},
        )
    elif profile_key == "lorentzian":
        component = LorentzianComponent(
            name="Hb_broad",
            center=4861.33,
            amp=10.0,
            gamma=20.0,
            bounds={"amp": (0.0, None), "center": (4800.0, 4920.0), "gamma": (3.0, 150.0)},
        )
    else:
        raise ValueError("local_hbeta profile must be 'gaussian' or 'lorentzian'.")

    return LineComplexConfig(
        name="Hb_OIII",
        center=4861.33,
        window=(4435.0, 5535.0),
        fit_windows=[(4435.0, 4640.0), (4700.0, 5100.0), (5100.0, 5535.0)],
        mask_windows=[(4930.0, 5035.0)],
        plot_window=(4700.0, 5100.0),
        local_continuum="linear",
        iron=_iron_config(iron_template, iron_template_path, iron_fwhm_kms),
        components=[component],
    )


def local_halpha(
    iron_template: Optional[str] = None,
    iron_template_path: Optional[str] = None,
    iron_fwhm_kms: float = 3000.0,
) -> LineComplexConfig:
    """Return a minimal broad H-alpha local-window recipe."""

    return LineComplexConfig(
        name="Ha_NII_SII",
        center=6562.8,
        window=(6005.0, 7000.0),
        fit_windows=[(6005.0, 6035.0), (6110.0, 6250.0), (6300.0, 6800.0), (6800.0, 7000.0)],
        mask_windows=[(6290.0, 6320.0), (6535.0, 6605.0), (6700.0, 6750.0)],
        plot_window=(6300.0, 6800.0),
        local_continuum="linear",
        iron=_iron_config(iron_template, iron_template_path, iron_fwhm_kms),
        components=[
            GaussianComponent(
                name="Ha_broad",
                center=6562.8,
                amp=10.0,
                sigma=40.0,
                bounds={"amp": (0.0, None), "center": (6450.0, 6650.0), "sigma": (5.0, 300.0)},
            )
        ],
    )


def local_mgii(
    iron_template: Optional[str] = None,
    iron_template_path: Optional[str] = None,
    iron_fwhm_kms: float = 3000.0,
) -> LineComplexConfig:
    """Return a minimal broad MgII local-window recipe."""

    return LineComplexConfig(
        name="MgII",
        center=2798.75,
        window=(1970.0, 3400.0),
        fit_windows=[(1970.0, 2400.0), (2480.0, 2675.0), (2700.0, 2900.0), (2925.0, 3400.0)],
        plot_window=(2700.0, 2900.0),
        local_continuum="linear",
        iron=_iron_config(iron_template, iron_template_path, iron_fwhm_kms),
        components=[
            GaussianComponent(
                name="MgII_broad",
                center=2798.75,
                amp=10.0,
                sigma=25.0,
                bounds={"amp": (0.0, None), "center": (2740.0, 2860.0), "sigma": (5.0, 180.0)},
            )
        ],
    )


def local_broad_lines(lines: Iterable[str]) -> LocalFitConfig:
    """Return a local-fit config for selected broad-line recipe names."""

    lookup = {"hb": local_hbeta, "hbeta": local_hbeta, "ha": local_halpha, "halpha": local_halpha, "mgii": local_mgii}
    windows: List[LineComplexConfig] = []
    for line in lines:
        key = str(line).strip().lower()
        if key not in lookup:
            raise ValueError(f"Unknown neofit local broad-line recipe: {line!r}")
        windows.append(lookup[key]())
    return LocalFitConfig(windows=windows)
