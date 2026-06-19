"""Immutable emission-complex recipes and built-in registry."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Any, Dict, Iterable, Optional, Tuple

from . import lines

Bounds = Tuple[Optional[float], Optional[float]]
Window = Tuple[float, float]


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


@dataclass(frozen=True)
class ComponentRecipe:
    id: str
    line_ids: Tuple[str, ...]
    role: str
    profile: str = "gaussian"
    multiplicity: int = 1
    enabled: bool = True
    required: bool = False
    flux_bounds: Bounds = (0.0, None)
    velocity_bounds_kms: Tuple[float, float] = (-1000.0, 1000.0)
    fwhm_bands_kms: Tuple[Tuple[float, float], ...] = ((70.0, 1200.0),)
    kinematic_group: Optional[str] = None
    fixed_ratio_to: Optional[str] = None
    fixed_ratio: Optional[float] = None
    selection_rule: Optional[str] = None

    def __post_init__(self) -> None:
        if self.role not in ("broad", "narrow", "very_broad", "wing", "blend"):
            raise ValueError(f"Unsupported component role: {self.role!r}")
        if self.profile not in ("gaussian", "lorentzian"):
            raise ValueError(f"Unsupported component profile: {self.profile!r}")
        if self.multiplicity < 1:
            raise ValueError("ComponentRecipe.multiplicity must be positive.")
        for line_id in self.line_ids:
            lines.get(line_id)
        if self.fixed_ratio_to is not None and (
            self.fixed_ratio is None or self.fixed_ratio <= 0
        ):
            raise ValueError("A fixed-ratio component requires fixed_ratio > 0.")


@dataclass(frozen=True)
class ComplexRecipe:
    id: str
    aliases: Tuple[str, ...]
    label: str
    fit_window: Window
    fit_windows: Tuple[Window, ...]
    mask_windows: Tuple[Window, ...]
    components: Tuple[ComponentRecipe, ...]
    required_line_ids: Tuple[str, ...]
    coverage_mode: str = "full"
    min_coverage_fraction: float = 0.8
    min_valid_pixels: int = 30
    edge_margin_kms: float = 1000.0
    continuum_mode: str = "fixed_global"
    qa_labels: Tuple[str, ...] = ()
    auto_enabled: bool = False
    priority: int = 0
    backend: str = "generic"
    exclusive_group: Optional[str] = None

    def __post_init__(self) -> None:
        if self.coverage_mode not in ("full", "component_adaptive"):
            raise ValueError("coverage_mode must be 'full' or 'component_adaptive'.")
        if self.continuum_mode not in ("fixed_global", "constant", "linear", "absent"):
            raise ValueError("Unsupported continuum_mode.")
        if self.backend not in ("generic", "mgii_adapter", "hbeta_adapter", "halpha_adapter"):
            raise ValueError("Unsupported recipe backend.")
        for line_id in self.required_line_ids:
            lines.get(line_id)

    def with_component(self, component_id: str, **changes: Any) -> "ComplexRecipe":
        """Return a copy with one component replaced."""

        found = False
        updated = []
        for component in self.components:
            if component.id == component_id:
                component = replace(component, **changes)
                found = True
            updated.append(component)
        if not found:
            raise ValueError(f"unknown_recipe_component: {component_id!r}")
        return replace(self, components=tuple(updated))


def _component(id, line_ids, role, **kwargs):
    return ComponentRecipe(id=id, line_ids=tuple(line_ids), role=role, **kwargs)


_RECIPES = (
    ComplexRecipe(
        id="mgii", aliases=("mgii_blend",), label="Mg II",
        fit_window=(2700.0, 2900.0), fit_windows=((2700.0, 2900.0),), mask_windows=(),
        components=(
            _component(
                "MgII_broad", ("mgii_blend",), "broad", multiplicity=2,
                velocity_bounds_kms=(-2000.0, 2000.0),
                fwhm_bands_kms=((900.0, 3500.0), (3500.0, 15000.0)),
                kinematic_group="MgII_broad",
            ),
            _component(
                "MgII_narrow",
                ("mgii_blend",),
                "narrow",
                kinematic_group="MgII_narrow",
            ),
        ), required_line_ids=("mgii_blend",), qa_labels=("mgii_blend",),
        auto_enabled=True, priority=90, backend="mgii_adapter", exclusive_group="mgii",
    ),
    ComplexRecipe(
        id="hbeta_oiii", aliases=("hbeta", "hb_oiii"), label="Hβ / [O III]",
        fit_window=(4640.0, 5100.0), fit_windows=((4640.0, 5100.0),), mask_windows=(),
        components=(
            _component(
                "Hb_broad", ("hbeta",), "broad", multiplicity=3,
                velocity_bounds_kms=(-2000.0, 2000.0),
                fwhm_bands_kms=((900.0, 2500.0), (2500.0, 6000.0), (6000.0, 20000.0)),
                kinematic_group="Hb_broad",
            ),
            _component("Hb_narrow", ("hbeta",), "narrow", kinematic_group="hbeta_narrow"),
            _component("OIII5008_core", ("oiii_5008",), "narrow", kinematic_group="hbeta_narrow"),
            _component(
                "OIII4960_core", ("oiii_4960",), "narrow",
                kinematic_group="hbeta_narrow", fixed_ratio_to="OIII5008_core",
                fixed_ratio=2.98,
            ),
            _component(
                "OIII5008_wing", ("oiii_5008",), "wing",
                kinematic_group="oiii_wing", selection_rule="bic_and_snr",
                fwhm_bands_kms=((150.0, 2500.0),),
            ),
            _component(
                "OIII4960_wing", ("oiii_4960",), "wing",
                kinematic_group="oiii_wing", fixed_ratio_to="OIII5008_wing",
                fixed_ratio=2.98, selection_rule="bic_and_snr",
                fwhm_bands_kms=((150.0, 2500.0),),
            ),
            _component(
                "HeII_broad", ("heii_4687",), "broad", enabled=False,
                kinematic_group="HeII_broad",
            ),
        ), required_line_ids=("hbeta", "oiii_4960", "oiii_5008"),
        qa_labels=("hbeta", "oiii_4960", "oiii_5008"), auto_enabled=True,
        priority=100, backend="hbeta_adapter", exclusive_group="hbeta",
    ),
    ComplexRecipe(
        id="halpha_nii_sii", aliases=("halpha", "ha_nii_sii"), label="Hα / [N II] / [S II]",
        fit_window=(6400.0, 6800.0), fit_windows=((6400.0, 6800.0),), mask_windows=(),
        components=(
            _component(
                "Ha_broad", ("halpha",), "broad", multiplicity=3,
                velocity_bounds_kms=(-2000.0, 2000.0),
                fwhm_bands_kms=((900.0, 2500.0), (2500.0, 6000.0), (6000.0, 20000.0)),
                kinematic_group="Ha_broad",
            ),
            _component("Ha_narrow", ("halpha",), "narrow", kinematic_group="halpha_narrow"),
            _component("NII6585", ("nii_6585",), "narrow", kinematic_group="halpha_narrow"),
            _component(
                "NII6550", ("nii_6550",), "narrow",
                kinematic_group="halpha_narrow", fixed_ratio_to="NII6585",
                fixed_ratio=2.96,
            ),
            _component("SII6718", ("sii_6718",), "narrow", kinematic_group="halpha_narrow"),
            _component("SII6733", ("sii_6733",), "narrow", kinematic_group="halpha_narrow"),
        ), required_line_ids=("halpha", "nii_6550", "nii_6585", "sii_6718", "sii_6733"),
        qa_labels=("halpha", "nii_6550", "nii_6585", "sii_6718", "sii_6733"),
        auto_enabled=True, priority=100, backend="halpha_adapter", exclusive_group="halpha",
    ),
    ComplexRecipe(
        id="oii_nev_neiii_hgamma", aliases=("optical_blue",), label="[Ne V] / [O II] / [Ne III] / Hγ",
        fit_window=(3380.0, 4425.0), fit_windows=((3380.0, 3970.0), (4050.0, 4425.0)),
        mask_windows=(),
        components=(
            _component("NeV3427", ("nev_3427",), "narrow", kinematic_group="blue_narrow"),
            _component("OII3727", ("oii_3727",), "narrow", kinematic_group="blue_narrow"),
            _component("OII3730", ("oii_3730",), "narrow", kinematic_group="blue_narrow"),
            _component("NeIII3870", ("neiii_3870",), "narrow", kinematic_group="blue_narrow"),
            _component("Hgamma_narrow", ("hgamma",), "narrow", kinematic_group="blue_narrow"),
            _component("Hgamma_broad", ("hgamma",), "broad", required=False,
                       velocity_bounds_kms=(-2000.0, 2000.0),
                       fwhm_bands_kms=((900.0, 15000.0),), kinematic_group="hgamma_broad"),
        ),
        required_line_ids=(), coverage_mode="component_adaptive", qa_labels=("nev_3427", "oii_blend", "neiii_3870", "hgamma"),
        auto_enabled=True, priority=40, backend="generic", exclusive_group="optical_blue",
    ),
    ComplexRecipe(
        id="paschen_nir", aliases=("nir", "paschen"), label="Paschen / NIR",
        fit_window=(9900.0, 13050.0),
        fit_windows=((9900.0, 10120.0), (10720.0, 11040.0), (11180.0, 11380.0), (12650.0, 13050.0)),
        mask_windows=(),
        components=(
            _component("Padelta_broad", ("padelta",), "broad", kinematic_group="padelta_broad",
                       velocity_bounds_kms=(-2000.0, 2000.0), fwhm_bands_kms=((900.0, 15000.0),)),
            _component("HeI10833_broad", ("hei_10833",), "broad", kinematic_group="hei_pgamma_broad",
                       velocity_bounds_kms=(-2000.0, 2000.0), fwhm_bands_kms=((900.0, 15000.0),)),
            _component("Pagamma_broad", ("pagamma",), "broad", kinematic_group="hei_pgamma_broad",
                       velocity_bounds_kms=(-2000.0, 2000.0), fwhm_bands_kms=((900.0, 15000.0),)),
            _component("OI11290_broad", ("oi_11290",), "broad", kinematic_group="oi11290_broad",
                       velocity_bounds_kms=(-2000.0, 2000.0), fwhm_bands_kms=((900.0, 15000.0),)),
            _component("Pabeta_broad", ("pabeta",), "broad", kinematic_group="pabeta_broad",
                       velocity_bounds_kms=(-2000.0, 2000.0), fwhm_bands_kms=((900.0, 15000.0),)),
        ),
        required_line_ids=(), coverage_mode="component_adaptive",
        qa_labels=("padelta", "hei_10833", "pagamma", "oi_11290", "pabeta"),
        auto_enabled=True, priority=40, backend="generic", exclusive_group="paschen_nir",
    ),
    ComplexRecipe(
        id="generic_narrow_lines", aliases=("narrow_lines",), label="Generic narrow lines",
        fit_window=(0.0, 1.0), fit_windows=(), mask_windows=(), components=(),
        required_line_ids=(), coverage_mode="component_adaptive", auto_enabled=False,
        priority=0, backend="generic",
    ),
    ComplexRecipe(
        id="civ", aliases=("civ1549",), label="C IV",
        fit_window=(1450.0, 1700.0), fit_windows=((1450.0, 1700.0),), mask_windows=(),
        components=(
            _component("CIV_broad", ("civ_blend",), "broad", kinematic_group="civ_broad",
                       velocity_bounds_kms=(-5000.0, 3000.0), fwhm_bands_kms=((900.0, 20000.0),)),
            _component(
                "CIV_blue", ("civ_blend",), "wing", enabled=False,
                kinematic_group="civ_blue", velocity_bounds_kms=(-8000.0, 0.0),
                fwhm_bands_kms=((900.0, 15000.0),),
                selection_rule="specialized_outflow_deferred",
            ),
        ),
        required_line_ids=("civ_blend",), qa_labels=("civ_blend",), auto_enabled=False,
        priority=80, backend="generic", exclusive_group="civ",
    ),
)

_BY_ID: Dict[str, ComplexRecipe] = {recipe.id: recipe for recipe in _RECIPES}
_ALIASES: Dict[str, str] = {}
for _recipe in _RECIPES:
    for _alias in (_recipe.id, _recipe.label, *_recipe.aliases):
        _ALIASES[_normalize(_alias)] = _recipe.id


def list_complexes() -> Tuple[ComplexRecipe, ...]:
    return tuple(sorted(_RECIPES, key=lambda recipe: (-recipe.priority, recipe.id)))


def resolve(value: str) -> str:
    key = _normalize(value)
    if key not in _ALIASES:
        raise ValueError(f"unknown_complex_recipe: {value!r}")
    return _ALIASES[key]


def get(value: str) -> ComplexRecipe:
    return _BY_ID[resolve(value)]


def describe(value: str) -> Dict[str, Any]:
    recipe = get(value)
    return {
        "id": recipe.id,
        "label": recipe.label,
        "aliases": recipe.aliases,
        "fit_window": recipe.fit_window,
        "fit_windows": recipe.fit_windows,
        "components": tuple(component.id for component in recipe.components),
        "required_line_ids": recipe.required_line_ids,
        "auto_enabled": recipe.auto_enabled,
        "priority": recipe.priority,
        "backend": recipe.backend,
    }


def generic_narrow_lines(line_ids: Iterable[str], **changes: Any) -> ComplexRecipe:
    canonical = tuple(lines.resolve(line_id) for line_id in line_ids)
    if not canonical:
        raise ValueError("generic_narrow_lines requires at least one line ID.")
    wavelengths = tuple(lines.get(line_id).vacuum_wavelength for line_id in canonical)
    padding = float(changes.pop("padding_angstrom", 50.0))
    components = tuple(
        _component(
            f"{line_id}_narrow",
            (line_id,),
            "narrow",
            kinematic_group="generic_narrow",
        )
        for line_id in canonical
    )
    base = get("generic_narrow_lines")
    return replace(
        base,
        id="generic_narrow_lines_" + "_".join(canonical),
        label=" / ".join(lines.get(line_id).label for line_id in canonical),
        fit_window=(min(wavelengths) - padding, max(wavelengths) + padding),
        fit_windows=((min(wavelengths) - padding, max(wavelengths) + padding),),
        components=components,
        required_line_ids=canonical,
        qa_labels=canonical,
        **changes,
    )
