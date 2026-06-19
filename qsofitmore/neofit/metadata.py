"""Spectrum metadata and unit/survey preset handling for neofit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional


_CGS_1E17_UNIT = "1e-17 erg s^-1 cm^-2 Angstrom^-1"
_SURVEY_ALIASES = {
    "desi": "desi",
    "desidr1": "desi",
    "desi-dr1": "desi",
    "desi_dr1": "desi",
    "desiedr": "desi",
    "desi-edr": "desi",
    "desi_edr": "desi",
    "sdss": "sdss",
}
_UNIT_ALIASES = {
    "input": "input",
    "1e-17cgs": "1e-17cgs",
    "1e17cgs": "1e-17cgs",
    "1e-17_cgs": "1e-17cgs",
    "1e-17 cgs": "1e-17cgs",
}


@dataclass
class SpectrumMetadata:
    """Wavelength and flux-density metadata kept outside numerical fitting."""

    wave_unit: str = "Angstrom"
    flux_density_unit: str = "input"
    flux_density_scale_to_cgs: Optional[float] = None
    survey: Optional[str] = None
    source: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly dictionary."""

        return {
            "wave_unit": self.wave_unit,
            "flux_density_unit": self.flux_density_unit,
            "flux_density_scale_to_cgs": self.flux_density_scale_to_cgs,
            "survey": self.survey,
            "source": self.source,
            "notes": list(self.notes),
        }


def _normalize_survey(survey: Optional[str]) -> Optional[str]:
    if survey is None:
        return None
    key = str(survey).strip().lower().replace(" ", "").replace(".", "")
    if key not in _SURVEY_ALIASES:
        raise ValueError(f"Unknown neofit survey preset: {survey!r}")
    return _SURVEY_ALIASES[key]


def _normalize_unit_preset(unit_preset: Optional[str]) -> Optional[str]:
    if unit_preset is None:
        return None
    key = str(unit_preset).strip().lower()
    if key not in _UNIT_ALIASES:
        raise ValueError(f"Unknown neofit unit preset: {unit_preset!r}")
    return _UNIT_ALIASES[key]


def _metadata_from_base(metadata: Optional[Any]) -> SpectrumMetadata:
    if metadata is None:
        return SpectrumMetadata()
    if isinstance(metadata, SpectrumMetadata):
        return SpectrumMetadata(
            wave_unit=metadata.wave_unit,
            flux_density_unit=metadata.flux_density_unit,
            flux_density_scale_to_cgs=metadata.flux_density_scale_to_cgs,
            survey=metadata.survey,
            source=metadata.source,
            notes=list(metadata.notes),
        )
    if isinstance(metadata, Mapping):
        return SpectrumMetadata(
            wave_unit=str(metadata.get("wave_unit", "Angstrom")),
            flux_density_unit=str(metadata.get("flux_density_unit", "input")),
            flux_density_scale_to_cgs=metadata.get("flux_density_scale_to_cgs"),
            survey=metadata.get("survey"),
            source=metadata.get("source"),
            notes=list(metadata.get("notes", [])),
        )
    raise TypeError("metadata must be a SpectrumMetadata, mapping, or None.")


def resolve_spectrum_metadata(
    *,
    survey: Optional[str] = None,
    unit_preset: Optional[str] = None,
    wave_unit: Optional[str] = None,
    flux_density_unit: Optional[str] = None,
    flux_density_scale_to_cgs: Optional[float] = None,
    source: Optional[str] = None,
    metadata: Optional[Any] = None,
) -> SpectrumMetadata:
    """Resolve metadata with explicit keywords taking highest priority."""

    resolved = _metadata_from_base(metadata)
    canonical_survey = _normalize_survey(survey)
    canonical_unit = _normalize_unit_preset(unit_preset)

    if canonical_survey in ("desi", "sdss"):
        resolved.wave_unit = "Angstrom"
        resolved.flux_density_unit = _CGS_1E17_UNIT
        resolved.flux_density_scale_to_cgs = 1e-17
        resolved.survey = canonical_survey

    if canonical_unit == "1e-17cgs":
        resolved.wave_unit = "Angstrom"
        resolved.flux_density_unit = _CGS_1E17_UNIT
        resolved.flux_density_scale_to_cgs = 1e-17
    elif canonical_unit == "input":
        resolved.flux_density_unit = "input"
        resolved.flux_density_scale_to_cgs = None

    if wave_unit is not None:
        resolved.wave_unit = str(wave_unit)
    if flux_density_unit is not None:
        resolved.flux_density_unit = str(flux_density_unit)
    if flux_density_scale_to_cgs is not None:
        resolved.flux_density_scale_to_cgs = float(flux_density_scale_to_cgs)
    if source is not None:
        resolved.source = str(source)

    return resolved
