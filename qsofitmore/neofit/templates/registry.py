"""Registry for bundled and external neofit iron templates."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from .iron import IronTemplate, IronTemplateError
from .normalize import area_normalize, read_two_column_template


_ALIASES: Dict[str, str] = {
    "bg92": "bg92_optical",
    "bg92_optical": "bg92_optical",
    "park22": "park22_optical",
    "park22_optical": "park22_optical",
    "veron04": "veron04_optical",
    "vc04": "veron04_optical",
    "veron04_optical": "veron04_optical",
    "vw01": "vw01_uv",
    "vw01_uv": "vw01_uv",
    "external": "external",
}

_REFERENCES: Dict[str, str] = {
    "bg92_optical": "Boroson & Green 1992 / legacy qsofitmore optical Fe II template",
    "park22_optical": "Park et al. 2022, ApJS, 258, 38",
    "veron04_optical": "Veron-Cetty, Joly & Veron 2004, A&A, 417, 515",
    "vw01_uv": "Vestergaard & Wilkes 2001 / legacy qsofitmore UV Fe template",
}

_NOTES: Dict[str, str] = {
    "bg92_optical": "Bundled area-normalized copy of qsofitmore's legacy optical template.",
    "park22_optical": "Bundled area-normalized copy generated from the provided Park22 tab1.txt.",
    "veron04_optical": "Bundled area-normalized copy generated from the provided VC04 iwz1.fit.",
    "vw01_uv": "Bundled area-normalized copy of qsofitmore's legacy UV/MgII template.",
}


def _data_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


def list_iron_templates() -> Dict[str, str]:
    """Return supported canonical template names and references."""

    return dict(_REFERENCES)


def resolve_iron_template_name(name: str) -> str:
    """Resolve a user-facing template alias to a canonical name."""

    key = str(name).strip().lower().replace("-", "_")
    if key not in _ALIASES:
        raise IronTemplateError("unknown_iron_template", f"Unknown iron template: {name!r}")
    return _ALIASES[key]


def load_iron_template(
    template: str,
    template_path: Optional[str] = None,
    normalization: str = "area",
) -> IronTemplate:
    """Load a bundled or user-provided iron template."""

    if normalization != "area":
        raise IronTemplateError("iron_template_parse_failed", "Only area-normalized iron templates are supported.")
    canonical = resolve_iron_template_name(template)
    if canonical == "external" and not template_path:
        raise IronTemplateError("missing_iron_template_path", "template_path is required for template='external'.")

    if template_path:
        try:
            wave, flux = read_two_column_template(template_path)
            wave, flux, area = area_normalize(wave, flux)
        except FileNotFoundError as exc:
            raise IronTemplateError("missing_iron_template_path", str(exc)) from exc
        except ValueError as exc:
            message = str(exc)
            code = "iron_template_not_monotonic" if "strictly increasing" in message else "iron_template_parse_failed"
            if "non-finite" in message:
                code = "iron_template_parse_failed"
            if "zero positive" in message:
                code = "iron_template_zero_norm"
            raise IronTemplateError(code, message) from exc
        source = str(Path(template_path).expanduser())
        reference = _REFERENCES.get(canonical)
        notes = [f"External template normalized by positive area {area:.6e}."]
    else:
        path = _data_dir() / f"{canonical}.txt"
        if not path.exists():
            raise IronTemplateError("missing_iron_template_path", f"Bundled iron template is missing: {path}")
        try:
            wave, flux = read_two_column_template(str(path))
            wave, flux, _ = area_normalize(wave, flux)
        except ValueError as exc:
            raise IronTemplateError("iron_template_parse_failed", str(exc)) from exc
        source = str(path)
        reference = _REFERENCES.get(canonical)
        notes = [_NOTES.get(canonical, "Bundled area-normalized iron template.")]

    return IronTemplate(
        name=canonical,
        wave_rest=wave,
        flux=flux,
        reference=reference,
        source_path=source,
        coverage=(float(wave.min()), float(wave.max())),
        notes=notes,
        normalization="area",
    )
