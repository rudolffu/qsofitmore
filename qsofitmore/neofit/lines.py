"""Immutable vacuum-wavelength registry used by global line recipes."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Optional, Tuple


def _normalize(value: str) -> str:
    text = str(value).strip().lower()
    replacements = {
        "α": "alpha",
        "β": "beta",
        "γ": "gamma",
        "δ": "delta",
        "λ": "",
        "å": "a",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return re.sub(r"[^a-z0-9]+", "", text)


@dataclass(frozen=True)
class LineDefinition:
    id: str
    aliases: Tuple[str, ...]
    label: str
    vacuum_wavelength: float
    transition_type: str
    default_roles: Tuple[str, ...]
    default_profile: str = "gaussian"
    blend_members: Tuple[str, ...] = ()
    notes: Tuple[str, ...] = ()
    reference: Optional[str] = None


_NIST = "NIST Atomic Spectra Database (vacuum wavelength)"
_NIST_APPROX = "NIST Atomic Spectra Database; rounded vacuum wavelength"


def _line(
    id: str,
    wavelength: float,
    label: str,
    transition_type: str,
    *,
    aliases=(),
    roles=("narrow",),
    blend_members=(),
    notes=(),
    reference=_NIST,
) -> LineDefinition:
    return LineDefinition(
        id=id,
        aliases=tuple(aliases),
        label=label,
        vacuum_wavelength=float(wavelength),
        transition_type=transition_type,
        default_roles=tuple(roles),
        blend_members=tuple(blend_members),
        notes=tuple(notes),
        reference=reference,
    )


_DEFINITIONS = (
    _line("lya_1216", 1215.67, "Lyα", "recombination", aliases=("lya", "lyalpha"), roles=("broad", "narrow")),
    _line("nv_1239", 1238.82, "N V 1239", "permitted", aliases=("nv1238",)),
    _line("nv_1243", 1242.80, "N V 1243", "permitted", aliases=("nv1242",)),
    _line("siiv_1394", 1393.76, "Si IV 1394", "permitted", aliases=("siiv1393",)),
    _line("oiv_1401", 1401.16, "O IV] 1401", "forbidden", aliases=("oiv1401", "oiv]1401")),
    _line("siiv_1403", 1402.77, "Si IV 1403", "permitted", aliases=("siiv1402",)),
    _line("civ_1548", 1548.20, "C IV 1548", "permitted", aliases=("civ1548",), roles=("broad", "narrow")),
    _line("civ_1551", 1550.77, "C IV 1551", "permitted", aliases=("civ1550",), roles=("broad", "narrow")),
    _line("civ_blend", 1549.06, "C IV", "blend", aliases=("civ", "civ1549"), roles=("broad",), blend_members=("civ_1548", "civ_1551")),
    _line("heii_1640", 1640.42, "He II", "recombination", aliases=("heii1640",), roles=("broad", "narrow")),
    _line("ciii_1909", 1908.73, "C III]", "forbidden", aliases=("ciii1909", "ciii]1909")),
    _line("mgii_2796", 2796.35, "Mg II 2796", "permitted", aliases=("mgii2796",), roles=("broad",)),
    _line("mgii_2804", 2803.53, "Mg II 2804", "permitted", aliases=("mgii2803",), roles=("broad",)),
    _line("mgii_blend", 2798.75, "Mg II", "blend", aliases=("mgii", "mgii2798"), roles=("broad",), blend_members=("mgii_2796", "mgii_2804")),
    _line("nev_3427", 3426.84, "[Ne V] 3427", "forbidden", aliases=("nev3426", "nev3427")),
    _line("oii_3727", 3727.09, "[O II] 3727", "forbidden", aliases=("oii3726", "oii3727")),
    _line("oii_3730", 3729.88, "[O II] 3730", "forbidden", aliases=("oii3729", "oii3730")),
    _line("oii_blend", 3728.48, "[O II]", "blend", aliases=("oii", "oii3728"), blend_members=("oii_3727", "oii_3730")),
    _line("neiii_3870", 3869.86, "[Ne III] 3870", "forbidden", aliases=("neiii3869", "neiii3870")),
    _line("hdelta", 4102.93, "Hδ", "recombination", aliases=("hd", "hdelta4103"), roles=("broad", "narrow")),
    _line("hgamma", 4341.68, "Hγ", "recombination", aliases=("hg", "hgamma4342"), roles=("broad", "narrow")),
    _line("heii_4687", 4687.02, "He II", "recombination", aliases=("heii4686", "heii4687"), roles=("broad", "narrow")),
    _line("hbeta", 4862.68, "Hβ", "recombination", aliases=("hb", "hbeta4863", "hbeta4861"), roles=("broad", "narrow")),
    _line("oiii_4960", 4960.30, "[O III] 4960", "forbidden", aliases=("oiii4959", "oiii4960")),
    _line("oiii_5008", 5008.24, "[O III] 5008", "forbidden", aliases=("oiii5007", "oiii5008", "OIII5007")),
    _line("hei_5877", 5877.25, "He I", "permitted", aliases=("hei5876", "hei5877"), roles=("broad", "narrow")),
    _line("nii_6550", 6549.85, "[N II] 6550", "forbidden", aliases=("nii6549", "nii6550")),
    _line("halpha", 6564.61, "Hα", "recombination", aliases=("ha", "halpha6565", "halpha6563"), roles=("broad", "narrow")),
    _line("nii_6585", 6585.28, "[N II] 6585", "forbidden", aliases=("nii6583", "nii6585")),
    _line("sii_6718", 6718.29, "[S II] 6718", "forbidden", aliases=("sii6716", "sii6718")),
    _line("sii_6733", 6732.67, "[S II] 6733", "forbidden", aliases=("sii6731", "sii6733")),
    _line("oi_8449", 8448.68, "O I", "permitted", aliases=("oi8446", "oi8449"), roles=("broad", "narrow")),
    _line("siii_9071", 9071.1, "[S III] 9071", "forbidden", aliases=("siii9069", "siii9071"), reference=_NIST_APPROX),
    _line("siii_9533", 9533.2, "[S III] 9533", "forbidden", aliases=("siii9531", "siii9533"), reference=_NIST_APPROX),
    _line("padelta", 10052.1, "Paδ", "recombination", aliases=("pad", "padelta10052"), roles=("broad", "narrow"), reference=_NIST_APPROX),
    _line("hei_10833", 10833.3, "He I", "permitted", aliases=("hei10830", "hei10833"), roles=("broad", "narrow"), reference=_NIST_APPROX),
    _line("pagamma", 10941.1, "Paγ", "recombination", aliases=("pag", "pagamma10941"), roles=("broad", "narrow"), reference=_NIST_APPROX),
    _line("oi_11290", 11290.0, "O I", "permitted", aliases=("oi11287", "oi11290"), roles=("broad", "narrow"), reference=_NIST_APPROX),
    _line("pabeta", 12821.6, "Paβ", "recombination", aliases=("pab", "pabeta12822"), roles=("broad", "narrow"), reference=_NIST_APPROX),
)

_BY_ID: Dict[str, LineDefinition] = {item.id: item for item in _DEFINITIONS}
_ALIASES: Dict[str, str] = {}
for _definition in _DEFINITIONS:
    for _alias in (_definition.id, _definition.label, *_definition.aliases):
        _ALIASES[_normalize(_alias)] = _definition.id


def list() -> Tuple[LineDefinition, ...]:
    """Return all registered line definitions in wavelength order."""

    return tuple(sorted(_DEFINITIONS, key=lambda item: item.vacuum_wavelength))


def resolve(value: str) -> str:
    """Resolve a canonical line ID from a normalized or historical alias."""

    key = _normalize(value)
    if key not in _ALIASES:
        raise ValueError(f"unknown_line_id: {value!r}")
    return _ALIASES[key]


def get(value: str) -> LineDefinition:
    """Return one immutable line definition."""

    return _BY_ID[resolve(value)]
