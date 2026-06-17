"""pPXF template loading for externally installed SPS template bundles."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json

import numpy as np


SAMPLE_WAVELENGTHS = {
    "fHost_4000": 4000.0,
    "fHost_5100": 5100.0,
    "fHost_8000": 8000.0,
    "fHost_1um": 10000.0,
    "fHost_1p6um": 16000.0,
    "fHost_2p2um": 22000.0,
}


@dataclass
class PPXFTemplateLibrary:
    """Stellar templates loaded from a local pPXF-compatible NPZ file."""

    flux: np.ndarray
    wave: np.ndarray
    log_wave: np.ndarray
    family: str
    source_path: str
    wavelength_coverage: Tuple[float, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    original_shape: Tuple[int, ...] = field(default_factory=tuple)
    warnings: List[str] = field(default_factory=list)

    @property
    def n_templates(self) -> int:
        return int(self.flux.shape[1])


def _is_increasing_wave(arr: np.ndarray) -> bool:
    arr = np.asarray(arr)
    if arr.ndim != 1 or arr.size < 10 or not np.issubdtype(arr.dtype, np.number):
        return False
    finite = np.isfinite(arr)
    if np.sum(finite) < 10:
        return False
    vals = arr[finite]
    return bool(np.nanmin(vals) > 0 and np.all(np.diff(vals) > 0))


def _find_wave_key(npz: Any) -> str:
    preferred = ("lam", "lambda", "wave", "wavelength", "wavelengths")
    for key in preferred:
        if key in npz.files and _is_increasing_wave(npz[key]):
            return key
    for key in npz.files:
        if _is_increasing_wave(npz[key]):
            return key
    raise ValueError(f"Could not identify a wavelength grid in NPZ keys: {npz.files}")


def _find_template_key(npz: Any, n_wave: int) -> str:
    preferred = ("templates", "spectra", "flux", "ssp", "models")
    for key in preferred:
        if key in npz.files:
            arr = np.asarray(npz[key])
            if arr.ndim >= 2 and n_wave in arr.shape:
                return key
    for key in npz.files:
        arr = np.asarray(npz[key])
        if arr.ndim >= 2 and np.issubdtype(arr.dtype, np.number) and n_wave in arr.shape:
            return key
    raise ValueError(f"Could not identify template spectra in NPZ keys: {npz.files}")


def _flatten_templates(templates: np.ndarray, wave: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...]]:
    arr = np.asarray(templates, dtype=float)
    n_wave = len(wave)
    if arr.shape[0] == n_wave:
        original_shape = tuple(arr.shape)
        return arr.reshape(n_wave, -1), original_shape
    axis = [i for i, size in enumerate(arr.shape) if size == n_wave]
    if not axis:
        raise ValueError(f"Template array shape {arr.shape} is incompatible with wavelength length {n_wave}")
    arr = np.moveaxis(arr, axis[0], 0)
    original_shape = tuple(arr.shape)
    return arr.reshape(n_wave, -1), original_shape


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size <= 100:
            return value.tolist()
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "min": float(np.nanmin(value)) if np.issubdtype(value.dtype, np.number) else None,
            "max": float(np.nanmax(value)) if np.issubdtype(value.dtype, np.number) else None,
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _coverage_warnings(wave_min: float, wave_max: float) -> List[str]:
    warnings = []
    if wave_max < 10000.0:
        warnings.append("template_does_not_cover_nir")
    if wave_max < 22000.0:
        warnings.append("template_wavelength_coverage_insufficient")
    if wave_max < 10000.0:
        warnings.append("nir_extrapolation_not_available")
    return warnings


def _write_reports(report_dir: Path, report_name: str, payload: Dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{report_name}.json"
    md_path = report_dir / f"{report_name}.md"
    json_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        f"# pPXF Template Inspection: {payload['template_family']}",
        "",
        f"- Source file: `{payload['source_path']}`",
        f"- Keys: {', '.join(payload['keys'])}",
        f"- Wavelength key: `{payload['wavelength_key']}`",
        f"- Template key: `{payload['template_key']}`",
        f"- Coverage: {payload['wavelength_coverage'][0]:.2f} - {payload['wavelength_coverage'][1]:.2f} Angstrom",
        f"- Template matrix shape: {payload['template_shape']}",
        f"- Flattened template count: {payload['n_templates']}",
        f"- Warnings: {', '.join(payload['warnings']) if payload['warnings'] else 'none'}",
        "",
        "## Metadata",
    ]
    for key, value in payload["metadata"].items():
        lines.append(f"- `{key}`: {_json_safe(value)}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_ppxf_npz_templates(
    template_root: str = "~/tools/ppxf_data",
    template_file: str = "spectra_emiles_9.0.npz",
    report_dir: str = "outputs/ppxf_qsofitmore",
    template_family: str = "emiles",
    write_report: bool = True,
) -> PPXFTemplateLibrary:
    """Load externally installed pPXF NPZ templates without vendoring data."""

    root = Path(template_root).expanduser()
    source = root / template_file
    if not source.exists():
        raise FileNotFoundError(
            f"pPXF template file not found: {source}. Install/download templates locally "
            "and pass template_root/template_file, e.g. ~/tools/ppxf_data/spectra_emiles_9.0.npz."
        )

    with np.load(source, allow_pickle=True) as npz:
        wave_key = _find_wave_key(npz)
        wave = np.asarray(npz[wave_key], dtype=float)
        template_key = _find_template_key(npz, len(wave))
        flux, original_shape = _flatten_templates(npz[template_key], wave)
        metadata = {
            key: np.asarray(npz[key])
            for key in npz.files
            if key not in (wave_key, template_key)
        }
        payload = {
            "template_family": template_family,
            "source_path": str(source),
            "keys": list(npz.files),
            "wavelength_key": wave_key,
            "template_key": template_key,
            "wavelength_coverage": [float(np.nanmin(wave)), float(np.nanmax(wave))],
            "template_shape": list(original_shape),
            "n_templates": int(flux.shape[1]),
            "metadata": metadata,
            "warnings": _coverage_warnings(float(np.nanmin(wave)), float(np.nanmax(wave))),
        }

    if write_report:
        _write_reports(Path(report_dir), f"template_inspection_{template_family}", payload)

    return PPXFTemplateLibrary(
        flux=flux,
        wave=wave,
        log_wave=np.log(wave),
        family=template_family,
        source_path=str(source),
        wavelength_coverage=(float(np.nanmin(wave)), float(np.nanmax(wave))),
        metadata=metadata,
        original_shape=original_shape,
        warnings=list(payload["warnings"]),
    )
