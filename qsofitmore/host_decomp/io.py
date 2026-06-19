"""Input and output helpers for SPARCL/DESI-like host-decomposition data."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import json

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table


WAVE_ALIASES = ("wavelength", "wave", "lambda", "lam", "obs_wave")
FLUX_ALIASES = ("flux", "flam", "flux_lambda")
IVAR_ALIASES = ("ivar", "inverse_variance", "inverse_var")
ERR_ALIASES = ("error", "err", "sigma", "flux_error")
MASK_ALIASES = ("mask", "and_mask", "or_mask")
REDSHIFT_ALIASES = ("redshift", "z", "z_desi", "z_vi")
OBJECT_ID_ALIASES = ("targetid", "target_id", "object_id", "sparcl_id", "specid")
RA_ALIASES = ("ra", "ra_deg")
DEC_ALIASES = ("dec", "dec_deg", "declination")
DEFAULT_FLUX_DENSITY_UNIT = "1e-17 erg cm^-2 s^-1 Angstrom^-1"


@dataclass
class SpectrumData:
    """Standardized local DESI/SPARCL-like spectrum."""

    wave_obs: np.ndarray
    flux: np.ndarray
    ivar: Optional[np.ndarray] = None
    error: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    redshift: Optional[float] = None
    object_id: Optional[str] = None
    targetid: Optional[str] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def uncertainty(self) -> np.ndarray:
        """Return 1-sigma uncertainty, deriving it from ivar when possible."""

        if self.error is not None:
            return np.asarray(self.error, dtype=float)
        if self.ivar is not None:
            ivar = np.asarray(self.ivar, dtype=float)
            err = np.full_like(ivar, np.inf, dtype=float)
            valid = ivar > 0
            err[valid] = 1.0 / np.sqrt(ivar[valid])
            return err
        return np.ones_like(self.flux, dtype=float)


def _normalized_columns(columns: Iterable[str]) -> Dict[str, str]:
    return {str(col).strip().lower(): str(col) for col in columns}


def _find_column(columns: Iterable[str], aliases: Iterable[str]) -> Optional[str]:
    lookup = _normalized_columns(columns)
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    return None


def _is_vector_value(value: Any) -> bool:
    if isinstance(value, (str, bytes)):
        return False
    try:
        arr = np.asarray(value)
    except Exception:
        return False
    return arr.ndim > 0 and arr.size > 1


def _extract_vector(table: Any, column: str, row_index: Optional[int] = None) -> np.ndarray:
    values = table[column]
    if isinstance(values, pd.Series):
        idx = 0 if row_index is None else int(row_index)
        if len(values) and _is_vector_value(values.iloc[idx]):
            return np.asarray(values.iloc[idx], dtype=float)
        return np.asarray(values, dtype=float)

    arr = np.asarray(values)
    idx = 0 if row_index is None else int(row_index)
    if arr.ndim >= 2:
        return np.asarray(arr[idx], dtype=float)
    if arr.dtype == object and arr.size and _is_vector_value(arr[idx]):
        return np.asarray(arr[idx], dtype=float)
    return np.asarray(arr, dtype=float)


def _extract_scalar(table: Any, column: Optional[str], row_index: Optional[int] = None) -> Any:
    if column is None:
        return None
    values = table[column]
    idx = 0 if row_index is None else int(row_index)
    try:
        if isinstance(values, pd.Series):
            value = values.iloc[idx]
        else:
            arr = np.asarray(values)
            value = arr[idx] if arr.ndim > 0 else arr.item()
        if _is_vector_value(value):
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value
    except Exception:
        return None


def _read_table_file(path: Path, row_index: Optional[int] = None) -> Tuple[Any, str, List[str]]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
        return df, "parquet", list(df.columns)
    if suffix == ".csv":
        table = Table.read(path, format="csv")
        return table, "csv", list(table.colnames)
    if suffix == ".ecsv":
        table = Table.read(path, format="ascii.ecsv")
        return table, "ecsv", list(table.colnames)
    if suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        mapping = {key: data[key] for key in data.files}
        return mapping, "npz", list(mapping.keys())
    if suffix in (".fits", ".fit", ".fz") or path.name.lower().endswith(".fits.gz"):
        with fits.open(path) as hdul:
            for hdu in hdul:
                if hasattr(hdu, "columns") and hdu.data is not None:
                    table = Table(hdu.data)
                    return table, "fits", list(table.colnames)
        raise ValueError(f"No table HDU found in FITS file: {path}")
    raise ValueError(f"Unsupported spectrum file type: {path}")


def read_sparcli_spectrum(
    path: str,
    row_index: Optional[int] = None,
    redshift: Optional[float] = None,
    object_id: Optional[str] = None,
) -> SpectrumData:
    """Read a local SPARCL/DESI-like spectrum into a standard object."""

    input_path = Path(path).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Spectrum file not found: {input_path}")

    table, file_type, columns = _read_table_file(input_path, row_index=row_index)
    wave_col = _find_column(columns, WAVE_ALIASES)
    flux_col = _find_column(columns, FLUX_ALIASES)
    if wave_col is None or flux_col is None:
        raise ValueError(
            f"Could not identify wavelength/flux columns in {input_path}. "
            f"Columns: {columns}"
        )

    ivar_col = _find_column(columns, IVAR_ALIASES)
    err_col = _find_column(columns, ERR_ALIASES)
    mask_col = _find_column(columns, MASK_ALIASES)
    z_col = _find_column(columns, REDSHIFT_ALIASES)
    obj_col = _find_column(columns, OBJECT_ID_ALIASES)
    ra_col = _find_column(columns, RA_ALIASES)
    dec_col = _find_column(columns, DEC_ALIASES)
    targetid_col = _find_column(columns, ("targetid", "target_id"))

    z_value = redshift if redshift is not None else _extract_scalar(table, z_col, row_index=row_index)
    obj_value = object_id if object_id is not None else _extract_scalar(table, obj_col, row_index=row_index)
    targetid_value = _extract_scalar(table, targetid_col, row_index=row_index)

    metadata = {
        "input_file": str(input_path),
        "file_type": file_type,
        "flux_density_unit": DEFAULT_FLUX_DENSITY_UNIT,
        "columns": list(map(str, columns)),
        "selected_columns": {
            "wavelength": wave_col,
            "flux": flux_col,
            "ivar": ivar_col,
            "error": err_col,
            "mask": mask_col,
            "redshift": z_col,
            "object_id": obj_col,
            "ra": ra_col,
            "dec": dec_col,
        },
        "row_index": row_index,
    }

    ivar = _extract_vector(table, ivar_col, row_index=row_index) if ivar_col else None
    error = _extract_vector(table, err_col, row_index=row_index) if err_col else None
    mask = _extract_vector(table, mask_col, row_index=row_index) if mask_col else None

    return SpectrumData(
        wave_obs=_extract_vector(table, wave_col, row_index=row_index),
        flux=_extract_vector(table, flux_col, row_index=row_index),
        ivar=ivar,
        error=error,
        mask=mask,
        redshift=float(z_value) if z_value is not None else None,
        object_id=str(obj_value) if obj_value is not None else None,
        targetid=str(targetid_value) if targetid_value is not None else None,
        ra=float(_extract_scalar(table, ra_col, row_index=row_index)) if ra_col else None,
        dec=float(_extract_scalar(table, dec_col, row_index=row_index)) if dec_col else None,
        metadata=metadata,
    )


def inspect_spectrum(path: str, row_index: Optional[int] = None) -> Dict[str, Any]:
    """Return a compact inspection dictionary for a local spectrum."""

    spec = read_sparcli_spectrum(path, row_index=row_index)
    err = spec.uncertainty()
    valid = np.isfinite(spec.wave_obs) & np.isfinite(spec.flux) & np.isfinite(err)
    if spec.ivar is not None:
        valid &= np.asarray(spec.ivar) > 0
    wave_valid = spec.wave_obs[np.isfinite(spec.wave_obs)]
    return {
        "detected_file_type": spec.metadata.get("file_type"),
        "column_names": spec.metadata.get("columns", []),
        "selected_columns": spec.metadata.get("selected_columns", {}),
        "wavelength_range": [
            float(np.nanmin(wave_valid)) if wave_valid.size else None,
            float(np.nanmax(wave_valid)) if wave_valid.size else None,
        ],
        "n_pixels": int(len(spec.wave_obs)),
        "n_valid_pixels": int(np.sum(valid)),
        "redshift": spec.redshift,
        "object_id": spec.object_id,
        "targetid": spec.targetid,
        "ra": spec.ra,
        "dec": spec.dec,
    }


def write_json(path: str, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
