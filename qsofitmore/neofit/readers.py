"""Shared spectrum readers and efficient batch input discovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.dataset as pads
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u

from qsofitmore.host_decomp.io import (
    DEC_ALIASES,
    ERR_ALIASES,
    FLUX_ALIASES,
    IVAR_ALIASES,
    MASK_ALIASES,
    OBJECT_ID_ALIASES,
    RA_ALIASES,
    REDSHIFT_ALIASES,
    WAVE_ALIASES,
    SpectrumData,
    read_sparcli_spectrum,
)


@dataclass(frozen=True)
class SpectrumInput:
    """One deterministic input locator used by batch fitting."""

    source: str
    row_index: Optional[int] = None
    object_id: Optional[str] = None
    redshift: Optional[float] = None
    reader: str = "auto"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def object_key(self) -> str:
        locator = (
            f"{Path(self.source).expanduser().resolve()}#"
            f"{self.row_index if self.row_index is not None else ''}"
        )
        return locator


def _lookup(columns: Iterable[str], aliases: Iterable[str]) -> Optional[str]:
    normalized = {str(column).strip().lower(): str(column) for column in columns}
    for alias in aliases:
        if alias.lower() in normalized:
            return normalized[alias.lower()]
    return None


def _value(row: Mapping[str, Any], column: Optional[str]) -> Any:
    if column is None:
        return None
    value = row.get(column)
    if hasattr(value, "as_py"):
        value = value.as_py()
    return value


def _array(row: Mapping[str, Any], column: Optional[str]) -> Optional[np.ndarray]:
    value = _value(row, column)
    if value is None:
        return None
    return np.asarray(value)


def spectrum_data_from_mapping(
    row: Mapping[str, Any],
    *,
    source: str,
    row_index: Optional[int] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> SpectrumData:
    """Normalize one Arrow/Python mapping into :class:`SpectrumData`."""

    columns = tuple(row)
    wave_col = _lookup(columns, WAVE_ALIASES)
    flux_col = _lookup(columns, FLUX_ALIASES)
    if wave_col is None or flux_col is None:
        raise ValueError(
            f"Could not identify wavelength/flux columns in {source}: {columns}"
        )
    overrides = dict(overrides or {})
    ivar_col = _lookup(columns, IVAR_ALIASES)
    error_col = _lookup(columns, ERR_ALIASES)
    mask_col = _lookup(columns, MASK_ALIASES)
    redshift_col = _lookup(columns, REDSHIFT_ALIASES)
    object_col = _lookup(columns, OBJECT_ID_ALIASES)
    target_col = _lookup(columns, ("targetid", "target_id"))
    ra_col = _lookup(columns, RA_ALIASES)
    dec_col = _lookup(columns, DEC_ALIASES)
    redshift = overrides.get("redshift", _value(row, redshift_col))
    object_id = overrides.get("object_id", _value(row, object_col))
    targetid = _value(row, target_col)
    return SpectrumData(
        wave_obs=np.asarray(_array(row, wave_col), dtype=float),
        flux=np.asarray(_array(row, flux_col), dtype=float),
        ivar=(
            np.asarray(_array(row, ivar_col), dtype=float)
            if ivar_col is not None else None
        ),
        error=(
            np.asarray(_array(row, error_col), dtype=float)
            if error_col is not None else None
        ),
        mask=(
            np.asarray(_array(row, mask_col))
            if mask_col is not None else None
        ),
        redshift=float(redshift) if redshift is not None else None,
        object_id=str(object_id) if object_id is not None else None,
        targetid=str(targetid) if targetid is not None else None,
        ra=(
            float(overrides.get("ra", _value(row, ra_col)))
            if overrides.get("ra", _value(row, ra_col)) is not None else None
        ),
        dec=(
            float(overrides.get("dec", _value(row, dec_col)))
            if overrides.get("dec", _value(row, dec_col)) is not None else None
        ),
        metadata={
            "input_file": str(source),
            "file_type": "parquet",
            "row_index": row_index,
            "selected_columns": {
                "wavelength": wave_col,
                "flux": flux_col,
                "ivar": ivar_col,
                "error": error_col,
                "mask": mask_col,
                "redshift": redshift_col,
                "object_id": object_col,
                "ra": ra_col,
                "dec": dec_col,
            },
        },
    )


def scan_parquet_spectra(
    paths: Sequence[str] | str,
    *,
    row_indices: Optional[Mapping[str, Sequence[int]] | Sequence[int]] = None,
    filter_expression: Any = None,
    batch_size: int = 128,
) -> Iterator[Tuple[SpectrumInput, SpectrumData]]:
    """Scan Parquet sources once with projected columns and bounded batches."""

    source_paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    for source in source_paths:
        source_path = str(Path(source).expanduser())
        dataset = pads.dataset(source_path, format="parquet")
        columns = dataset.schema.names
        selected = {
            column
            for aliases in (
                WAVE_ALIASES,
                FLUX_ALIASES,
                IVAR_ALIASES,
                ERR_ALIASES,
                MASK_ALIASES,
                REDSHIFT_ALIASES,
                OBJECT_ID_ALIASES,
                RA_ALIASES,
                DEC_ALIASES,
                ("targetid", "target_id"),
            )
            if (column := _lookup(columns, aliases)) is not None
        }
        requested = None
        if row_indices is not None:
            if isinstance(row_indices, Mapping):
                requested = set(map(int, row_indices.get(source, ())))
                requested.update(map(int, row_indices.get(source_path, ())))
            else:
                requested = set(map(int, row_indices))
        scanner = dataset.scanner(
            columns=sorted(selected),
            filter=filter_expression,
            batch_size=int(batch_size),
        )
        absolute_index = 0
        for record_batch in scanner.to_batches():
            rows = record_batch.to_pylist()
            for offset, row in enumerate(rows):
                row_index = absolute_index + offset
                if requested is not None and row_index not in requested:
                    continue
                spectrum = spectrum_data_from_mapping(
                    row,
                    source=source_path,
                    row_index=row_index,
                )
                descriptor = SpectrumInput(
                    source=source_path,
                    row_index=row_index,
                    object_id=spectrum.object_id or spectrum.targetid,
                    redshift=spectrum.redshift,
                    reader="parquet",
                )
                yield descriptor, spectrum
            absolute_index += len(rows)


def _header_float(header, *names) -> Optional[float]:
    for name in names:
        value = header.get(name)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _coordinates(header) -> Tuple[Optional[float], Optional[float]]:
    ra = _header_float(header, "PLUG_RA", "RAOBJ", "RA")
    dec = _header_float(header, "PLUG_DEC", "DECOBJ", "DEC")
    if ra is not None and dec is not None:
        return ra, dec
    try:
        coordinate = SkyCoord(
            str(header["RA"]) + str(header["DEC"]),
            unit=(u.hourangle, u.deg),
        )
        return float(coordinate.ra.deg), float(coordinate.dec.deg)
    except Exception:
        return None, None


def _read_sdss(path: Path, redshift=None, object_id=None) -> SpectrumData:
    with fits.open(path, memmap=False) as hdul:
        table = hdul[1].data
        header = hdul[0].header
        names = {name.lower(): name for name in table.names}
        wave = 10.0 ** np.asarray(table[names["loglam"]], dtype=float)
        flux = np.asarray(table[names["flux"]], dtype=float)
        ivar = (
            np.asarray(table[names["ivar"]], dtype=float)
            if "ivar" in names else None
        )
        mask = None
        if "and_mask" in names or "or_mask" in names:
            mask = np.ones_like(flux, dtype=bool)
            if "and_mask" in names:
                mask &= np.asarray(table[names["and_mask"]]) == 0
            if "or_mask" in names:
                mask &= np.asarray(table[names["or_mask"]]) == 0
        if redshift is None and len(hdul) > 2 and hdul[2].data is not None:
            z_names = {name.lower(): name for name in hdul[2].data.names or ()}
            if "z" in z_names:
                redshift = float(hdul[2].data[z_names["z"]][0])
        ra, dec = _coordinates(header)
        identity = object_id or header.get("THING_ID") or path.stem
    return SpectrumData(
        wave_obs=wave,
        flux=flux,
        ivar=ivar,
        mask=mask,
        redshift=float(redshift or 0.0),
        object_id=str(identity),
        ra=ra,
        dec=dec,
        metadata={"input_file": str(path), "file_type": "sdss_fits"},
    )


def _read_lamost(path: Path, redshift=None, object_id=None) -> SpectrumData:
    with fits.open(path, memmap=False) as hdul:
        header = hdul[0].header
        data = hdul[1].data if len(hdul) > 1 and hdul[1].data is not None else None
        if data is not None and getattr(data, "names", None):
            names = {name.lower(): name for name in data.names}
            wave_name = next(
                (names[alias] for alias in WAVE_ALIASES if alias in names),
                names.get("loglam"),
            )
            flux_name = next(
                (names[alias] for alias in FLUX_ALIASES if alias in names),
                None,
            )
            if wave_name is None or flux_name is None:
                raise ValueError("LAMOST table lacks wavelength or flux columns.")
            wave = np.asarray(data[wave_name], dtype=float)
            if str(wave_name).lower() == "loglam":
                wave = 10.0 ** wave
            flux = np.asarray(data[flux_name], dtype=float)
            ivar_name = next(
                (names[alias] for alias in IVAR_ALIASES if alias in names),
                None,
            )
            error_name = next(
                (names[alias] for alias in ERR_ALIASES if alias in names),
                None,
            )
            ivar = np.asarray(data[ivar_name], dtype=float) if ivar_name else None
            error = (
                np.asarray(data[error_name], dtype=float)
                if error_name else None
            )
        else:
            array = np.asarray(hdul[0].data)
            flux = np.asarray(array[0] if array.ndim > 1 else array, dtype=float)
            ivar = (
                np.asarray(array[1], dtype=float)
                if array.ndim > 1 and array.shape[0] > 1 else None
            )
            error = None
            coeff0 = _header_float(header, "COEFF0")
            coeff1 = _header_float(header, "COEFF1")
            if coeff0 is not None and coeff1 is not None:
                wave = 10.0 ** (coeff0 + coeff1 * np.arange(flux.size))
            else:
                wave = _linear_wcs(header, flux.size)
        redshift = (
            redshift
            if redshift is not None
            else _header_float(header, "Z", "REDSHIFT")
        )
        ra, dec = _coordinates(header)
    return SpectrumData(
        wave_obs=wave,
        flux=flux,
        ivar=ivar,
        error=error,
        redshift=float(redshift or 0.0),
        object_id=str(object_id or header.get("OBSID") or path.stem),
        ra=ra,
        dec=dec,
        metadata={"input_file": str(path), "file_type": "lamost_fits"},
    )


def _linear_wcs(header, size: int) -> np.ndarray:
    crval = _header_float(header, "CRVAL1")
    step = _header_float(header, "CD1_1", "CDELT1")
    crpix = _header_float(header, "CRPIX1")
    if crval is None or step is None:
        raise ValueError("IRAF spectrum lacks linear wavelength WCS.")
    crpix = 1.0 if crpix is None else crpix
    start = (1.0 - crpix) * step + crval
    wave = start + step * np.arange(size)
    if int(header.get("DC-FLAG", 0) or 0) == 1:
        wave = 10.0 ** wave
    return wave


def _read_iraf(path: Path, redshift=None, object_id=None) -> SpectrumData:
    with fits.open(path, memmap=False) as hdul:
        header = hdul[0].header
        data = np.asarray(hdul[0].data)
        if data.ndim == 1:
            flux = np.asarray(data, dtype=float)
            error = None
        else:
            flattened = data.reshape((-1, data.shape[-1]))
            flux = np.asarray(flattened[0], dtype=float)
            error = (
                np.asarray(flattened[3], dtype=float)
                if flattened.shape[0] > 3 else None
            )
        wave = _linear_wcs(header, flux.size)
        redshift = (
            redshift
            if redshift is not None
            else _header_float(header, "REDSHIFT", "Z")
        )
        ra, dec = _coordinates(header)
        identity = object_id or header.get("OBJECT") or path.stem
    return SpectrumData(
        wave_obs=wave,
        flux=flux,
        error=error,
        redshift=float(redshift or 0.0),
        object_id=str(identity),
        ra=ra,
        dec=dec,
        metadata={"input_file": str(path), "file_type": "iraf_fits"},
    )


def detect_fits_reader(path: str) -> str:
    """Return ``sdss``, ``lamost``, or ``iraf`` from FITS structure."""

    with fits.open(Path(path).expanduser(), memmap=False) as hdul:
        header = hdul[0].header
        if len(hdul) > 1 and getattr(hdul[1].data, "names", None):
            names = {name.lower() for name in hdul[1].data.names}
            if {"loglam", "flux"}.issubset(names):
                return "sdss"
            if "flux" in names and (
                "wavelength" in names or "wave" in names or "loglam" in names
            ):
                return "lamost"
        if "LAMOST" in str(header.get("TELESCOP", "")).upper():
            return "lamost"
    return "iraf"


def read_spectrum(
    source: str,
    *,
    row_index: Optional[int] = None,
    redshift: Optional[float] = None,
    object_id: Optional[str] = None,
    reader: str = "auto",
) -> SpectrumData:
    """Read a Parquet/SPARCL, SDSS, LAMOST, or IRAF spectrum."""

    path = Path(source).expanduser()
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return read_sparcli_spectrum(
            str(path),
            row_index=row_index,
            redshift=redshift,
            object_id=object_id,
        )
    selected = detect_fits_reader(str(path)) if reader == "auto" else reader
    readers = {"sdss": _read_sdss, "lamost": _read_lamost, "iraf": _read_iraf}
    if selected not in readers:
        raise ValueError(f"Unknown spectrum reader: {selected!r}")
    return readers[selected](path, redshift=redshift, object_id=object_id)


def discover_fits_inputs(
    sources: Sequence[str] | str,
    *,
    recursive: bool = True,
) -> Tuple[SpectrumInput, ...]:
    """Expand FITS files, directories, and glob patterns deterministically."""

    items = [sources] if isinstance(sources, (str, Path)) else list(sources)
    paths = []
    for item in items:
        expanded = Path(item).expanduser()
        if expanded.is_dir():
            pattern = "**/*" if recursive else "*"
            paths.extend(
                path
                for path in expanded.glob(pattern)
                if path.is_file() and (
                    path.suffix.lower() in (".fits", ".fit", ".fz")
                    or path.name.lower().endswith(".fits.gz")
                )
            )
        elif any(token in str(item) for token in "*?[]"):
            paths.extend(Path(path) for path in glob(str(expanded), recursive=recursive))
        else:
            paths.append(expanded)
    return tuple(
        SpectrumInput(source=str(path.resolve()), object_id=path.stem)
        for path in sorted(set(paths), key=lambda value: str(value))
    )


def read_input_manifest(path: str) -> Tuple[SpectrumInput, ...]:
    """Read a CSV/Parquet manifest containing paths and optional overrides."""

    manifest_path = Path(path).expanduser()
    table = (
        pd.read_parquet(manifest_path)
        if manifest_path.suffix.lower() == ".parquet"
        else pd.read_csv(manifest_path)
    )
    path_column = next(
        (
            name
            for name in ("source", "input", "input_file", "path")
            if name in table.columns
        ),
        None,
    )
    if path_column is None:
        raise ValueError(
            "Input manifest requires source, input, input_file, or path."
        )
    descriptors = []
    for _, row in table.iterrows():
        source_path = Path(str(row[path_column])).expanduser()
        if not source_path.is_absolute():
            source_path = manifest_path.parent / source_path
        source = str(source_path)
        reader = (
            str(row["reader"])
            if "reader" in table.columns and pd.notna(row["reader"])
            else "auto"
        )
        descriptors.append(
            SpectrumInput(
                source=source,
                row_index=(
                    int(row["row_index"])
                    if "row_index" in table.columns
                    and pd.notna(row["row_index"])
                    else None
                ),
                object_id=(
                    str(row["object_id"])
                    if "object_id" in table.columns
                    and pd.notna(row["object_id"])
                    else None
                ),
                redshift=(
                    float(row["redshift"])
                    if "redshift" in table.columns
                    and pd.notna(row["redshift"])
                    else float(row["z"])
                    if "z" in table.columns and pd.notna(row["z"])
                    else None
                ),
                reader=reader,
                metadata={
                    str(key): value
                    for key, value in row.items()
                    if key not in {
                        path_column,
                        "row_index",
                        "object_id",
                        "redshift",
                        "z",
                        "reader",
                    }
                    and pd.notna(value)
                },
            )
        )
    return tuple(descriptors)
