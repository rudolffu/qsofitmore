"""
Utilities to view, edit, and convert QSO line parameter tables
between FITS, CSV, and YAML formats.

The goal is to make parameters human-editable (CSV/YAML) while
preserving the exact FITS schema expected by the fitter.

Columns supported (matching examples/1-make_parlist.ipynb):
- lambda, compname, minwav, maxwav, linename, ngauss,
  inisig, minsig, maxsig, voff, vindex, windex, findex, fvalue

Typical usage:
- Export existing FITS to CSV/YAML, edit, then convert back to FITS.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import os

import numpy as np
from astropy.io import fits
from astropy.io.fits import table_to_hdu
from astropy.table import Table


# Column names and dtypes consistent with the notebook generator
COL_NAMES = [
    "lambda", "compname", "minwav", "maxwav", "linename",
    "ngauss", "inisig", "minsig", "maxsig", "voff",
    "vindex", "windex", "findex", "fvalue",
]

COL_DTYPES = [
    "f4",   # lambda
    "U20",  # compname (unicode up to 20 chars)
    "f4",   # minwav
    "f4",   # maxwav
    "U20",  # linename (unicode up to 20 chars)
    "f4",   # ngauss
    "f4",   # inisig
    "f4",   # minsig
    "f4",   # maxsig
    "f4",   # voff
    "f4",   # vindex
    "f4",   # windex
    "f4",   # findex
    "f4",   # fvalue
]


def _ensure_table_schema(tab: Table) -> Table:
    """Ensure table has the expected columns in the expected order.

    Reorders, adds missing columns with NaN/defaults, and coerces dtypes.
    """
    # Add missing columns with NaN/empty defaults
    for name, dt in zip(COL_NAMES, COL_DTYPES):
        if name not in tab.colnames:
            if dt.startswith("U"):
                tab[name] = np.array([""] * len(tab))
            else:
                tab[name] = np.full(len(tab), np.nan, dtype=np.float32)

    # Reorder
    tab = tab[COL_NAMES]

    # Coerce dtypes
    for name, dt in zip(COL_NAMES, COL_DTYPES):
        if dt.startswith("U"):
            # astropy uses object/str internally; let it be but cast to str
            tab[name] = tab[name].astype(str)
        else:
            tab[name] = tab[name].astype(np.float32)
    return tab


def _default_header() -> fits.Header:
    hdr = fits.Header()
    hdr["lambda"] = "Vacuum Wavelength in Ang"
    hdr["minwav"] = "Lower complex fitting wavelength range"
    hdr["maxwav"] = "Upper complex fitting wavelength range"
    hdr["ngauss"] = "Number of Gaussians for the line"
    hdr["inisig"] = "Initial guess of linesigma [in lnlambda]"
    hdr["minsig"] = "Lower range of line sigma [lnlambda]"
    hdr["maxsig"] = "Upper range of line sigma [lnlambda]"
    hdr["voff  "] = "Limits on velocity offset from the central wavelength [lnlambda]"
    hdr["vindex"] = "Entries w/ same NONZERO vindex constrained to have same velocity"
    hdr["windex"] = "Entries w/ same NONZERO windex constrained to have same width"
    hdr["findex"] = "Entries w/ same NONZERO findex have constrained flux ratios"
    hdr["fvalue"] = "Relative scale factor for entries w/ same findex"
    return hdr


# ---------- FITS <-> CSV ----------

def fits_to_csv(fits_path: str, csv_path: str) -> None:
    """Export a FITS parameter table to CSV for easy editing."""
    t = Table.read(fits_path)
    t = _ensure_table_schema(t)
    t.write(csv_path, format="csv", overwrite=True)


def csv_to_fits(csv_path: str, fits_path: str, header: Optional[fits.Header] = None) -> None:
    """Convert a CSV parameter table back to FITS with required schema."""
    t = Table.read(csv_path, format="csv")
    t = _ensure_table_schema(t)
    hdu = table_to_hdu(t)  # robust conversion from astropy Table to BinTableHDU
    hdu.name = "data"
    # Merge/attach descriptive header if provided
    hdr = header or _default_header()
    for key, val in hdr.items():
        try:
            comment = hdr.comments[key]
        except Exception:
            comment = None
        if comment is None:
            hdu.header[key] = val
        else:
            hdu.header[key] = (val, comment)
    hdu.writeto(fits_path, overwrite=True)


# ---------- FITS <-> YAML ----------

def fits_to_yaml(fits_path: str, yaml_path: str) -> None:
    """Export FITS table to a simple YAML list of records.

    YAML schema: a list of dicts with keys matching COL_NAMES.
    """
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required for YAML export. Try: pip install pyyaml") from exc

    t = Table.read(fits_path)
    t = _ensure_table_schema(t)
    rows: List[Dict[str, Any]] = []
    for r in t:
        rows.append({k: (str(r[k]) if k in ("compname", "linename") else float(r[k])) for k in COL_NAMES})
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(rows, f, sort_keys=False)


def yaml_to_fits(yaml_path: str, fits_path: str, header: Optional[fits.Header] = None) -> None:
    """Convert a YAML parameter list back to FITS.

    Accepts a YAML list of dicts. Keys not present are filled with defaults.
    """
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required for YAML import. Try: pip install pyyaml") from exc

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or []
    if not isinstance(data, list):
        raise ValueError("YAML must be a list of records")

    # Build table from rows, filling defaults for missing columns
    rows: Dict[str, List[Any]] = {k: [] for k in COL_NAMES}
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Each YAML item must be a mapping/dict")
        for name, dt in zip(COL_NAMES, COL_DTYPES):
            if name in item:
                val = item[name]
            else:
                val = "" if dt.startswith("U") else np.nan
            rows[name].append(val)
    t = Table(rows)
    t = _ensure_table_schema(t)

    hdu = table_to_hdu(t)
    hdu.name = "data"
    hdr = header or _default_header()
    for key, val in hdr.items():
        try:
            comment = hdr.comments[key]
        except Exception:
            comment = None
        if comment is None:
            hdu.header[key] = val
        else:
            hdu.header[key] = (val, comment)
    hdu.writeto(fits_path, overwrite=True)


# ---------- Convenience: create a template CSV/YAML ----------

def write_template_csv(csv_path: str) -> None:
    """Write an empty CSV with headers for manual editing."""
    t = Table({k: [] for k in COL_NAMES})
    t = _ensure_table_schema(t)
    t.write(csv_path, format="csv", overwrite=True)


def write_template_yaml(yaml_path: str) -> None:
    """Write a minimal YAML template with a couple of example rows."""
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required for YAML template. Try: pip install pyyaml") from exc

    example = [
        {
            "lambda": 6564.61,
            "compname": "Ha",
            "minwav": 6400.0,
            "maxwav": 6800.0,
            "linename": "Ha_br",
            "ngauss": 3,
            "inisig": 5e-3,
            "minsig": 0.003,
            "maxsig": 0.01,
            "voff": 0.005,
            "vindex": 0,
            "windex": 0,
            "findex": 0,
            "fvalue": 0.05,
        },
        {
            "lambda": 4862.68,
            "compname": "Hb",
            "minwav": 4640.0,
            "maxwav": 5100.0,
            "linename": "Hb_na",
            "ngauss": 1,
            "inisig": 1e-3,
            "minsig": 2.3e-4,
            "maxsig": 0.0017,
            "voff": 0.01,
            "vindex": 1,
            "windex": 1,
            "findex": 0,
            "fvalue": 0.002,
        },
    ]
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(example, f, sort_keys=False)
