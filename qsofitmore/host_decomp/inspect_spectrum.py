"""CLI: inspect a local SPARCL/DESI-like spectrum."""

from __future__ import annotations

import argparse
import json

from .io import inspect_spectrum


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Inspect a SPARCL/DESI-like spectrum file.")
    parser.add_argument("--input", required=True, help="Input FITS/parquet/CSV/ECSV/NPZ spectrum")
    parser.add_argument("--row-index", type=int, default=None, help="Row index for vector-column tables")
    args = parser.parse_args(argv)
    print(json.dumps(inspect_spectrum(args.input, row_index=args.row_index), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
