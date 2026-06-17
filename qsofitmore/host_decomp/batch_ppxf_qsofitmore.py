"""CLI: simple serial batch runner for pPXF + qsofitmore host decomposition."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .ppxf_host import run_ppxf_qsofitmore_decomposition


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Batch pPXF host decomposition.")
    parser.add_argument("--input-table", required=True)
    parser.add_argument("--template-root", default="~/tools/ppxf_data")
    parser.add_argument("--template-file", default="spectra_emiles_9.0.npz")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-workers", type=int, default=1, help="Reserved for future parallelism; currently serial.")
    args = parser.parse_args(argv)
    table = pd.read_csv(args.input_table)
    for _, row in table.iterrows():
        input_path = row.get("input") or row.get("input_file") or row.get("path")
        object_id = str(row.get("object_id", row.get("targetid", Path(str(input_path)).stem)))
        row_index = None if pd.isna(row.get("row_index", None)) else int(row.get("row_index"))
        redshift = None if pd.isna(row.get("redshift", row.get("z", None))) else float(row.get("redshift", row.get("z")))
        out = Path(args.output_dir) / object_id
        result = run_ppxf_qsofitmore_decomposition(
            input_path=input_path,
            row_index=row_index,
            redshift=redshift,
            object_id=object_id,
            template_root=args.template_root,
            template_file=args.template_file,
            output_dir=str(out),
            run_qsofitmore=bool(row.get("run_qsofitmore", False)),
        )
        print(object_id, result.summary.get("ppxf_status"), result.summary.get("qsofitmore_status"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
