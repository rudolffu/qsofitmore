"""CLI: run optional pPXF + qsofitmore host decomposition."""

from __future__ import annotations

import argparse
import json

from .ppxf_host import run_ppxf_qsofitmore_decomposition


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run pPXF host decomposition on a DESI/SPARCL-like spectrum.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--row-index", type=int, default=None)
    parser.add_argument("--redshift", type=float, default=None)
    parser.add_argument("--object-id", default=None)
    parser.add_argument("--template-root", default="~/tools/ppxf_data")
    parser.add_argument("--template-file", default="spectra_emiles_9.0.npz")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fit-range", nargs=2, type=float, default=(3600.0, 7000.0))
    parser.add_argument("--run-qsofitmore", action="store_true")
    parser.add_argument("--line-list-path", default=None)
    parser.add_argument("--n-iterations", type=int, default=1)
    args = parser.parse_args(argv)
    result = run_ppxf_qsofitmore_decomposition(
        input_path=args.input,
        row_index=args.row_index,
        redshift=args.redshift,
        object_id=args.object_id,
        template_root=args.template_root,
        template_file=args.template_file,
        output_dir=args.output_dir,
        fit_range=tuple(args.fit_range),
        run_qsofitmore=args.run_qsofitmore,
        n_iterations=args.n_iterations,
        line_list_path=args.line_list_path,
    )
    print(json.dumps(result.summary, indent=2, sort_keys=True))
    print(json.dumps(result.output_files, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
