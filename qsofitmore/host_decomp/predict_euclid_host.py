"""CLI: predict a DESI-derived host model on a Euclid spectrum grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .euclid import predict_host_for_euclid_spectrum, write_euclid_prediction
from .io import read_sparcli_spectrum
from .ppxf_host import HostSED


def _load_host_sed(npz_path: str) -> HostSED:
    data = np.load(npz_path)
    return HostSED(
        wave_rest=np.asarray(data["host_sed_wave"], dtype=float),
        host_flux=np.asarray(data["host_sed_flux"], dtype=float),
        samples={},
        flags={},
        warnings=[],
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Predict host contamination on a Euclid spectrum grid.")
    parser.add_argument("--decomp", required=True, help="host_decomp_result.npz")
    parser.add_argument("--euclid-spectrum", required=True)
    parser.add_argument("--redshift", required=True, type=float)
    parser.add_argument("--scale-mode", default="free_scale")
    parser.add_argument("--aperture-scale", type=float, default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    host_sed = _load_host_sed(args.decomp)
    euclid = read_sparcli_spectrum(args.euclid_spectrum, redshift=args.redshift)
    prediction = predict_host_for_euclid_spectrum(
        host_sed,
        euclid.wave_obs,
        args.redshift,
        euclid_flux=euclid.flux,
        scale_mode=args.scale_mode,
        aperture_scale=args.aperture_scale,
    )
    path = write_euclid_prediction(prediction, args.output_dir)
    print(json.dumps({"output": path, "scale_factor": prediction.scale_factor, "warnings": prediction.warnings}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
