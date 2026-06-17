# pPXF + qsofitmore Host Decomposition

This optional workflow combines a pPXF stellar-continuum fit with the existing
qsofitmore AGN fitting machinery. It is intended for DESI/SPARCL-like optical
spectra where the DESI data are used to estimate a host-galaxy component and to
predict the host SED over the wavelength range covered by local stellar
population templates.

The workflow is empirical and DESI-calibrated. It is not a full 2D Euclid
slitless forward model. Euclid host subtraction requires aperture scaling, and
NIR host predictions are model-dependent.

## Local Setup and License Boundary

Install pPXF and templates outside this repository. Do not commit pPXF source,
pPXF examples, E-MILES files, downloaded templates, DESI spectra, Euclid spectra,
or large host-decomposition outputs to qsofitmore.

Expected local paths:

- `~/tools/ppxf_data/spectra_emiles_9.0.npz`
- `~/tools/ppxf_examples`

The code imports pPXF only when a pPXF fit is explicitly run. Importing
`qsofitmore` or `qsofitmore.host_decomp` does not require pPXF.

## Why pPXF and E-MILES

The legacy Yip et al. PCA decomposition used by qsofitmore is useful in the
optical fitting context, but it does not provide a physically traceable host SED
for NIR host-contamination tests. E-MILES templates provide a template-weighted
stellar model that can be evaluated over the template wavelength coverage. The
workflow never extrapolates beyond that coverage.

## Basic Commands

Inspect a local SPARCL/DESI-like file:

```bash
python -m qsofitmore.host_decomp.inspect_spectrum \
  --input /path/to/sparcli_output.fits
```

Run the validation target from the local DESI parquet bundle:

```bash
python -m qsofitmore.host_decomp.ppxf_host_fit \
  --input /Users/yuming/astro/ml_projects/dr1agn/mlspecz_data/raw/large_edfn_north_redshift/qc_tiers/desispecs_tierA.parquet \
  --row-index 19 \
  --redshift 0.3903328627652504 \
  --object-id 39633451355212677 \
  --template-root ~/tools/ppxf_data \
  --template-file spectra_emiles_9.0.npz \
  --output-dir outputs/ppxf_qsofitmore/39633451355212677 \
  --fit-range 3600 7000 \
  --run-qsofitmore \
  --n-iterations 1
```

Predict a host model on a Euclid spectrum grid:

```bash
python -m qsofitmore.host_decomp.predict_euclid_host \
  --decomp outputs/ppxf_qsofitmore/39633451355212677/host_decomp_result.npz \
  --euclid-spectrum path/to/euclid_spectrum.fits \
  --redshift 0.3903328627652504 \
  --scale-mode free_scale \
  --output-dir outputs/ppxf_qsofitmore/39633451355212677/euclid_host_prediction
```

## Outputs

Per object, outputs are written under `outputs/ppxf_qsofitmore/<object_id>/`.
The standard products are:

- `host_decomp_result.npz`
- `host_decomp_summary.json`
- `host_decomp_summary.csv`
- `desi_total_spectrum.csv`
- `desi_ppxf_host_model.csv`
- `desi_ppxf_agn_continuum_model.csv`
- `desi_host_subtracted.csv`
- `qsofitmore_model.csv` when available
- `host_sed_prediction.csv`
- diagnostic plots

Summary files include template coverage, pPXF and qsofitmore statuses, sampled
host fluxes at 4000, 5100, 8000, 10000, 16000, and 22000 Angstrom, NIR coverage
flags, and warnings.

## Flux and Host-Fraction Columns

The workflow preserves the input spectrum flux-density scale. For the local
DESI/SPARCL parquet files, flux-like values are in
`1e-17 erg cm^-2 s^-1 Angstrom^-1` by default. The summary field
`flux_density_unit` records this default unit.

Summary flux columns are rest-frame flux-density samples:

- `fHost_<wave>` is the template-weighted host SED prediction sampled at rest
  wavelength `<wave>`. It is evaluated on the stellar-template wavelength grid,
  so it can be finite outside the observed DESI rest-frame range when the
  template covers that wavelength. It is not a host fraction.
- `fAGN_5100` is kept for compatibility and is the pPXF AGN power-law component
  sampled at rest-frame 5100 Angstrom on the fitted DESI grid. It is `NaN` when
  5100 Angstrom is not inside the finite fitted pPXF range.
- `fHostFit_<wave>`, `fAGNFit_<wave>`, and `fTotalFit_<wave>` are sampled from
  the pPXF-fitted host, AGN-basis, and total-continuum models on the input
  spectrum rest-frame grid. They are `NaN` outside the finite fitted range.
- `fracHost_<wave>` is dimensionless and is computed as
  `fHostFit_<wave> / fTotalFit_<wave>`. It is only finite when `<wave>` lies
  within the input spectrum rest-frame coverage and the finite pPXF fitted
  model range. Host fractions are not reported outside the input rest-frame
  spectrum, even when `fHost_<wave>` has a template-SED prediction.

## Limitations

- Emission lines are masked in the first implementation rather than fit with
  pPXF gas components.
- The AGN contribution is represented by a small power-law basis.
- NIR prediction is template-limited; values outside template coverage are NaN.
- Euclid subtraction must include aperture scaling. The DESI fiber and Euclid
  extraction aperture are not assumed to match.
