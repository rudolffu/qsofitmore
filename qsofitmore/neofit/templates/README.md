# neofit Iron Templates

This directory contains normalized iron-template support for
`qsofitmore.neofit`. The optimizer uses plain NumPy arrays only; template file
I/O, provenance, and metadata are handled outside the residual and Jacobian hot
loop.

## Internal Format

Bundled normalized templates live in `data/*.txt` with two columns:

```text
wave_rest_angstrom  area_normalized_flux
```

The standard normalization is:

```text
integral(max(flux, 0) d wavelength) over native coverage = 1
```

The flux scale is therefore arbitrary but consistent across templates. In local
fits the fitted iron amplitude multiplies the template prepared at the fitted
FWHM. The reported `iron_flux_input` is the integral of
`iron_amp * prepared_template(fitted_fwhm)` over the local fitted wavelength
grid.

Built-in local recipes use enlarged windows plus line-free sub-windows inspired
by the legacy qsofitmore continuum fit. Strong unmodeled emission features are
masked before optimization so the iron template is constrained mainly by
continuum/Fe-dominated regions instead of by narrow-line residuals. These
masked pixels are still evaluated and plotted in the narrower line-complex
`plot_window`, so downstream continuum-subtracted line work can use the valid
emission-line pixels.

## Supported Names and Aliases

| User aliases | Canonical name | Region | Bundled? |
| --- | --- | --- | --- |
| `bg92`, `bg92_optical` | `bg92_optical` | optical/H-beta | yes |
| `park22`, `park22_optical` | `park22_optical` | optical/H-beta | yes |
| `veron04`, `vc04`, `veron04_optical` | `veron04_optical` | optical/H-beta to broad optical | yes |
| `vw01`, `vw01_uv` | `vw01_uv` | UV/MgII | yes |
| `external` | `external` | user supplied | no |

Users can choose bundled templates by name without passing a path:

```python
neofit.IronTemplateConfig(template="park22", fwhm_kms=4000.0)
```

External two-column ASCII templates are still supported:

```python
neofit.IronTemplateConfig(template="external", template_path="/path/to/template.txt")
```

## Template Notes

### `bg92`

Legacy qsofitmore/PyQSOFit optical Fe II template associated with
Boroson & Green (1992) and the existing package data file
`qsofitmore/data/iron_templates/fe_optical.txt`. Intended for H-beta-region
tests and backward compatibility.

### `park22`

Park et al. (2022), ApJS, 258, 38. The local source file supplied for this repo
was `Park22/tab1.txt`, the empirical Mrk 493 optical Fe II template for the
H-beta region, covering 4000--5600 Angstrom. This implementation stores an
area-normalized two-column copy in `data/park22_optical.txt`.

Recent literature suggests Park22 can perform better for broad optical Fe II,
while BG92 may perform better for narrower Fe II. Users should test both for
H-beta-region systematics.

### `veron04`

Veron-Cetty, Joly & Veron (2004), A&A, 417, 515. The local source file supplied
for this repo was `VC04/iwz1.fit`, a synthetic Fe II spectrum of I Zw 1. The
converted bundled copy covers 3535--7534 Angstrom in
`data/veron04_optical.txt`. It is useful for broad optical coverage tests.

### `vw01`

Legacy qsofitmore UV Fe template associated with Vestergaard & Wilkes (2001)
and the existing package data file `qsofitmore/data/iron_templates/fe_uv.txt`.
Intended for MgII-region local fits.

## Current Limitations

- Iron amplitude and FWHM are fitted in local neofit models.
- Iron velocity shift is not included in the local neofit iron component.
- Broadening is performed in log wavelength with a Gaussian velocity kernel.
- No physical Fe II template uncertainty model is included.
- No Fe II multiplet group tying or full legacy qsofitmore Fe II behavior is
  implemented yet.

## High-Order Balmer Templates

`data/balmer/` contains 12 CSV products copied from
`balmer_template_builder` commit `d3e1695`. They preserve vacuum wavelength,
H-beta-relative intensity, Case B physical conditions, and row-level
provenance.

The dimensions are:

- `log10 Ne = 9, 10`
- `n_min = 6, 7`
- direct Storey & Hummer 1995 lines through `n=50`
- K13-full and asymptotic systematics extensions through `n=400`

Pure SH95 is the fitting default. Extended products carry
`balmer_high_n_extension_model_dependent`; the diagnostic energy-only
extension is deliberately excluded.
