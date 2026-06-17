# neofit Development Plan

## Purpose

`qsofitmore.neofit` is a new array-based fitting core for future
high-performance quasar spectral fitting. It will live alongside the legacy
`QSOFitNew` API and will not replace `fitmodule.py` until it has enough science
coverage, validation, and user-facing documentation.

The main goals are:

- a small, explicit object model: `Spectrum`, config/recipe objects, and result
  objects;
- packed numerical parameter vectors instead of mutable `lmfit.Parameters`;
- optimizer-facing functions that are pure NumPy array operations;
- `scipy.optimize.least_squares` as the first production optimizer;
- analytic Jacobians where feasible;
- a structure that can grow into sparse Jacobians for line-complex and batch
  fitting;
- no plotting, file writing, pandas, Astropy tables, or mutable legacy instance
  attributes inside optimization loops.

## neofit vs Legacy QSOFitNew

The current `QSOFitNew.Fit(...)` workflow is a large orchestration method. It
handles preprocessing, smoothing, dereddening, continuum models, Fe II, Balmer
continuum, optional host decomposition, line fitting, Monte Carlo runs, output
writing, and plotting from one mutable object.

`neofit` should instead separate concerns:

- immutable-ish inputs (`Spectrum`);
- explicit fit recipes (`LineComplexConfig`, later continuum/full-model config);
- packed parameter vectors and bounds;
- pure residual/Jacobian functions;
- result objects that can be converted to dictionaries or tables by caller code.

Legacy qsofitmore remains the reference implementation and the stable public API
for full science workflows while `neofit` matures.

## Intended Public API

MVP API:

```python
from qsofitmore import neofit

spectrum = neofit.Spectrum.from_arrays(
    wave=wave,
    flux=flux,
    err=err,
    z=z,
    wave_frame="observed",
)

config = neofit.LineComplexConfig(
    center=4861.33,
    window=(4700.0, 5100.0),
    components=[
        neofit.GaussianComponent(
            name="Hb_broad",
            center=4861.33,
            amp=10.0,
            sigma=30.0,
            bounds={
                "amp": (0.0, None),
                "center": (4800.0, 4920.0),
                "sigma": (5.0, 200.0),
            },
        ),
    ],
    local_continuum="linear",
)

result = neofit.fit_line_complex(spectrum, config)
```

Initial exported names:

- `Spectrum`
- `GaussianComponent`
- `LineComplexConfig`
- `IronTemplateConfig`
- `FitResult`
- `fit_line_complex`
- `SpectrumMetadata`
- `LocalFitConfig`
- `LocalFitResult`
- `fit_local`
- `NeoFitWarning`
- `list_iron_templates`
- `load_iron_template`

## Internal Module Layout

```text
qsofitmore/neofit/
  __init__.py
  api.py
  spectrum.py
  config.py
  parameters.py
  result.py
  optimize.py
  residuals.py
  jacobian.py
  templates/
    __init__.py
    iron.py
    normalize.py
    registry.py
    README.md
    data/
  models/
    __init__.py
    gaussian.py
    continuum.py
```

Responsibilities:

- `spectrum.py`: array validation, observed/rest wavelength handling, valid-pixel
  masks.
- `config.py`: dataclass fit recipes and model-component definitions.
- `parameters.py`: named-parameter to packed-vector conversion, bounds, unpacking.
- `models/`: pure array model pieces.
- `residuals.py`: model assembly and weighted residual functions.
- `jacobian.py`: analytic model and residual derivatives, dense now and sparse
  structure-ready.
- `templates/`: bundled iron-template registry, parsing, normalization, and
  fitting-grid preparation. Template file I/O and convolution happen before
  optimization, not inside residual/Jacobian calls.
- `optimize.py`: thin `scipy.optimize.least_squares` wrapper.
- `result.py`: `FitResult` dataclass plus `to_dict()` and `to_table()`.
- `api.py`: public orchestration helpers.

## Initial MVP Scope

The MVP is a local line-complex fitter:

- rest-frame fitting window;
- one or more Gaussian components;
- local continuum modes: `None`, `"constant"`, `"linear"`;
- optional iron-template component with fitted amplitude and FWHM;
- optional fit sub-windows and masked wavelength intervals so continuum/iron
  constraints can follow the legacy line-free continuum-window strategy;
- packed parameter vectors and bounds;
- weighted residual convention `r = (flux - model) / err`;
- analytic dense Jacobian;
- sparse Jacobian option returning CSR from the same derivative structure;
- fast synthetic tests.

This is intentionally much smaller than the full legacy science model.

## Spectrum Metadata and Unit Policy

`neofit` stores wavelength and flux-density labels in a lightweight
`SpectrumMetadata` dataclass attached to `Spectrum.metadata`. The numerical core
continues to work only with plain NumPy arrays. No Astropy `Quantity`, pandas
objects, or unit-aware wrappers are passed into residual or Jacobian functions.

Current metadata fields:

```python
SpectrumMetadata(
    wave_unit="Angstrom",
    flux_density_unit="input",
    flux_density_scale_to_cgs=None,
    survey=None,
    source=None,
    notes=[],
)
```

If `flux_density_scale_to_cgs` is known, then:

```text
physical_flux_density_cgs = flux_array * flux_density_scale_to_cgs
```

If it is unknown, fitting still works in input units. Input-unit line fluxes are
reported, while cgs line fluxes are left as NaN and a
`flux_scale_unknown_cgs_not_reported` warning is attached.

### Survey and Unit Presets

Common DESI/SDSS-style spectra can be declared directly:

```python
spectrum = neofit.Spectrum.from_arrays(wave, flux, err=err, z=z, survey="desi")
spectrum = neofit.Spectrum.from_arrays(wave, flux, err=err, z=z, survey="sdss")
```

Accepted aliases include `DESI`, `desi-dr1`, `desi_edr`, `sdss`, and `SDSS`.
They normalize to canonical `desi` or `sdss` metadata:

```text
wave_unit = "Angstrom"
flux_density_unit = "1e-17 erg s^-1 cm^-2 Angstrom^-1"
flux_density_scale_to_cgs = 1e-17
```

Unit-only convenience:

```python
spectrum = neofit.Spectrum.from_arrays(wave, flux, err=err, z=z, unit_preset="1e-17cgs")
spectrum = neofit.Spectrum.from_arrays(wave, flux, err=err, z=z, unit_preset="input")
```

Unknown/input units remain the default:

```python
spectrum = neofit.Spectrum.from_arrays(wave, flux, err=err, z=z)
```

Manual keyword overrides take priority over presets:

```python
spectrum = neofit.Spectrum.from_arrays(
    wave,
    flux,
    err=err,
    z=z,
    survey="desi",
    flux_density_unit="erg s^-1 cm^-2 Angstrom^-1",
    flux_density_scale_to_cgs=1.0,
)
```

Priority order:

```text
explicit user keyword > unit_preset > survey preset > base metadata/defaults
```

`neofit` does not infer physical units from numerical flux amplitudes. Unknown
survey or unit presets raise a clear `ValueError`.

## Local Fitting Mode

Local fitting is an explicit science mode for cases where only selected
emission-line windows are useful or trustworthy. It fits each requested
`LineComplexConfig` independently using the same array-based
`fit_line_complex(...)` machinery.

Example:

```python
config = neofit.LocalFitConfig(
    windows=[
        neofit.recipes.local_hbeta(),
        neofit.recipes.local_halpha(),
    ],
)
result = neofit.fit_local(spectrum, config)
```

The first implemented mode is:

```text
mode = "independent"
```

This means no parameters are shared between windows. Future global or joint
modes may share continuum parameters, tie line velocities/widths, or fit a full
multi-window model with block-sparse Jacobians.

`LocalFitResult.success` means at least one requested window succeeded.
Individual window status is available through:

```python
result.window_results["Hb_OIII"].success
```

`LocalFitResult.to_table()` combines successful window tables and adds a
`window` column.

## Optional Host Subtraction Before neofit

`neofit.fit_with_optional_host_decomp(...)` can read a DESI/SPARCL-like
spectrum, optionally run the existing pPXF host-decomposition workflow, and then
run local neofit on the host-subtracted spectrum. The public switch is
`run_host_decomp=True`.

For host subtraction, the code uses the pPXF-fitted template-weighted host SED
evaluated on the same valid rest-frame wavelength grid as the quasar spectrum:

```text
host_subtracted_flux = total_flux - host_sed_on_quasar_grid
```

No extrapolation is performed. Pixels outside the template SED coverage are
excluded from the host-subtracted neofit spectrum. The original total spectrum,
host-subtracted spectrum, pPXF fit, host SED, and local neofit result are all
returned in a `NeoFitHostWorkflowResult`.

`fit_kind="global"` is reserved and currently raises `NotImplementedError`; the
global neofit model is not implemented yet.

Recipes use enlarged local windows and legacy-style line-free continuum bands
when iron is enabled or when enough wavelength coverage is available. The
configured `fit_windows` define the wavelength intervals sent to the optimizer,
while `mask_windows` remove strong unmodeled emission features from those
intervals. This mirrors the legacy continuum-fit idea: constrain continuum and
Fe II from bands that avoid the brightest emission lines, instead of forcing the
iron template to absorb line residuals.

Masked pixels are still valid science pixels. `FitResult` therefore keeps both
the fitted arrays (`wave_rest_fit`, `flux_fit`, `model`, etc.) and full valid
local-window arrays (`wave_rest_window`, `flux_window`, `model_window`,
`component_models_window`, and `fit_used_window`). Plots use the full valid
local-window arrays and zoom to `plot_window`, so masked emission-line regions
remain visible for assessing continuum subtraction and later line fitting.

Current recipe defaults:

- MgII: window `1970--3400 Å`, fitting `1970--2400`, `2480--2675`,
  `2700--2900`, and `2925--3400 Å`; plotting `2700--2900 Å`.
- Hβ/[OIII]: window `4435--5535 Å`, fitting `4435--4640`,
  `4700--5100`, and `5100--5535 Å`, masking `4930--5035 Å` for [OIII],
  and plotting `4700--5100 Å`.
- Hα/[NII]/[SII]: window `6005--7000 Å`, fitting `6005--6035`,
  `6110--6250`, `6300--6800`, and `6800--7000 Å`, masking common [OI],
  [NII]/narrow-Hα-core, and [SII] intervals, and plotting `6300--6800 Å`.

## Local Iron Templates

`neofit` can add one optional iron-template component to a local line complex.
The current implementation fits the iron amplitude and FWHM. No iron velocity
shift is exposed in the local model. Template file I/O happens before
optimization; during fitting, the loaded normalized template is broadened on the
local grid as the trial FWHM changes.

Bundled template aliases:

- `bg92` / `bg92_optical`: legacy qsofitmore optical Fe II template.
- `park22` / `park22_optical`: Park et al. 2022 optical H-beta-region template.
- `vc04`, `veron04` / `veron04_optical`: Veron-Cetty, Joly & Veron 2004 optical template.
- `vw01` / `vw01_uv`: legacy qsofitmore UV/MgII-region template.

Examples:

```python
config = neofit.recipes.local_hbeta(iron_template="park22", iron_fwhm_kms=4000.0)
config = neofit.recipes.local_mgii(iron_template="vw01", iron_fwhm_kms=3000.0)
```

Manual configuration:

```python
config = neofit.LineComplexConfig(
    center=4861.33,
    window=(4435.0, 5535.0),
    fit_windows=[(4435.0, 4640.0), (4700.0, 5100.0), (5100.0, 5535.0)],
    mask_windows=[(4930.0, 5035.0)],
    components=[...],
    iron=neofit.IronTemplateConfig(template="vc04", fwhm_kms=2500.0),
)
```

Custom two-column or FITS templates remain possible:

```python
iron = neofit.IronTemplateConfig(
    template="external",
    template_path="/path/to/template.txt",
    fwhm_kms=3000.0,
)
```

All templates are converted to increasing rest wavelength in Angstrom and
positive-area-normalized flux. Fitted amplitudes therefore have units of
input-flux-density times Angstrom for the current area-normalized basis.
`FitResult.to_table()` appends an `iron` row with `iron_template`, `iron_amp`,
`iron_fwhm_kms`, `iron_flux_input`, `iron_flux_cgs`, and template
coverage/reference fields when an iron component is active.

## Warning and Status-Code Policy

Science-facing reliability issues are represented as `NeoFitWarning` objects:

```python
NeoFitWarning(code, message, severity="warning", context={})
```

Stable codes currently include:

- `flux_scale_unknown_cgs_not_reported`
- `window_not_covered`
- `window_too_few_pixels`
- `line_center_outside_coverage`
- `line_center_near_edge`
- `all_pixels_invalid`
- `fit_failed`
- `iron_template_no_overlap`
- `iron_template_partial_coverage`
- `unknown_iron_template`
- `missing_iron_template_path`
- `iron_template_parse_failed`
- `iron_template_not_monotonic`
- `iron_template_zero_norm`

Direct constructor misuse, such as unknown survey or unit presets, raises
`ValueError`. Data/coverage problems in local fitting are returned as failed
window results with warnings, so a multi-window fit can still complete for other
valid windows.

## Analytic and Sparse Jacobians

For each Gaussian

```text
G(lambda) = A exp[-0.5 ((lambda - mu) / sigma)^2]
```

the model derivatives are:

```text
dG/dA     = exp[-0.5 u^2]
dG/dmu    = G * (lambda - mu) / sigma^2
dG/dsigma = G * (lambda - mu)^2 / sigma^3
```

with `u = (lambda - mu) / sigma`.

Because the residual is `r = data - model`, the optimizer Jacobian is:

```text
dr/dtheta = -dmodel/dtheta / err
```

The MVP implements `jacobian="analytic_dense"` directly. It also accepts
`jacobian="analytic_sparse"` and returns a `scipy.sparse.csr_matrix`; for the
first implementation this can wrap the dense local-complex derivative because a
single local complex is small. The API and module boundaries should make it
straightforward to assemble truly sparse block Jacobians for many independent
windows later.

## Explicit Non-Goals for the First Implementation

Do not implement yet:

- full/global Fe II fitting with velocity-shift parameters;
- Balmer continuum;
- full qsofitmore line-table compatibility;
- Monte Carlo uncertainty loops;
- pPXF host decomposition inside `neofit`;
- JAX, Numba, or Cython;
- DESI batch processing;
- plotting-heavy outputs;
- legacy FITS output compatibility;
- replacement of `QSOFitNew`.

Additional current non-goals:

- automatic flux-unit inference from numerical scale;
- equivalent-width reporting;
- shared-parameter local-window fitting;
- true block-sparse multi-window Jacobian assembly.

## Staged Roadmap

### Stage 1: MVP Line-Complex Fitter

- Synthetic Gaussian line recovery.
- Constant/linear local continuum.
- Dense analytic Jacobian.
- CSR sparse-compatible Jacobian output.
- Tests against finite-difference derivatives.

### Stage 2: Continuum and Fe II Extensions

- Add continuum recipe objects.
- Add global continuum windows.
- Port selected Fe II models into pure array functions.
- Extend the local iron-template implementation only after validating
  template-specific science behavior.
- Keep legacy templates and science behavior as references.

### Stage 3: Full Quasar Model

- Combine continuum, Fe II, Balmer continuum, and line complexes.
- Support tied parameters and flux ratios.
- Decide how to represent legacy line-table semantics cleanly.
- Add result adapters that can produce legacy-like tables when needed.

### Stage 4: Batch Fitting for DESI-Scale Use

- Batch-safe input schemas.
- Reusable compiled/configured model contexts.
- Sparse block Jacobians across many independent line windows.
- Better failure reporting and per-object quality flags.

### Stage 5: Optional Acceleration

- Evaluate JAX or Numba only after the NumPy/SciPy implementation is validated.
- Keep pure NumPy as the reference backend.
- Avoid requiring accelerator dependencies for normal imports.

## Validation Strategy

Short term:

- finite-difference checks for Gaussian and continuum Jacobians;
- synthetic spectra with known Gaussian parameters and noise;
- invalid-pixel masking tests;
- regression tests that `import qsofitmore` and legacy `QSOFitNew` still work.

Medium term:

- compare simple line-complex fits against legacy qsofitmore on selected
  examples;
- test linear/km-s line-parameter behavior once line-table compatibility is
  introduced;
- add continuum-only and continuum-plus-line synthetic cases.

Long term:

- curated validation spectra from SDSS/LAMOST/DESI;
- batch benchmarks for dense vs sparse Jacobian modes;
- science validation of equivalent widths, FWHM, line fluxes, and host fractions
  against trusted legacy outputs.
