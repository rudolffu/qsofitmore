# neofit Status and Development Plan

## Purpose

`qsofitmore.neofit` is the array-based NumPy/SciPy fitting core being developed
alongside the legacy `QSOFitNew` API. It now supports a complete global
continuum and broad-line workflow, but the legacy implementation remains the
stable reference for science features that have not yet been ported.

The design principles remain:

- explicit `Spectrum`, configuration, and result objects;
- packed numerical parameter vectors rather than mutable fit state;
- pure array model, residual, and Jacobian functions inside optimization;
- bounded variable projection where amplitudes or fluxes are linear;
- analytic or semi-analytic derivatives with finite-difference validation;
- plotting, table writing, and host orchestration outside optimization loops;
- backward-compatible legacy imports and H-beta-named wrappers.

## Current Status

### Global science workflow

The production global entry point is:

```python
from qsofitmore import neofit

result = neofit.fit_global_lines_workflow(
    spectrum_path,
    row_index=19,
    run_host_decomp=True,
)

files = neofit.write_global_line_products(
    result,
    output_dir,
    qa_plot_config=neofit.GlobalQAPlotConfig(
        show_host_context_in_overview=True,
    ),
)
```

The workflow currently provides:

- optional pPXF host decomposition and subtraction;
- global AGN continuum fitting with a power law, independently broadened UV and
  optical iron, Balmer continuum, and high-order Balmer series;
- constrained H-beta/[O III] fitting with three ordered broad components,
  shared narrow kinematics, fixed [O III] ratio, optional He II, and
  evidence-based [O III] wing selection;
- a freely fitted Balmer-series FWHM by default, with reliability-gated,
  transactional H-beta synchronization as an optional refinement;
- automatic Mg II fitting with two ordered broad components when covered;
- automatic H-alpha/[N II]/[S II] fitting with three ordered broad components,
  shared narrow kinematics, fixed [N II] ratio, and independent [S II] fluxes;
- immutable vacuum-wavelength and emission-complex registries;
- adaptive recipe selection from rest-frame coverage, including executable
  optical-blue and component-adaptive Paschen/NIR recipes;
- covariance errors and optional Monte Carlo propagation, including optional
  pPXF host refitting;
- continuum samples, host fractions, broad-profile moments, numerical FWHM,
  equivalent widths, line fluxes, and per-complex measurements;
- generic global APIs plus compatibility wrappers retaining historical
  H-beta-oriented names.

H-beta is no longer mandatory. When its recipe is not covered, `hbeta` and
`hbeta_initial` are `None` and no H-beta result is inserted into
`line_complexes`. Generic workflow results expose `continuum_success`,
`complex_statuses`, and per-complex success values; they do not synthesize an
aggregate workflow verdict. H-beta compatibility wrappers expose the
deprecated `legacy_hbeta_success` property.

`complexes=None` selects covered, executable, auto-enabled recipes.
`complexes=[]` performs a continuum-only fit. Explicit recipe IDs or immutable
recipe objects can be supplied to select a custom set.

### Registry and generic complex fitting

`neofit.lines` provides canonical definitions, normalized aliases, vacuum
wavelengths, transition types, blend membership, and provenance.
`neofit.recipes` provides immutable `ComplexRecipe` and `ComponentRecipe`
objects with discovery, description, and copy-based component overrides.

The generic compiler supports Gaussian and Lorentzian integrated-flux
components, shared velocity/FWHM groups, fixed ratios, independent positive
fluxes, component-adaptive coverage, separate fit windows, and fixed-global or
local continuum modes. Current Mg II, H-beta, and H-alpha implementations
remain authoritative through dedicated recipe adapters.

### Optimizers and derivatives

Global continuum, H-beta, Mg II, and H-alpha fits use bounded variable
projection by default:

- linear amplitudes and integrated fluxes are solved with bounded dense linear
  least squares;
- slopes, widths, and velocities are optimized nonlinearly;
- exact Gaussian, velocity, width, power-law, and Balmer-series derivatives are
  used;
- iron-width derivatives use the differentiated broadening kernel;
- active-bound linear coefficients are handled by a fixed-active-set reduced
  Jacobian;
- final full Jacobians preserve the original parameter ordering for covariance
  calculations.

Solver controls are available on the global and complex configurations:

```python
global_config = neofit.GlobalContinuumConfig(
    optimizer_method="auto",       # auto, variable_projection, legacy_joint
    jacobian_method="semi_analytic",  # semi_analytic, 2-point
)
```

`auto` falls back to the legacy finite-difference joint solver when the
optimized path fails. Results record requested and used solvers, fallback
status, linear/nonlinear parameter counts, nonlinear evaluations, Jacobian
evaluations, and linear-solve counts.

### Balmer and iron templates

The global workflow uses:

- VW01 UV iron by default;
- Park22 optical iron by default;
- bundled Storey & Hummer high-order Balmer templates;
- pure SH95 lines through `n=50` as the production Balmer-series choice;
- K13-full and asymptotic extensions through `n=400` as explicit systematics
  choices with model-dependence warnings.

Local fitting supports bundled and external iron templates with fitted
amplitude and FWHM. Iron velocity shifts are not exposed in the local model.
Templates are normalized and prepared before optimization.

### Local fitting

Independent local fitting remains available:

```python
config = neofit.LocalFitConfig(
    windows=[
        neofit.recipes.local_mgii(),
        neofit.recipes.local_hbeta(profile="lorentzian"),
        neofit.recipes.local_halpha(),
    ],
)
result = neofit.fit_local(spectrum, config)
```

Local models support:

- Gaussian and Lorentzian profiles;
- constant, linear, or absent local continua;
- optional iron templates;
- explicit fit and mask windows;
- dense analytic and sparse-compatible Jacobians;
- independent success and warning states for each requested window.

Shared-parameter multi-window local fitting is not yet implemented.

### Metadata and units

`SpectrumMetadata` stores wavelength, flux-density, survey, source, and physical
scale information outside numerical fitting. DESI and SDSS presets use
Angstroms and `10^-17 erg s^-1 cm^-2 Angstrom^-1`.

Explicit metadata overrides presets. Numerical flux amplitudes are never used
to infer physical units. Input-unit measurements remain available when the cgs
scale is unknown, while cgs measurements are omitted with a stable warning.

Global DESI workflows preserve TARGETID, redshift, RA, and Dec for product
titles and summaries.

## Output Products and QA Figures

`write_global_line_products(...)` writes:

- generic and H-beta-compatible JSON summaries;
- continuum and per-complex measurement CSV files;
- generic and compatible full-grid CSV files;
- `main_qa_<normalized_object_name>.png` by default;
- optional PDF or combined PNG/PDF QA output;
- opt-in host-context, Balmer-edge, and specialized H-beta diagnostics.

### Global QA figure

The main object-named QA figure currently uses:

- a fixed `10.5 x 6.2` inch canvas;
- a full-spectrum overview plus up to four equally divided zoom panels;
- H-beta, Mg II, and H-alpha selection priority, displayed in wavelength order;
- original spectrum and host galaxy in the overview when
  `show_host_context_in_overview=True`;
- host-subtracted spectra in every zoom panel;
- one shared wavelength label and one shared flux-density label;
- zero lower limits and model-driven rounded upper limits;
- inward major and minor ticks;
- one broad-component legend in the first zoom only;
- direct emission-line labels, with vacuum wavelengths shown only for forbidden
  lines;
- combined broad profiles in the overview and individual broad components in
  zoom panels;
- consistent solid-line component styling and color separation:
  narrow lines remain green, iron remains purple, Balmer continuum/series are
  ochre, and outflow wings are dark red;
- continuum and per-complex reduced chi-square annotations;
- pPXF decomposition state and host fractions at 3000 and 5100 Angstrom when
  covered;
- automatic DESI TARGETID titles with redshift, RA, and Dec.

Configuration example:

```python
qa_config = neofit.GlobalQAPlotConfig(
    figure_width=10.5,
    figure_height=6.2,
    max_zoom_panels=4,
    show_smoothed_data=False,
    smoothing_window_pixels=7,
    show_host_context_in_overview=True,
    object_name=None,
    object_label=None,
    show_coordinates=True,
    output_format="png",  # png, pdf, or both
    write_other_diagnostics=False,
)
```

`object_name` and `object_label` can replace the catalog identifier and prefix.
The filename uses a normalized lowercase object name with punctuation and
spaces replaced by underscores.
If host context is requested but unavailable, the overview falls back to the
host-subtracted presentation.

### Host-context companion

`diagnostic_global_host_context.png` separates host-decomposition assessment
from detailed line QA:

- original spectrum, host galaxy, and reconstructed total model above;
- host-subtracted spectrum and final AGN plus emission-line model below;
- host-fraction annotations when available;
- shared rest-frame wavelength coverage and model-driven limits.

## Public API

The currently exported science-facing names include:

- spectrum and metadata: `Spectrum`, `SpectrumMetadata`;
- local configuration/results: `GaussianComponent`, `LorentzianComponent`,
  `IronTemplateConfig`, `LineComplexConfig`, `LocalFitConfig`, `FitResult`,
  `LocalFitResult`;
- global configuration/results: `PowerLawConfig`, `BalmerContinuumConfig`,
  `BalmerSeriesConfig`, `GlobalContinuumConfig`, `HbetaComplexConfig`,
  `MgIIComplexConfig`, `HalphaComplexConfig`, `UncertaintyConfig`,
  `GlobalContinuumResult`, `EmissionComplexResult`, `NeoFitWorkflowResult`;
- registries: `neofit.lines`, `neofit.recipes`, `LineDefinition`,
  `ComponentRecipe`, and `ComplexRecipe`;
- fitting: `fit_line_complex`, `fit_local`, `fit_global_continuum`,
  `fit_hbeta_complex`, `fit_mgii_complex`, `fit_halpha_complex`,
  `fit_global_lines`, and workflow wrappers;
- products: `GlobalQAPlotConfig`, `write_global_line_products`, and
  `write_global_hbeta_products`;
- templates: iron and Balmer template registry/list/load functions.

H-beta-named global functions and product writers remain compatibility wrappers.

## Validation Status

The current non-benchmark suite passes with `140` tests. Coverage includes:

- Gaussian, Lorentzian, continuum, iron, Balmer, velocity, and width derivative
  checks against centered finite differences;
- active- and inactive-bound reduced variable-projection Jacobians;
- variable-projection parity with the legacy joint optimizer;
- full and partial continuum coverage, clipping, and fixed/free Balmer widths;
- accepted and rejected [O III] wings and optional He II;
- Mg II and H-alpha recovery, coverage rules, tied kinematics, and fixed ratios;
- covariance and Monte Carlo workflows, including host refitting;
- optional-complex failure and optimizer fallback behavior;
- unit and metadata policies;
- output schemas and compatibility products;
- QA layout, fixed dimensions, annotations, colors, labels, legends, smoothing,
  host context, and one-to-four zoom-panel behavior;
- legacy import compatibility.
- line and recipe aliases, immutable overrides, adaptive selection,
  continuum-only fitting, optional H-beta policies, partial NIR coverage,
  fixed-ratio generic compilation, and generic Gaussian/Lorentzian
  derivatives.

Rows 0, 1, and 19 of the current DESI validation table are used for repeated
real-data QA inspection. Their plots are regenerated after visual changes.

## Current Limitations

- Only Mg II, H-beta/[O III], and H-alpha/[N II]/[S II] have dedicated adapter
  models; other initial recipes use the generic compiler.
- Complexes are fitted independently after any accepted H-beta synchronization;
  no cross-complex kinematic ties are imposed.
- The final continuum is fixed during Mg II and H-alpha fitting.
- Covariance errors condition on the fitted continuum and host unless a Monte
  Carlo mode explicitly refits them.
- Local windows remain independent and do not share continuum or kinematic
  parameters.
- Sparse Jacobian support for local fitting is API-compatible but is not a true
  block-sparse global implementation.
- Legacy line-table/FITS output compatibility is incomplete.
- Specialized C IV blue-wing/outflow selection and YAML recipe loading remain
  deferred.
- `QSOFitNew` remains required for legacy-only models and workflows.

## Next Steps

### 1. Freeze and document the current global workflow

- Add a concise user guide covering configuration, coverage rules, warnings,
  output columns, QA modes, and uncertainty interpretation.
- Document the exact scientific definitions of broad-profile flux, centroid,
  dispersion, FWHM, EW, and host fractions.
- Add a changelog entry and versioned compatibility expectations for public
  result fields and product filenames.

### 2. Broaden real-spectrum validation

- Build a curated validation set spanning redshift, S/N, host fraction, iron
  strength, line width, and wavelength coverage.
- Compare continuum samples, line fluxes, FWHM, EWs, and host fractions against
  trusted legacy fits and manual inspection.
- Define acceptance thresholds and persistent regression tables for science
  outputs, not only synthetic recovery.
- Record optimizer fallback rates and optional-complex failure rates.

### 3. Benchmark and profile production workloads

- Repeat warm-cache timing for representative objects and both solver modes.
- Report pPXF, continuum, H-beta, synchronization, optional-complex, product
  writing, and total runtime separately.
- Profile template broadening, bounded linear solves, covariance reconstruction,
  and repeated Monte Carlo setup.
- Decide whether further NumPy/SciPy optimization is warranted before adding
  JAX, Numba, or compiled extensions.

### 4. Validate and extend the recipe system deliberately

- Validate the optical-blue and Paschen/NIR recipes on real spectra, especially
  the He I 10833/Pa-gamma decomposition.
- Add specialized C IV blue-wing/outflow model selection.
- Consider YAML loading only after the Python recipe schema and warning
  semantics are stable.
- Add new global line complexes only with explicit coverage rules, derivative
  tests, science parity tests, and QA labels.
- Prioritize complexes based on the target survey/redshift distribution rather
  than attempting a full legacy line-table port at once.
- Decide whether future complexes need cross-complex kinematic ties or should
  remain independent.

### 5. Harden the new run-bundle workflow

The first batch-safe implementation is now available:

- `fit_object_to_store(...)` and `fit_batch(...)` write the same Parquet run
  bundle;
- Arrow scans DESI/SPARCL-like Parquet inputs in projected, bounded batches;
- SDSS, LAMOST, and IRAF FITS readers share one input registry;
- spawned process workers are the batch default, with one numerical-library
  thread per worker and a serial fallback on platforms without process
  semaphores;
- deterministic object seeds, resumable object/config pairs, isolated failure
  records, checksummed staging shards, and deterministic multi-job partitions
  are implemented;
- scalar tables compact automatically, while model shards remain canonical
  unless model compaction is requested;
- archived models can regenerate QA figures without refitting.

Next:

- validate the readers against a broader set of real survey files and unusual
  FITS layouts;
- profile Parquet microbatch size, worker count, and model-shard size on a
  representative catalog;
- add cluster-facing command-line wrappers after the Python API settles;
- define science-catalog specifications and derived-quantity calibrations
  separately, without freezing the authoritative long-form measurement schema.

### 6. Defer major backend changes

Do not add JAX, Numba, Cython, or a true sparse global optimizer until:

- the current NumPy/SciPy science results are frozen;
- real-spectrum validation thresholds are met;
- profiling identifies a dominant bottleneck that the new backend addresses;
- pure NumPy remains available as the reference implementation.

## Compatibility Policy

- Keep legacy `qsofitmore` imports working.
- Keep `QSOFitNew` available until feature and science validation justify a
  migration.
- Preserve existing dedicated measurement names, covariance ordering, output
  filenames, and H-beta compatibility wrappers. Generic summaries intentionally
  use per-recipe status instead of the removed aggregate success flags.
- Treat changes to scientific defaults, parameter bounds, coverage rules,
  warning codes, or model-selection criteria as user-visible changes requiring
  dedicated regression tests.
