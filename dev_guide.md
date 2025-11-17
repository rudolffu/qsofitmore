# dev_guide.md

This file acts as the instruction manual for contributors and coding agents (serves the role of AGENTS.md). Follow it for code style, testing, scope, and repo-specific conventions.

## Agent Instructions (Read First)

- Scope: This file is the instruction source for the entire repository. If a more deeply nested dev_guide.md exists, it takes precedence for files within its directory tree.
- Change scope: Prefer minimal, surgical changes. Avoid broad refactors, renames, or file moves unless requested.
- Code style: Follow PEP 8, format with `black`, lint with `flake8`. Keep changes consistent with surrounding code. Don’t add new heavy dependencies.
- Tests: Add/update tests when changing behavior. Use `pytest` with existing markers. Keep tests fast by default; isolate benchmarks behind `-m benchmark`.
- Data: Don’t add large binary assets. All data used by the library ships in `qsofitmore/data/`. External datasets (e.g., dust maps) must be fetched locally by the user or CI.
- Docs: Update this guide and `README.md` when changing usage, flags, or workflows.
- Backwards compatibility: Don’t break public API unless coordinated; deprecate with a clear path and tests.

## Overview

`qsofitmore` is a Python package for fitting UV–optical QSO (quasar) spectra, developed from PyQSOFit and extended for LAMOST and Galactic-plane use cases. It is a standalone spectroscopy fitting toolkit with template data bundled under `qsofitmore/data/`.

## Supported Python and Dependencies

- Python: 3.8–3.11
- Core dependencies (see `pyproject.toml`): `numpy`, `scipy`, `matplotlib`, `astropy`, `pandas`, `dustmaps`, `lmfit>=1.3.0`.
- Optional legacy dependency: `kapteyn` (install via `pip install .[legacy]`) to run the kmpfit backend.
- Install Kapteyn manually (requires Cython < 3.0) if you need kmpfit locally:
  ```bash
  pip install "cython<3.0"
  pip install https://www.astro.rug.nl/software/kapteyn/kapteyn-3.4.tar.gz
  ```
- Dust maps: fetch locally (no auto-download at import). Example for SFD98:
  ```python
  from dustmaps.config import config
  config['data_dir'] = './dustmaps_data'
  import dustmaps.sfd
  dustmaps.sfd.fetch()
  ```

## Installation

```bash
# Standard install
python -m pip install .

# Development install
python -m pip install -e .

# Optional dev tools
python -m pip install -e .[dev]
```

## Developer Workflows

- Quick tests (quiet, no benchmarks): `pytest -q`
- Focus tests by marker: `pytest -m lmfit -q`, `pytest -m "not benchmark"`
- Migration flags locally (lmfit is the default backend):
  - `export QSOFITMORE_USE_LMFIT=false` to force legacy kmpfit globally
  - `export QSOFITMORE_USE_LMFIT_CONTINUUM=false` or `..._LINES=false` to keep individual components on kmpfit
  - `export QSOFITMORE_USE_LMFIT_MC=true` to run Monte Carlo error estimation entirely through lmfit (natively supported when the lmfit backend is selected)
- Tox common runs:
  - `tox -e py311-lmfit` to validate lmfit infra
  - `tox -e py311-kmpfit` to validate legacy path
  - `tox -e benchmark` to run performance checks (marker-gated)
  - `tox -e lint` for formatting/lint/mypy checks

## Running Tests

This repository uses `pytest` with markers configured in `pytest.ini`.

- Quick run:
  ```bash
  pytest -q
  ```
- Exclude benchmarks (default via addopts):
  ```bash
  pytest -m "not benchmark"
  ```
- Useful markers: `benchmark`, `slow`, `integration`, `kmpfit`, `lmfit`, `migration`.

### Tox Environments

`tox.ini` defines matrix envs for Python 3.8–3.11 and for migration modes:

- Envs: `py{38,39,310,311}-{kmpfit,lmfit,migration}`
- Feature flags are set via environment variables (see next section).
- Tox pre-steps fetch Kapteyn and SFD dust maps into `./dustmaps_data`.

Run examples:
```bash
tox -e py311-lmfit
tox -e py310-migration
tox -e coverage
tox -e lint
```

### Feature Flags (Environment Variables)

Configured in `qsofitmore/config.py` and used in tests/CI. Defaults shown in parentheses.

- `QSOFITMORE_USE_LMFIT` (true): lmfit is the default backend; set to `false` to force kmpfit globally.
- `QSOFITMORE_USE_LMFIT_CONTINUUM` (true by default via the global flag): Override continuum backend selection.
- `QSOFITMORE_USE_LMFIT_LINES` (true by default via the global flag): Override line-fitting backend selection.
- `QSOFITMORE_USE_LMFIT_MC` (false): Enable lmfit-based Monte Carlo (line MC always defers to the active backend; this flag controls whether lmfit is preferred even when legacy code paths exist).
- `QSOFITMORE_VALIDATE_KMPFIT` (true): Validate against kmpfit results.
- `QSOFITMORE_BENCHMARK` (false): Enable performance benchmarks.
- Tolerances: `QSOFITMORE_RTOL` (1e-6), `QSOFITMORE_ATOL` (1e-8).

#### Linear Wavelength Mode + km/s Parameters

For easier parameter limits and interpretation, you can fit emission lines on a linear wavelength axis and specify Gaussian widths/offsets in km/s:

- `QSOFITMORE_WAVE_SCALE` (`log`|`linear`): choose the model axis for line fitting. Defaults to `log` for backward compatibility.
- `QSOFITMORE_VELOCITY_UNITS` (`lnlambda`|`km/s`): units for `inisig/minsig/maxsig/voff` in the line parameter tables. Use `km/s` for linear mode.
- `QSOFITMORE_NARROW_MAX_KMS` (default `1200`): maximum allowed Gaussian sigma for “narrow” components (applied to non-`*br` lines).

Example session:
```python
import os
os.environ['QSOFITMORE_WAVE_SCALE'] = 'linear'
os.environ['QSOFITMORE_VELOCITY_UNITS'] = 'km/s'
os.environ['QSOFITMORE_NARROW_MAX_KMS'] = '1200'
from qsofitmore import QSOFitNew
```

CSV/YAML conversion utilities (legacy ln(lambda) → km/s) live in `qsofitmore/line_params_io.py`:

```python
from qsofitmore.line_params_io import (
    convert_csv_lnlambda_to_kms, convert_yaml_lnlambda_to_kms,
    csv_to_fits, yaml_to_fits,
)

# Convert legacy tables
convert_csv_lnlambda_to_kms('examples/output/qsopar.csv', 'examples/output/qsopar.csv')
convert_yaml_lnlambda_to_kms('examples/output/qsopar.yaml', 'examples/output/qsopar.yaml')

# Write FITS; header will record VELUNIT=km/s when env is set
csv_to_fits('examples/output/qsopar.csv', 'examples/output/qsopar.fits')
```

See `qsofitmore/examples/2b-fit_qso_spectrum_linear.ipynb` for an end-to-end notebook using linear axis and km/s widths.

Examples:
```bash
# Default (lmfit) - disable Kapteyn validation for speed
export QSOFITMORE_VALIDATE_KMPFIT=false
pytest -m "not benchmark" -q

# Force the legacy kmpfit backend (requires Kapteyn)
export QSOFITMORE_USE_LMFIT=false
pytest -m kmpfit -q
```

## CI Notes

The workflow `.github/workflows/migration-tests.yml` runs:
- Infrastructure tests to validate config and flags.
- lmfit-only tests (no Kapteyn) and optional benchmarks.
- Integration checks for migration flags.

If you change feature flags, markers, or test paths, update this workflow and this guide accordingly.

## Code Architecture

### Main Class

- `QSOFitNew` (`qsofitmore/fitmodule.py`): primary fitting class for QSO spectra.
  - Init: arrays `lam`, `flux`, `err`, `z`; optional RA/DEC/name; SDSS helpers.
  - Methods: `Fit()`, `setmapname()`, `set_pl_pivot()`, `fromiraf()`, `fromsdss()`.
  - Note: `fitmodule.py` is large and performance-critical; keep changes localized.

### Core Modules

- `fitmodule.py`: main engine (currently uses Kapteyn `kmpfit`).
- `auxmodule.py`: helpers and plotting style (`sciplotstyle()`).
- `extinction.py`: extinction/dust laws.
- `config.py`: `migration_config` with feature flags and tolerances.

### Data Layout

- `qsofitmore/data/`
  - `bc03/`: Bruzual & Charlot SSP models
  - `pca/`: Yip et al. 2004 PCA templates
  - `iron_templates/`: FeII templates (BG92–VW01, Verner 2009, Garcia‑Rissmann 2012)
  - `balmer/`: Storey & Hummer (1995) Balmer series templates

## Usage Patterns

### Basic Fitting Workflow

```python
from qsofitmore import QSOFitNew

# Initialize from custom data (flux/err in 1e-17 units expected)
q = QSOFitNew(lam=wavelength, flux=flux*1e17, err=error*1e17,
              z=redshift, ra=ra, dec=dec, name='object_name',
              is_sdss=False, path=output_path)

# Or from IRAF multispec
q = QSOFitNew.fromiraf("spectrum.fits", redshift=z, telescope='LJT', path=output_path)

# Configure dust map
q.setmapname("planck")  # or "sfd", "planck14"

# Perform fit
q.Fit(name='fit_name',
      deredden=True,
      include_iron=True,
      iron_temp_name="V09",  # "BG92-VW01", "V09", "G12"
      broken_pl=True,
      BC=True,   # Balmer continuum + high-order Balmer lines
      MC=True,   # Monte Carlo errors (uses lmfit when enabled)
      save_result=True,
      plot_fig=True)
```

## Style, Lint, and Type Hints

- Format: `black qsofitmore tests`
- Lint: `flake8 qsofitmore tests`
- Optional static checks: `tox -e lint` (includes `mypy` if configured).
- Docstrings: NumPy or Google style are acceptable; keep concise and actionable.

## Contribution Guidelines

- Keep public API stable; document and test any behavior changes.
- Add tests for new features or bug fixes; use markers to control runtime.
- Prefer small, focused PRs; keep diffs minimal, avoid unrelated changes.
- Update `CHANGES.txt` and bump version in `pyproject.toml` when preparing a release.

## Important Notes and Pitfalls

- Flux units: expected in 1e-17 erg/s/cm^2/Å. Multiply raw erg/s/cm^2/Å by 1e17.
- External data: dust maps are not bundled; fetch locally and configure `dustmaps` data dir.
- Performance: Monte Carlo estimation can be expensive; keep defaults conservative in tests. When lmfit is the active backend, MC draws reuse the lmfit optimizer directly for both continuum and line fits.
- Examples: notebooks in `qsofitmore/examples/` demonstrate end-to-end usage; outputs are included for reference.

## Release Checklist

- Update `pyproject.toml` version and `CHANGES.txt`.
- Verify `README.md` stays accurate (dependencies, examples, flags).
- Ensure tests pass locally and in CI across environments.
- Tag release and draft GitHub release notes.
