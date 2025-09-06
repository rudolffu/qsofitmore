# dev_guide.md

This file provides guidance for developing in this repository.

## Overview

`qsofitmore` is a Python package for fitting UV-optical QSO (quasar) spectra, 
developed based on PyQSOFit with additional features for LAMOST quasar survey 
and quasars behind the Galactic plane. It's a standalone astronomical 
spectroscopy fitting package.

## Development Setup

### Installation for Development
```bash
# Install in editable mode
python -m pip install -e .

# Or use the standard installation
python -m pip install .
```

### Dependencies

Core dependencies are automatically handled, but some require special 
installation:

- `kapteyn` (requires cython): 
  `pip install cython && pip install https://www.astro.rug.nl/software/kapteyn/kapteyn-3.4.tar.gz`
- Dust maps setup: Download SFD98 dust map via `dustmaps.sfd.fetch()` after 
  configuring data directory

### Development Dependencies (Optional)

```bash
pip install pytest black flake8
```

## Code Architecture

### Main Classes

- **QSOFitNew** (`qsofitmore/fitmodule.py`): Primary fitting class that handles 
  QSO spectrum analysis
  - Initialization: Takes wavelength, flux, error arrays plus redshift and 
    coordinates
  - Key methods: `Fit()`, `setmapname()`, `set_pl_pivot()`, `fromiraf()`
  - Supports SDSS and custom spectra formats

### Core Modules

- **fitmodule.py**: Main fitting engine with QSOFitNew class (~4000+ lines)
- **auxmodule.py**: Auxiliary functions and utilities  
- **extinction.py**: Dust extinction calculations
- **config.py**: Configuration management

### Data Structure

- **qsofitmore/data/**: Template data files
  - `bc03/`: Bruzual & Charlot stellar population models
  - `pca/`: PCA templates from Yip et al. 2004
  - `iron_templates/`: FeII templates (Verner, Garcia-Rissmann)
  - `balmer/`: Balmer series templates (Storey & Hummer 1995)

### Key Features

- Fit high-order Balmer emission lines (n=6 to n=50)
- Multiple FeII templates: BG92-VW01 (default), V09, G12
- Optional broken power-law continuum model
- Support for different dust maps: SFD98, Planck 2014/2016
- Monte Carlo error estimation
- Host galaxy decomposition using PCA templates

## Usage Patterns

### Basic Fitting Workflow

```python
from qsofitmore import QSOFitNew

# Initialize from custom data
q = QSOFitNew(lam=wavelength, flux=flux*1e17, err=error*1e17, 
              z=redshift, ra=ra, dec=dec, name='object_name', 
              is_sdss=False, path=output_path)

# Or from IRAF multispec
q = QSOFitNew.fromiraf("spectrum.fits", redshift=z, 
                      telescope='LJT', path=output_path)

# Configure dust map
q.setmapname("planck")  # or "sfd", "planck14"

# Perform fit
q.Fit(name='fit_name', 
      deredden=True,
      include_iron=True,
      iron_temp_name="V09",  # "BG92-VW01", "V09", "G12"
      broken_pl=True,
      BC=True,  # Balmer continuum
      MC=True,  # Monte Carlo errors
      save_result=True,
      plot_fig=True)
```

### Testing

No specific test framework is configured in this repository. The examples in 
`qsofitmore/examples/` serve as functional tests and tutorials.

## Important Notes

- Flux units: Code expects flux in 10^-17 erg/s/cm^2/Å, multiply input by 
  1e17 if needed
- The package includes extensive spectral templates and requires ~100MB of 
  data files
- Output includes both FITS tables and plots (JPG/PDF) with fitting results
- Monte Carlo error estimation can be computationally intensive
- Supports both rest-frame and observed-frame input spectra