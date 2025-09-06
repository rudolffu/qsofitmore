# Migration Plan: kapteyn.kmpfit to lmfit

This document outlines a comprehensive plan to migrate the qsofitmore codebase from `kapteyn.kmpfit` to `lmfit` for spectral fitting operations.

## Executive Summary

The qsofitmore package currently relies on `kapteyn.kmpfit` for non-linear least squares fitting. This migration plan addresses the replacement with `lmfit`, a modern Python fitting library that offers better maintainability, performance, and ecosystem integration.

## Current Status (Repository)

- Dependencies: `lmfit>=1.3.0` is already declared in `pyproject.toml`; `kapteyn` remains for now and will be removed at the end of migration.
- Feature flags: Implemented in `qsofitmore/config.py` (`migration_config`) with environment variables to progressively enable lmfit by component and control validation/benchmarks.
- Tests: `pytest` suite exists under `tests/` with markers (`kmpfit`, `lmfit`, `migration`, `benchmark`, etc.). Integration and utility tests scaffold comparisons and benchmarks.
- CI: `.github/workflows/migration-tests.yml` runs infrastructure checks, lmfit-only tests, optional benchmarks, and integration flag checks across Python 3.9–3.11.

Action: Implement lmfit code paths gated by `migration_config` flags, validate in CI, then retire `kapteyn`.

## Migration Analysis Summary

### Current kapteyn.kmpfit Usage Patterns

**Found 6 main usage locations:**
1. **Continuum fitting**: Lines 335, 342 - Main continuum model fitting
2. **Continuum Monte Carlo**: Line 1115 - Error estimation via MC sampling  
3. **Line fitting**: Line 2234 - Individual emission line fitting
4. **Line Monte Carlo**: Lines 1510, 2352 - Line parameter uncertainty estimation

**Key Usage Patterns:**
- `kmpfit.Fitter(residuals=func, data=(x,y,err), maxiter=50)`
- Parameter bounds via `parinfo` with `{'limits': (min, max)}` dictionaries
- Initial parameter guesses via `params0` 
- Parameter constraints via custom logic in residual functions
- Results accessed via `.params` attribute

### Key API Differences

| Aspect | kapteyn.kmpfit | lmfit |
|--------|----------------|-------|
| Parameter Specification | Simple arrays + parinfo dicts | Rich Parameters objects |
| Bounds Handling | `{'limits': (min, max)}` | `min=val, max=val` in Parameters |
| Residual Function | `func(params, data)` | `func(params, *args, **kwargs)` |
| Results Access | `.params` array | `.params` dict-like object |
| Error Estimation | Manual MC implementation | Built-in uncertainty methods |

## Migration Strategy

## Phase 1: Dependencies and Infrastructure

### 1.1 Dependencies
Already added: `lmfit>=1.3.0` in `pyproject.toml`. Keep `kapteyn` until parity is validated, then remove it (and Cython pin in README).

### 1.2 Imports and Flags
```python
# In fitmodule.py (during migration):
from kapteyn import kmpfit                    # legacy path (kept)
from lmfit import minimize, Parameters        # new path
from .config import migration_config          # feature flags
```

Guard usage with flags:
```python
if migration_config.use_lmfit_continuum:
    # call lmfit-based continuum
else:
    # call kmpfit-based continuum
```

## Phase 2: Core Function Migrations

### 2.1 Continuum Fitting Migration

Current kmpfit pattern (lines 335-350):
```python
conti_fit = kmpfit.Fitter(residuals=self._residuals, data=(wave[tmp_all], flux[tmp_all], err[tmp_all]))
tmp_parinfo = [{'limits': (0., 10.**10)}, {'limits': (1200., 10000.)}, ...]
conti_fit.parinfo = tmp_parinfo
conti_fit.fit(params0=pp0)
```

**New lmfit equivalent (sketch):**
```python
def _fit_continuum_lmfit(self, wave, flux, err, pp0, param_bounds):
    """Replacement for continuum kmpfit fitting"""
    
    # Create Parameters object
    params = Parameters()
    param_names = ['amp1', 'wave1', 'offset1', 'amp2', 'wave2', 'offset2', 
                   'pl_norm', 'pl_index', 'bc_norm', 'bc_temp', 'fe_norm',
                   'poly1', 'poly2', 'poly3']
    
    if self.broken_pl:
        param_names.append('pl_index2')
    
    # Add parameters with bounds
    for i, (name, init_val) in enumerate(zip(param_names, pp0)):
        bounds = param_bounds[i] if param_bounds[i] else {}
        if bounds:
            params.add(name, value=init_val, min=bounds['limits'][0], max=bounds['limits'][1])
        else:
            params.add(name, value=init_val)
    
    # Define residual function for lmfit
    def residual_func(params, wave, flux, err):
        param_vals = [params[name].value for name in param_names]
        return (flux - self._f_conti_all(wave, param_vals)) / err
    
    # Perform fit
    # Prefer Levenberg–Marquardt for parity with kmpfit
    result = minimize(residual_func, params, args=(wave, flux, err), method='leastsq')
    
    # Store results in compatible format
    conti_fit = type('FitResult', (), {})()
    conti_fit.params = [result.params[name].value for name in param_names]
    conti_fit.stderr = [result.params[name].stderr for name in param_names]
    conti_fit.success = result.success
    conti_fit.result = result
    
    return conti_fit
```

### 2.2 Line Fitting Migration

Current pattern (lines 2234-2258):
```python
line_fit = kmpfit.Fitter(self._residuals_line, data=(np.log(self.wave[ind_n]), line_flux[ind_n], self.err[ind_n]))
# Complex parameter setup loop...
line_fit.parinfo = line_fit_par  
line_fit.fit(params0=line_fit_ini)
```

**New lmfit equivalent (sketch):**
```python
def _do_line_lmfit(self, linelist, line_flux, ind_line, ind_n, nline_fit, ngauss_fit):
    """Replacement for line kmpfit fitting"""
    
    params = Parameters()
    param_names = []
    
    # Build parameters dynamically (same logic as original)
    for n in range(nline_fit):
        for nn in range(ngauss_fit[n]):
            base_name = f'line_{n}_gauss_{nn}'
            
            # Amplitude, wavelength, sigma
            amp_name = f'{base_name}_amp'
            wave_name = f'{base_name}_wave'  
            sig_name = f'{base_name}_sig'
            
            param_names.extend([amp_name, wave_name, sig_name])
            
            # Initial values
            params.add(amp_name, value=0., min=0., max=1e10)
            
            lambda_center = np.log(linelist['lambda'][ind_line][n])
            lambda_range = linelist['voff'][ind_line][n]
            params.add(wave_name, value=lambda_center, 
                      min=lambda_center - lambda_range,
                      max=lambda_center + lambda_range)
            
            params.add(sig_name, value=linelist['inisig'][ind_line][n],
                      min=linelist['minsig'][ind_line][n],
                      max=linelist['maxsig'][ind_line][n])
    
    # Apply parameter constraints (tied parameters)
    self._apply_line_constraints_lmfit(params, param_names)
    
    # Define residual with parameter tying logic
    def residual_func(params, xval, yval, weight):
        param_vals = self._extract_line_params_lmfit(params, param_names)
        return (yval - self.Manygauss(xval, param_vals)) / weight
    
    # Fit
    result = minimize(residual_func, params, 
                     args=(np.log(self.wave[ind_n]), line_flux[ind_n], self.err[ind_n]),
                     method='leastsq')
    
    # Return compatible result object
    line_fit = type('FitResult', (), {})()
    line_fit.params = self._extract_line_params_lmfit(result.params, param_names)
    line_fit.result = result
    line_fit.success = result.success
    
    return line_fit
```

### 2.3 Monte Carlo Error Estimation

Current pattern (lines 1115-1117, 1510-1512):
```python
for tra in range(n_trails):
    flux = y + np.random.randn(len(y)) * err
    conti_fit = kmpfit.Fitter(residuals=self._residuals, data=(x, flux, err), maxiter=50)
    conti_fit.parinfo = pp_limits
    conti_fit.fit(params0=pp0)
```

**New lmfit equivalent:**
```python
def _monte_carlo_errors_lmfit(self, x, y, err, params_best, n_trails):
    """Monte Carlo error estimation using lmfit"""
    
    param_samples = []
    
    for tra in range(n_trails):
        # Add noise
        flux_noisy = y + np.random.randn(len(y)) * err
        
        # Create parameters from best fit
        params_mc = params_best.copy()
        
        # Define residual function
        def residual_func(params, x, flux, err):
            param_vals = [params[name].value for name in params.keys()]
            return (flux - self._f_conti_all(x, param_vals)) / err
        
        # Fit noisy data
        try:
            result = minimize(residual_func, params_mc, args=(x, flux_noisy, err), 
                            method='leastsq', max_nfev=50)
            if result.success:
                param_samples.append([result.params[name].value for name in params_mc.keys()])
        except:
            continue
    
    # Calculate statistics
    param_samples = np.array(param_samples)
    param_std = np.std(param_samples, axis=0)
    
    return param_std, param_samples
```

Note: For continuum fitting under lmfit, we can skip Monte Carlo to save time and use lmfit’s parameter uncertainties directly (`param.stderr`). The implementation in `fitmodule.py` uses these when `QSOFITMORE_USE_LMFIT_CONTINUUM=true`, populating parameter errors and setting luminosity errors to zero (as a placeholder until analytic propagation is added).

### 2.4 Parameter Tying and Constraints (Details)

The line list uses group indices for constraints:
- `vindex`: velocity (center) tying group; same group => same logged center (`log(lambda)`) or fixed offset.
- `windex`: width tying group; same group => same `sigma`.
- `findex`: flux ratio group; `fvalue` gives ratio relative to group anchor.

Implementation in lmfit:
- Velocity tie: set expressions so members share a base parameter. Example: `params['Ha_na_wave'].expr = 'Ha_br_wave'`.
- Width tie: `params['OIII4959_sig'].expr = 'OIII5007_sig'`.
- Flux ratios: `params['NII6585_amp'].expr = '3.0 * NII6549_amp'` or using the provided `fvalue` with an agreed anchor.

Choose one anchor per group deterministically (first encounter) and set other parameters’ `.expr` to tie to it. For allowable small offsets (e.g., Doppler), include additive terms or pre-transform inputs appropriately.

Note: the code currently uses `np.log(wave)` in line residuals. Keep parity: define tie expressions in terms of log-wavelength parameters.

### 2.5 Residuals and Weighting

- Weighting: keep `(model - data) / err` residuals exactly as in kmpfit.
- Method: default to `method='leastsq'` for closest behavior; `least_squares` is an alternative when strict bounds enforcement is required.
- Scaling: if convergence differs, consider normalizing parameter scales in initial guesses.

## Phase 3: Interface Compatibility

### 3.1 Maintain Backward Compatibility
```python
class LmfitCompatibilityWrapper:
    """Wrapper to maintain API compatibility during migration"""
    
    def __init__(self):
        self.use_lmfit = True  # Feature flag
    
    def create_fitter(self, residuals, data, maxiter=None):
        if self.use_lmfit:
            return LmfitFitterWrapper(residuals, data, maxiter)
        else:
            return kmpfit.Fitter(residuals=residuals, data=data, maxiter=maxiter)

class LmfitFitterWrapper:
    """Wrapper to make lmfit behave like kmpfit.Fitter"""
    
    def __init__(self, residuals, data, maxiter=None):
        self.residuals = residuals
        self.data = data  
        self.maxiter = maxiter
        self.parinfo = None
        self.params = None
    
    def fit(self, params0):
        # Convert kmpfit-style parinfo to lmfit Parameters
        params = self._convert_parinfo_to_lmfit_params(params0, self.parinfo)
        
        # Run lmfit
        def residual_wrapper(params):
            param_vals = [params[f'p{i}'].value for i in range(len(params0))]
            return self.residuals(param_vals, self.data)
        
        result = minimize(residual_wrapper, params, method='leastsq')
        
        # Store results in kmpfit-compatible format
        self.params = [result.params[f'p{i}'].value for i in range(len(params0))]
        self.success = result.success
        self.result = result
```

Alternatively (preferred in this repo): keep separate kmpfit/lmfit code paths for continuum, lines, and MC, and switch between them via `migration_config` flags. This keeps the public API unchanged and avoids a heavy wrapper abstraction.

## Phase 4: Testing and Validation

### 4.1 Regression Tests
The repository already includes scaffolding for migration tests:
- `tests/test_migration_integration.py`: feature flags and workflow scaffolding.
- `tests/test_continuum_fitting.py`: continuum structure and validation hooks.
- `tests/test_line_fitting.py`: line fitting structure and validation hooks.
- `tests/test_utilities.py`: spectrum generation and comparison helpers.

Actions to implement once lmfit paths are added:
- Add paired tests that run both kmpfit and lmfit paths on the same synthetic data, asserting parameter agreement within `migration_config.rtol/atol`.
- Use markers `kmpfit` and `lmfit` (or environment flags) to select paths.
- Add benchmarks guarded by `-m benchmark` to compare runtime.

### 4.2 Performance Benchmarking
Prefer using `pytest-benchmark` (already configured) for comparable measurements; do not add ad-hoc timers.

## Phase 5: Implementation Steps

### 5.1 Preparation
- [x] Add lmfit to dependencies in `pyproject.toml`
- [x] Create feature flags in `qsofitmore/config.py`
- [x] Add test suite scaffolding and markers
- [x] Set up CI workflow (`migration-tests.yml`)

### 5.2 Incremental Migration
- [ ] Implement lmfit continuum path guarded by `use_lmfit_continuum` (see `fitmodule.py:339` et al.)
- [ ] Implement lmfit line fitting path guarded by `use_lmfit_lines` with parameter ties
- [ ] Implement lmfit MC errors guarded by `use_lmfit_mc`
- [ ] Keep kmpfit behavior as default until parity is met
- [ ] Add residual wrappers for shared logic; keep log-wavelength handling identical

### 5.3 Validation & Cleanup
- [ ] Enable `QSOFITMORE_VALIDATE_KMPFIT=true` and compare parameters within tolerances
- [ ] Performance benchmarking with `pytest -m benchmark`
- [ ] Remove `kapteyn` dependency and Kapteyn-specific code paths
- [ ] Update `README.md` and `dev_guide.md` to remove legacy instructions

## Recent Changes (in this migration)

- Added lmfit paths for continuum and line fitting, gated by feature flags.
- Always include error columns in outputs; when using lmfit and `MC=False`, use lmfit stderr-based errors for parameters and analytic errors for broad line metrics.
- Tightened bounds for Fe and Balmer continuum/high-order norms (≤ 1e3) for stability; left PL norm wide.
- Fixed broken power-law model to be continuous at 4661 Å and aligned with input wavelengths.
- Global flag `use_lmfit` now cascades at runtime to enable both components.

## Open To‑Dos

1. Narrow-line uncertainty propagation (lmfit, MC=False)
   - Propagate uncertainties for narrow-line metrics (fwhm/sigma/ew/peak/area) using local covariance of each 3‑parameter group.
2. Optional: lmfit MC for lines
   - Mirror the implemented continuum MC for lines when `use_lmfit_mc=True` for full parity.
3. Optional: Analytic λLλ uncertainties for continuum when MC=False
   - Re‑enable Jacobian+covariance propagation for λLλ if desired (currently skipped for speed).
4. Configurable bounds
   - Make Fe/BC norm upper limits configurable (env vars or Fit args) to ease tuning across datasets.
5. Tests
   - Add focused tests to assert: (a) flag gating, (b) presence of error columns with lmfit/MC=False, (c) realistic BC/Fe behavior under bounds, (d) parity windows for log‑wavelength usage.
6. Docs
   - Reflect current default bounds and error reporting behavior in `dev_guide.md` and `README.md`.

## Commit Guidance

- Keep changes focused on migration; avoid over‑eager refactors.
- Preserve output schema stability (value + `_err` pairs) for downstream consumers.

## Key Benefits of Migration

1. **Modern Python Ecosystem**: lmfit is actively maintained, kapteyn is legacy
2. **Better Error Handling**: More robust parameter bounds and constraints
3. **Enhanced Uncertainty Estimation**: Built-in support for parameter correlations
4. **Improved Performance**: Modern optimization algorithms
5. **Better Documentation**: Comprehensive API documentation and examples
6. **Reduced Dependencies**: Eliminates complex kapteyn/cython installation requirements

## Risk Assessment

### Low Risk
- Continuum fitting migration (isolated functions)
- Parameter bounds handling (direct mapping)

### Medium Risk
- Monte Carlo error estimation (requires careful validation)
- Parameter tying logic (complex constraint handling)

### High Risk
- Line fitting with complex parameter relationships
- Numerical precision differences between libraries

### Mitigation Strategy
- Gradual rollout with feature flags
- Extensive regression testing
- Side-by-side comparison during transition period
- Rollback capability if issues arise

Additional mitigations:
- Use deterministic random seeds in MC tests to stabilize comparisons.
- Compare derived quantities (e.g., FWHM, EWs, fluxes) in addition to raw parameters.

## Success Criteria

1. **Functionality**: All existing fitting operations produce equivalent results
2. **Performance**: No significant performance degradation (< 20% slower acceptable)
3. **Reliability**: All tests pass with new lmfit implementation
4. **Maintainability**: Code is cleaner and more maintainable
5. **Dependencies**: Successful removal of kapteyn dependency

## Timeline Estimate

- **Phase 1-2**: 2-3 weeks (Core migration)
- **Phase 3**: 1 week (Compatibility layer)
- **Phase 4**: 2 weeks (Testing and validation)
- **Phase 5**: 1 week (Cleanup and documentation)

**Total: 6-7 weeks**

## Files Requiring Changes

1. **Primary**: `qsofitmore/fitmodule.py` - All kmpfit usage
2. **Dependencies**: `pyproject.toml` (remove Kapteyn when done)
3. **Documentation**: `README.md`, `dev_guide.md`
4. **Tests**: Implement assertions in existing scaffolds for regression testing
5. **Examples**: Update notebooks in `examples/` directory

## Contact and Support

This migration plan should be reviewed by:
- Original package maintainer
- Users of the fitting functionality
- CI/CD pipeline maintainer

For questions or concerns about this migration plan, please open an issue in the project repository.
