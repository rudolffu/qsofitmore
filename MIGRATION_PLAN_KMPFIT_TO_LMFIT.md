# Migration Plan: kapteyn.kmpfit to lmfit

This document outlines a comprehensive plan to migrate the qsofitmore codebase from `kapteyn.kmpfit` to `lmfit` for spectral fitting operations.

## Executive Summary

The qsofitmore package currently relies on `kapteyn.kmpfit` for non-linear least squares fitting. This migration plan addresses the replacement with `lmfit`, a modern Python fitting library that offers better maintainability, performance, and ecosystem integration.

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

### 1.1 Update Dependencies
```python
# Replace in requirements.txt and pyproject.toml
- kapteyn  # Remove
+ lmfit>=1.3.0  # Add
```

### 1.2 Update Imports
```python
# In fitmodule.py, replace:
from kapteyn import kmpfit
# With:
import lmfit
from lmfit import minimize, Parameters
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

**New lmfit equivalent:**
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

**New lmfit equivalent:**
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

## Phase 4: Testing and Validation

### 4.1 Regression Tests
```python
def test_continuum_fitting_compatibility():
    """Test that lmfit produces same results as kmpfit for continuum fitting"""
    
    # Load reference spectrum
    wave, flux, err = load_test_spectrum()
    
    # Fit with both methods
    result_kmpfit = fit_continuum_kmpfit(wave, flux, err)  # Original
    result_lmfit = fit_continuum_lmfit(wave, flux, err)    # New
    
    # Compare parameters (within tolerance)
    np.testing.assert_allclose(result_kmpfit.params, result_lmfit.params, 
                              rtol=1e-6, atol=1e-8)

def test_line_fitting_compatibility():
    """Test line fitting equivalence"""
    # Similar structure for line fitting tests
    pass

def test_monte_carlo_errors():
    """Test MC error estimation produces consistent uncertainties"""
    pass
```

### 4.2 Performance Benchmarking
```python
def benchmark_fitting_performance():
    """Compare performance between kmpfit and lmfit"""
    
    import time
    
    # Test data
    wave, flux, err = generate_test_spectrum()
    
    # Benchmark kmpfit
    start_time = time.time()
    for i in range(100):
        result_kmpfit = fit_spectrum_kmpfit(wave, flux, err)
    kmpfit_time = time.time() - start_time
    
    # Benchmark lmfit  
    start_time = time.time()
    for i in range(100):
        result_lmfit = fit_spectrum_lmfit(wave, flux, err)
    lmfit_time = time.time() - start_time
    
    print(f"kmpfit: {kmpfit_time:.3f}s, lmfit: {lmfit_time:.3f}s")
```

## Phase 5: Implementation Steps

### 5.1 Preparation
- [ ] Add lmfit to dependencies in `pyproject.toml` and `requirements.txt`
- [ ] Create feature flag for gradual migration
- [ ] Add comprehensive test suite covering current functionality
- [ ] Set up CI/CD pipeline to run both kmpfit and lmfit tests

### 5.2 Incremental Migration
- [ ] Start with continuum fitting (most isolated) - `fitmodule.py:335-350`
- [ ] Migrate line fitting functions - `fitmodule.py:2234-2258`
- [ ] Update Monte Carlo error estimation - Multiple locations
- [ ] Migrate remaining usage patterns
- [ ] Update all residual functions to lmfit format

### 5.3 Validation & Cleanup
- [ ] Run full regression test suite
- [ ] Performance benchmarking against kmpfit baseline
- [ ] Remove kapteyn dependency from all configuration files
- [ ] Update documentation and examples
- [ ] Update CLAUDE.md with new dependency information

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
2. **Dependencies**: `pyproject.toml`, `requirements.txt`, `setup.py`
3. **Documentation**: `README.md`, `CLAUDE.md`
4. **Tests**: New test files for regression testing
5. **Examples**: Update notebooks in `examples/` directory

## Contact and Support

This migration plan should be reviewed by:
- Original package maintainer
- Users of the fitting functionality
- CI/CD pipeline maintainer

For questions or concerns about this migration plan, please open an issue in the project repository.