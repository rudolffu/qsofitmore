# QSOFITMORE
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15571037.svg)](https://doi.org/10.5281/zenodo.15571037)  
A python package for fitting UV-optical QSO spectra. This package is developed based on [`PyQSOFit`](https://github.com/legolason/PyQSOFit) (v1.1), with some additional features for the LAMOST quasar survey ([Jin et al. 2023](https://ui.adsabs.harvard.edu/abs/2023ApJS..265...25J/abstract)) and the survey for quasars behind the Galactic plane ([Fu et al. 2022](https://ui.adsabs.harvard.edu/abs/2022ApJS..261...32F/abstract)). `qsofitmore` is now a standalone package with the same GNU license as `PyQSOFit`. For the latest version of `PyQSOFit`, please see https://github.com/legolason/PyQSOFit. 

### Features  
- Fit high-order (n_upper=6 to n_upper=50) Balmer emission line series using templates from Storey & Hummer (1995).
- Add FeII template from Verner et al. (2009) within [2000, 10000] AA.
- Add FeII template from Garcia-Rissmann et al. (2012) within [8100, 11500] AA.
- Add an optional broken power-law model in the continuum fitting process.
- Reading non-SDSS spectra without plateid, mjd and fiberid. 
- Narrow line measurements in the line MC process (which enables estimating uncertainties for narrow line parameters). 
- Dust maps and extinction laws other than SFD98 and CCM. 
- LaTeX rendering for line names in plots. 

## 1. Installation
Dependencies:
```
astropy
lmfit
dustmaps
uncertainties
```

Assuming you have anaconda installed (`astropy` included), the following steps demonstrate how to install dependencies above.
- Install these packages via conda:
```
conda install -c conda-forge lmfit dustmaps uncertainties
```

You can also install these packages via pip:
```
pip install lmfit dustmaps uncertainties
```

Download the SFD98 dust map:
```python
from dustmaps.config import config
config['data_dir'] = '/path/to/store/maps/in'

import dustmaps.sfd
dustmaps.sfd.fetch()
```

Check https://dustmaps.readthedocs.io/en/latest/installation.html for more dust maps.

After installing the dependencies, download and set up the `qsofitmore` package.

One-line installation with pip:

```
pip install git+https://github.com/rudolffu/qsofitmore.git
```

Alternatively, clone the repository and install manually:

```
git clone https://github.com/rudolffu/qsofitmore.git 
cd qsofitmore 
python -m pip install .
# for development use:
python -m pip install -e .
```

### Using legacy kmpfit (optional)

By default, qsofitmore uses the lmfit-based fitting paths. The legacy kmpfit path is still available but requires the optional Kapteyn package.

- Install Kapteyn (optional):
  ```bash
  pip install "cython<3.0" # suggested by Kapteyn maintainer
  pip install https://www.astro.rug.nl/software/kapteyn/kapteyn-3.4.tar.gz
  # or via optional extra
  pip install .[legacy]
  ```

- Enable legacy kmpfit at runtime (if you prefer kmpfit over lmfit):
  ```python
  from qsofitmore.config import migration_config
  migration_config.use_lmfit = False  # use legacy kmpfit path
  ```

- If Kapteyn is not installed and you try to use kmpfit, qsofitmore will print a friendly warning on import and raise a clear ImportError when a kmpfit path is invoked. To avoid this, either install Kapteyn (above) or enable lmfit:
  ```python
  from qsofitmore.config import migration_config
  migration_config.use_lmfit = True
  ```

- CI and tox:
  - kmpfit-specific tox envs install Kapteyn automatically.
  - Example: `tox -e py311-kmpfit`

## 2. Tutorial
 
This tutorial can be run under `examples` directory of `qsofitmore`. 

### 2.1 Generate line parameter file

The following script of generating `qsopar.fits` is based on https://github.com/legolason/PyQSOFit/blob/master/example/example.ipynb


```python
import numpy as np
from astropy.io import fits
from astropy.table import Table
path='./output/'

newdata = np.rec.array([(6564.61,'Ha',6400.,6800.,'Ha_br',3,5e-3,0.003,0.01,0.005,0,0,0,0.05),\
                        (6564.61,'Ha',6400.,6800.,'Ha_na',1,1e-3,5e-4,0.0017,0.01,1,1,0,0.002),\
                        (6549.85,'Ha',6400.,6800.,'NII6549',1,1e-3,2.3e-4,0.0017,5e-3,1,1,1,0.001),\
                        (6585.28,'Ha',6400.,6800.,'NII6585',1,1e-3,2.3e-4,0.0017,5e-3,1,1,1,0.003),\
                        (6718.29,'Ha',6400.,6800.,'SII6718',1,1e-3,2.3e-4,0.0017,5e-3,1,1,2,0.001),\
                        (6732.67,'Ha',6400.,6800.,'SII6732',1,1e-3,2.3e-4,0.0017,5e-3,1,1,2,0.001),\
                        (4862.68,'Hb',4640.,5100.,'Hb_br',3,5e-3,0.003,0.01,0.003,0,0,0,0.01),\
                        (4862.68,'Hb',4640.,5100.,'Hb_na',1,1e-3,2.3e-4,0.0017,0.01,1,1,0,0.002),\
                        (4960.30,'Hb',4640.,5100.,'OIII4959',1,1e-3,2.3e-4,0.0017,0.01,1,1,0,0.002),\
                        (5008.24,'Hb',4640.,5100.,'OIII5007',1,1e-3,2.3e-4,0.0017,0.01,1,1,0,0.004),\
                        (4955.30,'Hb',4640.,5100.,'OIII4959w',1,1e-3,2.3e-4,0.0017,0.01,2,2,0,0.001),\
                        (4995.24,'Hb',4640.,5100.,'OIII5007w',1,1e-3,2.3e-4,0.0017,0.01,2,2,0,0.002),\
                        (4341.68,'Hg',4250.,4440.,'Hg_br',1,5e-3,0.004,0.025,0.0017,0,0,0,0.05),\
                        (4341.68,'Hg',4250.,4440.,'Hg_na',1,1e-3,2.3e-4,0.0017,5e-3,1,1,0,0.001),\
                        (3728.48,'OII',3650.,3800.,'OII3728',1,1e-3,3.333e-4,0.0017,0.01,1,1,0,0.001),\
                        (3426.84,'NeV',3380.,3480.,'NeV3426',1,1e-3,3.333e-4,0.0017,0.005,0,0,0,0.001),\
                        (2798.75,'MgII',2700.,2900.,'MgII_br',2,5e-3,0.004,0.015,0.0017,0,0,0,0.05),\
                        (2798.75,'MgII',2700.,2900.,'MgII_na',1,1e-3,2.3e-4,0.0017,0.01,0,0,0,0.002),\
                        (1908.73,'CIII',1700.,1970.,'CIII_br',2,5e-3,0.004,0.015,0.015,99,0,0,0.01),\
                        (1549.06,'CIV',1500.,1700.,'CIV_br',3,5e-3,0.004,0.015,0.015,0,0,0,0.05),\
                        ],\
                     formats='float32,a20,float32,float32,a20,float32,float32,float32,float32,\
                     float32,float32,float32,float32,float32',\
                     names='lambda,compname,minwav,maxwav,linename,ngauss,inisig,minsig,maxsig,voff,vindex,windex,findex,fvalue')
#------header-----------------
hdr = fits.Header()
hdr['lambda'] = 'Vacuum Wavelength in Ang'
hdr['minwav'] = 'Lower complex fitting wavelength range'
hdr['maxwav'] = 'Upper complex fitting wavelength range'
hdr['ngauss'] = 'Number of Gaussians for the line'
hdr['inisig'] = 'Initial guess of linesigma [in lnlambda]'
hdr['minsig'] = 'Lower range of line sigma [lnlambda]'  
hdr['maxsig'] = 'Upper range of line sigma [lnlambda]'
hdr['voff  '] = 'Limits on velocity offset from the central wavelength [lnlambda]'
hdr['vindex'] = 'Entries w/ same NONZERO vindex constrained to have same velocity'
hdr['windex'] = 'Entries w/ same NONZERO windex constrained to have same width'
hdr['findex'] = 'Entries w/ same NONZERO findex have constrained flux ratios'
hdr['fvalue'] = 'Relative scale factor for entries w/ same findex'
#------create fits table-------
hdu = fits.BinTableHDU(data=newdata,header=hdr,name='data')
hdu.writeto(path+'qsopar.fits',overwrite=True)
```

### 2.2 Import `QSOFitNew` class from `qsofitmore` 


```python
from qsofitmore import QSOFitNew
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.table import Table
```

The output path (`path`) should contain a line list file (`qsopar.fits` generated in `1-make_parlist.ipynb`). The output files (including fits table and plots) are stored in `path`. 


```python
path = "./output/"
```

### 2.3 Initialise an instance of `QSOFitNew` from a custom spectrum 

#### a) From numpy-array like data 
We can read an example spectrum in csv format using `pandas`, and load the data to `QSOFitNew` manually. The data should contain wavelength (in Å), flux and flux error. In this example, the flux and error are in the unit of erg/s/cm^2/Å. Because the code assumes the flux is in unit of 10^-17 erg/s/cm^2/Å, we need to multiply the flux and error by 1e17. The redshift, RA, DEC, and name of the quasar are also required. The `is_sdss` parameter is set to `False` because this spectrum is not from SDSS. The object `J001554.18+560257.5` is from the suvey for quasars behind the Galactic plane ([Fu et al. 2022](https://ui.adsabs.harvard.edu/abs/2022ApJS..261...32F/abstract)), and was observed with the 2.4m Lijiang Telescope (LJT) in China. The `path` parameter is set to the output path where the fits file and plots will be saved.


```python
df = pd.read_csv("./data/spec_J001554.18+560257.5_LJT.csv")
```


```python
df
```


| Index | lam         | flux         | err          |
|-------|-------------|--------------|--------------|
| 0     | 3665.797515 | 7.947896e-16 | 1.031744e-16 |
| 1     | 3668.657947 | 9.944345e-16 | 1.128481e-16 |
| 2     | 3671.518379 | 7.201746e-16 | 1.086394e-16 |
| 3     | 3674.378811 | 7.498753e-16 | 1.073345e-16 |
| 4     | 3677.239243 | 8.567060e-16 | 1.078132e-16 |
| ...   | ...         | ...          | ...          |
| 1895  | 9086.315977 | 3.270070e-16 | 3.116500e-17 |
| 1896  | 9089.176409 | 3.494074e-16 | 3.172321e-17 |
| 1897  | 9092.036841 | 3.376591e-16 | 3.185541e-17 |
| 1898  | 9094.897273 | 2.639512e-16 | 3.125944e-17 |
| 1899  | 9097.757705 | 3.047675e-16 | 3.164424e-17 |

1900 rows × 3 columns





```python
q = QSOFitNew(lam=df.lam, flux=df.flux*1e17, err=df.err*1e17, 
              z=0.1684, ra=3.97576206, dec=56.04931383,
              name='J001554.18+560257.5', is_sdss=False, path=path)
```

#### b) From IRAF multispec

If you have a spectrum generated by `IRAF`/`PyRAF`, in which case the 4 bands of the fits file are:  
BANDID1 = 'spectrum - background fit, weights variance, clean no'               
BANDID2 = 'raw - background fit, weights none, clean no'                        
BANDID3 = 'background - background fit'                                         
BANDID4 = 'sigma - background fit, weights variance, clean no'  
The first and fourth bands are flux and flux error, respectively, in unit erg/s/cm^2/Å.
You can simply load the data with the classmethod `QSOFitNew.fromiraf`, which does the unit conversion automatically.


```python
q = QSOFitNew.fromiraf(
      "./data/spec_J001554.18+560257.5_LJT.fits",
      redshift=0.1684, telescope='LJT', path=path)
```


### 2.4 Fit the spectrum 

#### Choose a dust map  
The code supports three Galactic extinction maps:

 - `"sfd"`: Schlegel, Finkbeiner & Davis (1998) dust map  (this is the default)  
 - `"planck"` or `"planck16"`: Planck GNILC dust map (Planck Collaboration 2016)  
 - `"planck14"`: Planck dust map (Planck Collaboration 2014)

```python
# use the default SFD98 map
q.setmapname("sfd")

# use the default Planck GNILC (2016) map
q.setmapname("planck")     # same as "planck16"

# explicitly use the older Planck (2014) map
q.setmapname("planck14")
```

 If you select `"planck"`/`"planck16"` or `"planck14"`, make sure you have:
 
 1. Installed and configured `dustmaps`  
 2. Downloaded the Planck dust maps via:
    ```python
    from dustmaps.config import config
    config['data_dir'] = '/path/to/store/maps'
    import dustmaps.planck
    dustmaps.planck.fetch()
    ```

#### Apply `q.Fit()`  

Derived quantities of narrow and broad lines, including FWHM, sigma, EW, and integrated flux (area), are calculated during the fitting process and stored in the output fits file. By specifying `MC = True` when calling `q.Fit()`, the code will also calculate the uncertainties of these quantities using Monte Carlo simulations. The uncertainties are stored in the output fits file as well.


```python
q.setmapname("planck")
q.Fit(name              = 'J001554.18+560257.5_LJT_ADV',
      deredden          = True,
      wave_range        = None,
      wave_mask         = None,
      Fe_flux_range     = [4434, 4684] , # Wavelength range of FeII flux saved to the output file
      decomposition_host= True,
      Mi                = None,
      npca_gal          = 5,
      npca_qso          = 20,
      include_iron      = True,          # enable FeII fitting
      iron_temp_name    = "V09",         # options: "BG92-VW01", "V09", "G12"
      poly              = False,
      broken_pl         = True,          # enable broken power-law
      BC                = True,          # enable Balmer continuum and high-order Balmer lines
      MC                = True,         # optional: enable Monte Carlo error estimation
      n_trails          = 20,
      linefit           = True,
      tie_lambda        = True,
      tie_width         = True,
      tie_flux_1        = True,
      tie_flux_2        = True,
      save_result       = True,
      plot_fig          = True,
      save_fig          = True,
      plot_line_name    = True,
      plot_legend       = True,
      save_fits_name    = None)
```

    
![jpg](qsofitmore/examples/output/plot_fit_J001554.18+560257.5_LJT_ADV.jpg)
    

#### Print fitting results
Try: 
`q.result_table`




### 2.5 Other options in the fitting process

#### a) The broken power-law model

The broken power-law model is an optional feature in the continuum fitting process. It is enabled by setting `broken_pl = True` in `q.Fit()`. The default is `False`.

#### b) Choose the FeII template
The FeII template can be chosen by setting `iron_temp_name` in `q.Fit()`. The options are:
- `"BG92-VW01"`: the Boroson & Green (1992) and Vestergaard & Wilkes (2001) template (default). 
- `"V09"`: the Verner et al. (2009) template, which covers the wavelength range from 2000 to 10000 Å.
- `"G12"`: the Garcia-Rissmann et al. (2012) template, which covers the wavelength range from 8100 to 11500 Å.

#### c) The Storey & Hummer (1995) Balmer emission line series templates

The Storey & Hummer (1995) Balmer emission line series templates are enabled by setting `BC = True` in `q.Fit()`. The default value of `BC` is `False`. The default electron density is 10^9 cm^-3, and the default electron temperature is 10^4 K. To change the electron density to 10^10 cm^-3, use `q.set_log10_electron_density(ne=10)` before calling `q.Fit()`. Currently, only 10^9 and 10^10 cm^-3 are supported.

#### d) Configurable power-law pivot wavelength (single power-law model only)
By default the continuum power-law is normalized at 3000 Å. If your spectrum lies mostly in the NIR (e.g. > 7000 Å), you can change that pivot:

```python
# normalize continuum at 9800 Å instead of the default 3000 Å
q.set_pl_pivot(9800.0)
q.Fit(…)
```
This ensures all PL continuum calculations use your chosen pivot wavelength.


### Updates from v1.2.1 to v1.2.2:
- Added the Garcia-Rissmann et al. (2012) FeII template within [8100, 11500] AA.
- Added the `set_pl_pivot` method to set the pivot wavelength for the power-law continuum model.


### Updates from v1.1.0 to v1.2.1:
- Added the Verner et al. (2009) FeII template within [2000, 10000] AA.
- Added the Storey & Hummer (1995) Balmer emission line series templates from n_upper=6 to n_upper=50.
- Merged the `PyQSOFit.py` code into `fitmodule.py` to simplify the installation process. `qsofitmore` is now a standalone package with the same GNU license as `PyQSOFit`.

### Updates from v1.0.0 to v1.1.0:
- Added an optional broken power-law model in the continuum fitting process.  
- Enabled line property outputs for all narrow lines, OIII core+wing as a whole, and CIV br+na as a whole.  
- Used new criterion to verify narrow/broad components in self._PlotFig() to prevent narrow components from being plotted as red (broad) lines.  
- Changed prefix of comp_result from number to the complex name.  
- Bug fixes.
