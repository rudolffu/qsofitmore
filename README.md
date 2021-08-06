# QSOFITMORE
A wrapper of PyQSOFit for customization

### Features  
- Reading non-SDSS spectra without plateid, mjd and fiberid. 
- Narrow line measurements in the line MC process (which enables estimating uncertainties for narrow line parameters). 
- Dust maps and extinction laws other than SFD98 and CCM. 
- LaTeX rendering for line names in plots. 

### To do:  
- Parallel the MC processes.
- Intergrate narrow line measurements to the output fits file. 
- A simple tutorial of using `qsofitmore` with example(s).  


## 1. Installation
Currently, the fitting class of `qsofitmore` is a wrapper of that of `PyQSOFit`. Therefore `PyQSOFit` and its dependencies are required to run `qsofitmore`. 

Dependencies of `PyQSOFit`: astropy (included in anaconda), sfdmap, kapteyn, PyAstronomy. 

Dependencies of `qsofitmore`: uncertainties, dustmaps (optional, needed when using dust maps other than sfd, e.g. planck dust map). 

Assuming you have anaconda installed, the following steps demonstrate how to install dependencies above.

Install sfdmap: 
```
# Install sfdmap from conda-forge channel
conda install -c conda-forge sfdmap
# Install sfdmap with pip
pip install sfdmap
```

Install [kapteyn](https://www.astro.rug.nl/software/kapteyn/):  
```
# From kapteyn website
pip install https://www.astro.rug.nl/software/kapteyn/kapteyn-3.0.tar.gz
# From conda-forge (NOT tested)
conda install -c conda-forge kapteyn
```

Install [PyAstronomy](https://pyastronomy.readthedocs.io/en/latest/pyaCDoc/installingPyA.html) with pip: 
```
pip install PyAstronomy
```

```
conda install -c conda-forge uncertainties
```

After installing the dependencies, download and set up both `PyQSOFit` and `qsofitmore` packages.

```
mkdir ~/tools # or any other path you like
cd ~/tools 
git clone https://github.com/legolason/PyQSOFit.git 
git clone https://github.com/rudolffu/qsofitmore.git 
cp ~/tools/qsofitmore/qsofitmore/bin/pyqsofit-setup.py ~/tools/PyQSOFit/ 
cd ~/tools/PyQSOFit  
python pyqsofit-setup.py develop 
cd ~/tools/qsofitmore 
python setup.py develop  
```

## 2. Tutorial
 

