#!/usr/bin/env python
import sys
import os
import glob
import warnings
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import integrate
try:
    from kapteyn import kmpfit as _kmpfit
except Exception:
    _kmpfit = None
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy import constants as ac
from .extinction import *
from .auxmodule import *
try:
    from importlib.resources import files
except ImportError:
    # Fallback for Python < 3.9
    from importlib_resources import files
import pandas as pd
from astropy.modeling.models import BlackBody
from .config import migration_config

# Update resource paths
datapath = str(files('qsofitmore') / 'data')

__all__ = ['QSOFitNew']

getnonzeroarr = lambda x: x[x != 0]
sciplotstyle()

class QSOFitNew:
    def __init__(self, lam, flux, err, z, ra=0, dec=0, name=None, plateid=None, mjd=None, fiberid=None, 
                 path=None, and_mask=None, or_mask=None, is_sdss=False, in_rest_frame=False):
        """
        Get the input data prepared for the QSO spectral fitting
        
        Parameters:
        -----------
        lam: 1-D array with Npix
             Observed wavelength in unit of Angstrom
             
        flux: 1-D array with Npix
             Observed flux density in unit of 10^{-17} erg/s/cm^2/Angstrom
        
        err: 1-D array with Npix
             1 sigma err with the same unit of flux
             
        z: float number
            redshift
        
        ra, dec: float number, optional 
            the location of the source, right ascension and declination. The default number is 0
        name: str
            name of the object
        
        plateid, mjd, fiberid: integer number, optional
            If the source is SDSS object, they have the plate ID, MJD and Fiber ID in their file header.
            
        path: str
            the path of the input data
            
        and_mask, or_mask: 1-D array with Npix, optional
            the bad pixels defined from SDSS data, which can be got from SDSS datacube.
        """
        
        self.lam = np.asarray(lam, dtype=np.float64)
        self.flux = np.asarray(flux, dtype=np.float64)
        self.err = np.asarray(err, dtype=np.float64)
        self.sn_obs = self.flux / self.err
        self.z = z
        self.and_mask = and_mask
        self.or_mask = or_mask
        self.ra = ra
        self.dec = dec
        self.name = name
        self.plateid = plateid
        self.mjd = mjd  
        self.fiberid = fiberid
        self.path = path    
        self.is_sdss = is_sdss
        self.in_rest_frame = in_rest_frame
        # pivot for the power‐law continuum (default 3000 Å)
        self.pl_pivot = 3000.0


    @classmethod
    def fromsdss(cls, fname, redshift=None, path=None, plateid=None, mjd=None, fiberid=None, 
                 ra=None, dec=None, telescope=None, in_rest_frame=False):
        """
        Initialize QSOFit object from a SDSS spectrum fits file.
        Parameters:
        ----------
            fname : str
                name of the fits file.
            redshift : float
                redshift of the spectrum. Should be provided if not recorded in the fits header.
            path : str
                working directory.
        Returns:
        ----------
            cls : class
                A QSOFitNew object.
        Other parameters:
        ----------
            plateid, mjd, and fiberid: int
                Default None for non-SDSS spectra.
        Example:
        ----------
        q = QSOFitNew.fromsdss("custom_iraf_spectrum.fits", path=path)
        """
        hdu = fits.open(fname)
        hdr = hdu[0].header
        ra=hdr['plug_ra']          # RA 
        dec=hdr['plug_dec']        # DEC
        plateid = hdr['plateid']   # SDSS plate ID
        mjd = hdr['mjd']           # SDSS MJD
        fiberid = hdr['fiberid']   # SDSS fiber ID
        if redshift is None:
            try: 
                redshift = hdu[2].data['z'][0]
            except:
                print('Redshift not provided.')
                pass
        data = hdu[1].data
        hdu.close()
        wave = 10**data['loglam']
        flux = data['flux'] 
        ivar = pd.Series(data['ivar'])
        ivar.replace(0, np.nan, inplace=True)
        ivar_safe = ivar.interpolate()
        err = 1./np.sqrt(ivar_safe.values)   
        return cls(lam=wave, flux=flux, err=err, z=redshift, ra=ra, dec=dec, plateid=plateid, 
                   mjd=mjd, fiberid=fiberid, path=path, is_sdss=True, in_rest_frame=in_rest_frame)


    @classmethod
    def fromiraf(cls, fname, redshift=None, path=None, plateid=None, mjd=None, fiberid=None, 
                 ra=None, dec=None, telescope=None, in_rest_frame=False):
        """
        Initialize QSOFit object from a custom fits file
        generated by IRAF.
        Parameters:
        ----------
            fname : str
                name of the fits file.
            redshift : float
                redshift of the spectrum. Should be provided if not recorded in the fits header.
            path : str
                working directory.
        Returns:
        ----------
            cls : class
                A QSOFitNew object.
        Other parameters:
        ----------
            plateid, mjd, and fiberid: int
                Default None for non-SDSS spectra.
        Example:
        ----------
        q = QSOFitNew.fromiraf("custom_iraf_spectrum.fits", redshift=0.01, path=path)
        """
        hdu = fits.open(fname)
        header = hdu[0].header
        if redshift is None:
            try:
                redshift = float(header['redshift'])
            except:
                print("Redshift not provided, setting redshift to zero.")
                redshift = 0
        if ra is None or dec is None:
            try:
                ra = float(header['ra'])
                dec = float(header['dec'])
            except:
                coord = SkyCoord(header['RA']+header['DEC'], 
                                 frame='icrs',
                                 unit=(u.hourangle, u.deg))
                ra = coord.ra.value
                dec = coord.dec.value
        try:
            objname = header['object']
        except:
            objname = designation(ra, dec, telescope)
        if 'J' in objname:
            try:
                name = designation(ra, dec, telescope)
            except:
                name = objname
        else:
            name = objname
        if path is None:
            path = './'
        if mjd is None:
            try:
                mjd = float(header['mjd'])
            except:
                pass
        CRVAL1 = float(header['CRVAL1'])
        CD1_1 = float(header['CD1_1'])
        CRPIX1 = float(header['CRPIX1'])
        W1 = (1-CRPIX1) * CD1_1 + CRVAL1
        data = hdu[0].data
        dim = len(data.shape)
        if dim==1:
            num_pt = len(data)
            wave = np.linspace(W1, 
                               W1 + (num_pt - 1) * CD1_1, 
                               num=num_pt)
            wave = wave.flatten()
            flux = data.flatten()
            err = None
        elif dim==3:
            num_pt = data.shape[2]
            wave = np.linspace(W1, 
                               W1 + (num_pt - 1) * CD1_1, 
                               num=num_pt)
            flux = data[0,0,:]
            err = data[3,0,:]
        else:
            raise NotImplementedError("The IRAF spectrum has yet to be provided, not implemented.")
        hdu.close() 
        flux *= 1e17
        if err is not None:
            err *= 1e17
        return cls(lam=wave, flux=flux, err=err, z=redshift, ra=ra, dec=dec, name=name, plateid=plateid, 
                   mjd=mjd, fiberid=fiberid, path=path, is_sdss=False, in_rest_frame=in_rest_frame) 


    def setmapname(self, mapname):
        """
        Parameters:
            mapname : str
                name of the dust map. Currently only support
                'sfd' or 'planck'.
        """
        mapname = str(mapname).lower()
        self.mapname = mapname

    def _DeRedden(self, lam, flux, err, ra, dec, dustmap_path):
        """Correct the Galactic extinction"""
        try:
            print("The dust map is {}.".format(self.mapname))
        except AttributeError:
            print('`mapname` for extinction not set.\nSetting `mapname` to `sfd`.')
            mapname = 'sfd'
            self.mapname = mapname
        if self.mapname == 'sfd':
            self.ebv = getebv(self.ra, self.dec, mapname=self.mapname, map_dir=dustmap_path)
            zero_flux = np.where(flux == 0, True, False)
            flux[zero_flux] = 1e-10
            Alam = f99law(self.lam, self.ebv)
            flux_unred = deredden(Alam, self.flux)
            err_unred = err*flux_unred/flux
            flux_unred[zero_flux] = 0
            del self.flux, self.err
            self.flux = flux_unred
            self.err = err_unred
        elif self.mapname == 'planck':
            self.ebv = getebv(self.ra, self.dec, mapname=self.mapname)
            Alam = wang2019(self.lam, self.ebv)
            zero_flux = np.where(flux == 0, True, False)
            flux[zero_flux] = 1e-10
            flux_unred = deredden(Alam, self.flux) 
            err_unred = err*flux_unred/flux
            flux_unred[zero_flux] = 0
            del self.flux, self.err
            self.flux = flux_unred
            self.err = err_unred           
        return self.flux

    def set_pl_pivot(self, pivot):
        r"""
        Set the pivot wavelength (Å) for f_{\lambda} = (\lambda/\lambda_{\rm pivot})^{-\alpha}
        """
        self.pl_pivot = float(pivot)
    
    def _DoContiFit(self, wave, flux, err, ra, dec, plateid, mjd, fiberid):
        """Fit the continuum with PL, Polynomial, UV/optical FeII, Balmer continuum"""
        if self.plateid is None:
            plateid = 0
        if self.plateid is None:
            mjd = 0
        if self.plateid is None:
            fiberid = 0
        self.fe_uv = np.genfromtxt(os.path.join(datapath, 'iron_templates', 'fe_uv.txt'))
        self.fe_op = np.genfromtxt(os.path.join(datapath, 'iron_templates','fe_optical.txt'))
        self.fe_verner = np.genfromtxt(os.path.join(datapath, 'iron_templates','Fe_Verner_1micron.txt'))
        self.fe_nir = np.genfromtxt(os.path.join(datapath, 'iron_templates','Fe_G12_NIR.txt'))
        if self.BC == True:
            try:
                print("N_e = 1E{}.".format(self.ne))
            except AttributeError:
                print('N_e for Balmer line series not set.\nSetting N_e = 1E09. (q.set_log10_electron_density(9))')
                ne = 9
                self.ne = ne
            if self.ne == 9:
                balmer_file = os.path.join(datapath, 'balmer', 'balmer_n6_n50_em_NE09.csv')
            elif self.ne == 10:
                balmer_file = os.path.join(datapath, 'balmer', 'balmer_n6_n50_em_NE10.csv')
            self.df_balmer_series = pd.read_csv(balmer_file)
        else:
            balmer_file = os.path.join(datapath, 'balmer', 'balmer_n6_n50_em_NE09.csv')
            self.df_balmer_series = pd.read_csv(balmer_file)
        
        # do continuum fit--------------------------
        window_all = np.array(
            [[1150., 1170.], [1275., 1290.], [1350., 1360.], [1445., 1465.], [1690., 1705.], [1770., 1810.],
             [1970., 2400.], [2480., 2675.], [2925., 3400.], [3500., 3600.], [3600., 4260.],
            #  [3775., 3832.], [3833., 3860.], [3890., 3960.],
            #  [4000., 4090.], [4115., 4260.],
             [4435., 4640.], [5100., 5535.], [6005., 6035.], [6110., 6250.], [6800., 7000.], [7160., 7180.],
             [7500., 7800.], [8050., 8300.], [8580, 9000.], [9150, 9500.],
             [9650., 9900.], [10180., 10700.],
             [11050., 12480.]
             ])
        
        tmp_all = np.array([np.repeat(False, len(wave))]).flatten()
        for jj in range(len(window_all)):
            tmp = np.where((wave > window_all[jj, 0]) & (wave < window_all[jj, 1]), True, False)
            tmp_all = np.any([tmp_all, tmp], axis=0)
        
        if wave[tmp_all].shape[0] < 10:
            print('Continuum fitting pixel < 10.  ')

        SNR_SPEC = np.nanmedian(self.flux_prereduced/self.err_prereduced)

        # set initial paramiters for continuum
        if self.initial_guess is not None:
            pp0 = self.initial_guess
        else:
            pp0 = np.array([0., 3000., 0., 0., 3000., 0., 1., -1.5, 0., 5000, 0., 0., 0., 0.])
        if self.broken_pl == True:
            pp0 = np.array([0., 3000., 0., 0., 3000., 0., 1., -1.5, 0., 5000, 0., 0., 0., 0., -1.5])
        # Nudge initial guesses for optional components to small positive values
        if self.include_iron:
            pp0[0] = max(pp0[0], 1e-3)  # Fe (UV/MgII) norm
            pp0[3] = max(pp0[3], 1e-3)  # Fe (Balmer/optical) norm
        if self.BC:
            # Minimal positive seeds to avoid boundary lock
            pp0[8] = max(pp0[8], 1e-3)   # Balmer continuum norm seed
            pp0[10] = max(pp0[10], 1e-3)  # High-order series amplitude seed
        # build parameter bounds (static upper limits for speed/stability)
        bs_upper = 1e4
        tmp_parinfo = [{'limits': (0., 1000.)}, {'limits': (1200., 10000.)}, {'limits': (-0.01, 0.01)},
                       {'limits': (0., 1000.)}, {'limits': (1200., 10000.)}, {'limits': (-0.01, 0.01)},
                       {'limits': (0., 1000.)}, {'limits': (-5., 3.)}, 
                       {'limits': (0., 1)}, {'limits': (1200, 9000)}, {'limits': (0., bs_upper)}, 
                       None, None, None, ]
        if self.broken_pl == True:
            tmp_parinfo = [{'limits': (0., 1000.)}, {'limits': (1200., 10000.)}, {'limits': (-0.01, 0.01)},
                           {'limits': (0., 1000.)}, {'limits': (1200., 10000.)}, {'limits': (-0.01, 0.01)},
                           {'limits': (0., 1000.)}, {'limits': (-5., 3.)}, 
                           {'limits': (0., 1)}, {'limits': (1000, 9000)}, {'limits': (0., bs_upper)}, 
                           None, None, None, {'limits': (-5., 3.)}]

        # choose fitter by migration flag
        if getattr(migration_config, 'use_lmfit_continuum', False):
            conti_fit = self._fit_continuum_lmfit(wave[tmp_all], flux[tmp_all], err[tmp_all], pp0, tmp_parinfo)
        else:
            if _kmpfit is None:
                raise ImportError("kapteyn is required for kmpfit path. Install with: pip install 'cython<3.0' && pip install https://www.astro.rug.nl/software/kapteyn/kapteyn-3.4.tar.gz or enable lmfit via migration_config.use_lmfit_continuum=True")
            conti_fit = _kmpfit.Fitter(residuals=self._residuals, data=(wave[tmp_all], flux[tmp_all], err[tmp_all]))
            conti_fit.parinfo = tmp_parinfo
            conti_fit.fit(params0=pp0)
        
        # Perform one iteration to remove 3sigma pixel below the first continuum fit
        # to avoid the continuum windows falls within a BAL trough
        if self.rej_abs == True:
            if self.poly == True and self.broken_pl == False:
                tmp_conti = (conti_fit.params[6]*(wave[tmp_all]/self.pl_pivot)**conti_fit.params[7]+self.F_poly_conti(
                    wave[tmp_all], conti_fit.params[11:14]))
            elif self.poly ==False and self.broken_pl == False:
                tmp_conti = (conti_fit.params[6]*(wave[tmp_all]/self.pl_pivot)**conti_fit.params[7])
            elif self.poly == True and self.broken_pl == True:
                tmp_conti = broken_pl_model(wave[tmp_all],
                                            conti_fit.params[7],
                                            conti_fit.params[14],
                                            conti_fit.params[6])+self.F_poly_conti(wave[tmp_all], 
                                                                                   conti_fit.params[11:14])
            elif self.poly == False and self.broken_pl == True:
                tmp_conti = broken_pl_model(wave[tmp_all],
                                            conti_fit.params[7],
                                            conti_fit.params[14],
                                            conti_fit.params[6])
            ind_noBAL = ~np.where(((flux[tmp_all] < tmp_conti-3.*err[tmp_all]) & (wave[tmp_all] < 3500.)), True, False)
            if getattr(migration_config, 'use_lmfit_continuum', False):
                conti_fit = self._fit_continuum_lmfit(
                    wave[tmp_all][ind_noBAL], self.Smooth(flux[tmp_all][ind_noBAL], 10), err[tmp_all][ind_noBAL],
                    pp0, tmp_parinfo)
            else:
                _f = _kmpfit.Fitter(residuals=self._residuals, data=(
                    wave[tmp_all][ind_noBAL], self.Smooth(flux[tmp_all][ind_noBAL], 10), err[tmp_all][ind_noBAL]))
                conti_fit.parinfo = tmp_parinfo
                conti_fit.fit(params0=pp0)
        
        # calculate continuum luminoisty
        L = self._L_conti(wave, conti_fit.params)
        if self.Fe_flux_range is not None:
            end_pts = np.array(self.Fe_flux_range).flatten()
            n_fe_ranges = end_pts.shape[0]//2
            Fe_wave_keys = []
            Fe_range_list = []
            for i in range(n_fe_ranges):
                range_low = end_pts[2*i]
                range_high = end_pts[2*i+1]
                key = "w{}_w{}".format(int(range_low), int(range_high))
                Fe_wave_keys.append(key)
                Fe_range_list.append([range_low, range_high])
            self.Fe_wave_keys = Fe_wave_keys
            self.Fe_range_list = Fe_range_list

        # get conti result -----------------------------
        if self.MC == True and self.n_trails > 0:
            # Uncertainty estimation
            if getattr(migration_config, 'use_lmfit_continuum', False):
                # With MC=True, run lmfit-based MC to estimate errors
                conti_para_std, all_L_std = self._conti_mc_lmfit(self.wave[tmp_all], self.flux[tmp_all],
                                                                 self.err[tmp_all], pp0, tmp_parinfo,
                                                                 self.n_trails)
            else:
                conti_para_std, all_L_std = self._conti_mc(self.wave[tmp_all], self.flux[tmp_all],
                                                           self.err[tmp_all], pp0, conti_fit.parinfo,
                                                           self.n_trails)
            
            self.conti_result = np.array(
                [ra, dec, int(plateid), int(mjd), int(fiberid), self.z, SNR_SPEC, self.SN_ratio_conti, conti_fit.params[0],
                 conti_para_std[0], conti_fit.params[1], conti_para_std[1], conti_fit.params[2], conti_para_std[2],
                 conti_fit.params[3], conti_para_std[3], conti_fit.params[4], conti_para_std[4], conti_fit.params[5],
                 conti_para_std[5], conti_fit.params[6], conti_para_std[6], conti_fit.params[7], conti_para_std[7],
                 conti_fit.params[8], conti_para_std[8], conti_fit.params[9], conti_para_std[9], conti_fit.params[10],
                 conti_para_std[10], conti_fit.params[11], conti_para_std[11], conti_fit.params[12], conti_para_std[12],
                 conti_fit.params[13], conti_para_std[13], L[0], all_L_std[0], L[1], all_L_std[1], L[2], all_L_std[2]])
            self.conti_result_type = np.array(
                ['float', 'float', 'int', 'int', 'int', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float',
                 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float',
                 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float',
                 'float', 'float', 'float', 'float', 'float', 'float', 'float'])
            self.conti_result_name = np.array(
                ['ra', 'dec', 'plateid', 'MJD', 'fiberid', 'redshift', 'SNR_SPEC', 'SN_ratio_conti', 'Fe_uv_norm', 'Fe_uv_norm_err',
                 'Fe_uv_FWHM', 'Fe_uv_FWHM_err', 'Fe_uv_shift', 'Fe_uv_shift_err', 'Fe_op_norm', 'Fe_op_norm_err',
                 'Fe_op_FWHM', 'Fe_op_FWHM_err', 'Fe_op_shift', 'Fe_op_shift_err', 'PL_norm', 'PL_norm_err', 'PL_slope',
                 'PL_slope_err', 'BalmerC_norm', 'BalmerC_norm_err', 'BalmerS_FWHM', 'BalmerS_FWHM_err', 'BalmerS_norm',
                 'BalmerS_norm_err', 'POLY_a', 'POLY_a_err', 'POLY_b', 'POLY_b_err', 'POLY_c', 'POLY_c_err', 'LOGL1350',
                 'LOGL1350_err', 'LOGL3000', 'LOGL3000_err', 'LOGL5100', 'LOGL5100_err'])
            if self.broken_pl == True:
                self.conti_result = np.concatenate((self.conti_result, 
                                                    conti_fit.params[14], 
                                                    conti_para_std[14]), axis=None)
                self.conti_result_type = np.concatenate((self.conti_result_type, 
                                                         'float', 
                                                         'float'), axis=None)
                self.conti_result_name = np.concatenate((self.conti_result_name, 
                                                         'PL_slope2', 
                                                         'PL_slope2_err'), axis=None)                                         
        else:
            # Always provide parameter errors when MC is disabled; skip heavy propagation
            if getattr(migration_config, 'use_lmfit_continuum', False):
                try:
                    n_params = len(conti_fit.params)
                    param_stderr = []
                    for i in range(n_params):
                        par = conti_fit.result.params.get(f"p{i}")
                        se = getattr(par, 'stderr', None) if par is not None else None
                        if se is None or (isinstance(se, float) and (np.isnan(se))):
                            param_stderr.append(0.0)
                        else:
                            param_stderr.append(float(se))
                    conti_para_std = np.array(param_stderr)
                except Exception:
                    conti_para_std = np.zeros(len(conti_fit.params))
                # Skip analytic L uncertainty propagation for speed
                all_L_std = np.zeros(3, dtype=float)
            else:
                conti_para_std = np.zeros(len(conti_fit.params))
                all_L_std = np.zeros(3, dtype=float)

            # Build result arrays with error columns (same schema as MC branch)
            self.conti_result = np.array(
                [ra, dec, int(plateid), int(mjd), int(fiberid), self.z, SNR_SPEC, self.SN_ratio_conti, conti_fit.params[0],
                 conti_para_std[0], conti_fit.params[1], conti_para_std[1], conti_fit.params[2], conti_para_std[2],
                 conti_fit.params[3], conti_para_std[3], conti_fit.params[4], conti_para_std[4], conti_fit.params[5],
                 conti_para_std[5], conti_fit.params[6], conti_para_std[6], conti_fit.params[7], conti_para_std[7],
                 conti_fit.params[8], conti_para_std[8], conti_fit.params[9], conti_para_std[9], conti_fit.params[10],
                 conti_para_std[10], conti_fit.params[11], conti_para_std[11], conti_fit.params[12], conti_para_std[12],
                 conti_fit.params[13], conti_para_std[13], L[0], all_L_std[0], L[1], all_L_std[1], L[2], all_L_std[2]])
            self.conti_result_type = np.array(
                ['float', 'float', 'int', 'int', 'int', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float',
                 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float',
                 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float',
                 'float', 'float', 'float', 'float', 'float', 'float', 'float'])
            self.conti_result_name = np.array(
                ['ra', 'dec', 'plateid', 'MJD', 'fiberid', 'redshift', 'SNR_SPEC', 'SN_ratio_conti', 'Fe_uv_norm', 'Fe_uv_norm_err',
                 'Fe_uv_FWHM', 'Fe_uv_FWHM_err', 'Fe_uv_shift', 'Fe_uv_shift_err', 'Fe_op_norm', 'Fe_op_norm_err',
                 'Fe_op_FWHM', 'Fe_op_FWHM_err', 'Fe_op_shift', 'Fe_op_shift_err', 'PL_norm', 'PL_norm_err', 'PL_slope',
                 'PL_slope_err', 'BalmerC_norm', 'BalmerC_norm_err', 'BalmerS_FWHM', 'BalmerS_FWHM_err', 'BalmerS_norm',
                 'BalmerS_norm_err', 'POLY_a', 'POLY_a_err', 'POLY_b', 'POLY_b_err', 'POLY_c', 'POLY_c_err', 'LOGL1350',
                 'LOGL1350_err', 'LOGL3000', 'LOGL3000_err', 'LOGL5100', 'LOGL5100_err'])
            if self.broken_pl == True:
                # Ensure conti_para_std length covers index 14
                if len(conti_para_std) <= 14:
                    conti_para_std = np.pad(conti_para_std, (0, 15 - len(conti_para_std)), constant_values=0.0)
                self.conti_result = np.concatenate((self.conti_result,
                                                    conti_fit.params[14],
                                                    conti_para_std[14]), axis=None)
                self.conti_result_type = np.concatenate((self.conti_result_type,
                                                         'float',
                                                         'float'), axis=None)
                self.conti_result_name = np.concatenate((self.conti_result_name,
                                                         'PL_slope2',
                                                         'PL_slope2_err'), axis=None)
            # self.conti_result = np.append(self.conti_result, Fe_flux_result)
            # self.conti_result_type = np.append(self.conti_result_type, Fe_flux_type)
            # self.conti_result_name = np.append(self.conti_result_name, Fe_flux_name)
        # 
        self.conti_fit = conti_fit
        self.tmp_all = tmp_all
        if self.Fe_flux_range is not None:
            self.cal_Fe_line_res()
            Fe_line_result = np.array(list(self.Fe_line_result.values()))
            Fe_line_result_type = np.full_like(Fe_line_result,'float',dtype=object)
            Fe_line_result_name = np.asarray(list(self.Fe_line_result.keys()))
            self.conti_result = np.concatenate([self.conti_result, Fe_line_result])
            self.conti_result_type = np.concatenate([self.conti_result_type, Fe_line_result_type])
            self.conti_result_name = np.concatenate([self.conti_result_name, Fe_line_result_name])
        # save different models--------------------
        f_fe_mgii_model = self.Fe_flux_mgii(wave, conti_fit.params[0:3])
        f_fe_balmer_model = self.Fe_flux_balmer(wave, conti_fit.params[3:6])
        f_fe_verner_model = self.Fe_flux_verner(wave, conti_fit.params[3:6])
        f_fe_g12_model = self.Fe_flux_g12(wave, conti_fit.params[3:6])
        f_pl_model = conti_fit.params[6]*(wave/self.pl_pivot)**conti_fit.params[7]
        if self.broken_pl == True:
            f_pl_model = broken_pl_model(wave, conti_fit.params[7], conti_fit.params[14], conti_fit.params[6])
        f_bc_model = self.Balmer_conti(wave, conti_fit.params[8]) + self.Balmer_high_order(wave, conti_fit.params[9:11])
        f_poly_model = self.F_poly_conti(wave, conti_fit.params[11:14])
        # build conti_model using iron_temp_name
        if   self.iron_temp_name == "BG92-VW01":
            fe_sum = f_fe_mgii_model + f_fe_balmer_model
        elif self.iron_temp_name == "V09":
            fe_sum = f_fe_verner_model
        elif self.iron_temp_name == "G12":
            fe_sum = f_fe_g12_model
        else:
            raise RuntimeError(f"Unknown iron_temp_name='{self.iron_temp_name}'")
        f_conti_model = f_pl_model + fe_sum + f_poly_model + f_bc_model
        line_flux = flux-f_conti_model

        self.f_conti_model = f_conti_model
        self.f_bc_model = f_bc_model
        self.f_fe_model = fe_sum
        self.f_pl_model = f_pl_model
        self.f_poly_model = f_poly_model
        self.line_flux = line_flux
        self.PL_poly_BC = f_pl_model+f_poly_model+f_bc_model
        return self.conti_result, self.conti_result_name      

    
    def _f_conti_all(self, xval, pp):
        """
        Continuum components described by 14 parameters
         pp[0]: norm_factor for the MgII Fe_template
         pp[1]: FWHM for the MgII Fe_template
         pp[2]: small shift of wavelength for the MgII Fe template
         pp[3:5]: same as pp[0:2] but for the Hbeta/Halpha Fe template
         pp[6]: norm_factor for continuum f_lambda = (lambda/3000.0)^{-alpha}
         pp[7]: slope for the power-law continuum (blueward of 4661 A, if self.broken_pl == True)
         pp[8:10]: norm, Te and Tau_e for the Balmer continuum at <3646 A
         pp[11:13]: polynomial for the continuum
         pp[14]: optional, power-law index on the redward of 4661 A, if self.broken_pl == True
        """
        # iron flux for MgII line region
        f_Fe_MgII = self.Fe_flux_mgii(xval, pp[0:3])
        # iron flux for balmer line region
        f_Fe_Balmer = self.Fe_flux_balmer(xval, pp[3:6])
        f_Fe_verner = self.Fe_flux_verner(xval, pp[3:6])
        f_Fe_g12 = self.Fe_flux_g12(xval, pp[3:6])
        # power-law continuum
        f_pl = pp[6]*(xval/self.pl_pivot)**pp[7]
        # Balmer continuum
        f_conti_BC = self.Balmer_conti(xval, pp[8]) + self.Balmer_high_order(xval, pp[9:11])
        # polynormal conponent for reddened spectra
        f_poly = self.F_poly_conti(xval, pp[11:14])
        if self.broken_pl == True:
            f_pl = broken_pl_model(xval, pp[7], pp[14], pp[6])
        
        if self.include_iron:
            if   self.iron_temp_name == "BG92-VW01":
                f_Fe_all = f_Fe_MgII + f_Fe_Balmer
            elif self.iron_temp_name == "V09":
                f_Fe_all = f_Fe_verner
            elif self.iron_temp_name == "G12":
                f_Fe_all = f_Fe_g12
            else:
                raise RuntimeError(f"Unknown iron_temp_name='{self.iron_temp_name}'")

        if self.include_iron:
            if self.poly:
                yval = f_pl + f_Fe_all + f_poly + (f_conti_BC if self.BC else 0.0)
            else:
                yval = f_pl + f_Fe_all + (f_conti_BC if self.BC else 0.0)
        else:
            if self.poly:
                yval = f_pl + f_poly + (f_conti_BC if self.BC else 0.0)
            else:
                yval = f_pl + (f_conti_BC if self.BC else 0.0)
        return yval


    def _L_conti(self, wave, pp):
        r"""
        Calculate log10 \lambda L_{\lambda} at 1350, 3000, and 5100 A.
            L_{\lambda} is the monochromatic luminosity in erg/s/A, and
            \lambda L_{\lambda} is the a luminosity measure in erg/s.
        
        Parameters:
        ----------
        wave : array
            wavelength array
        pp : array
            parameters for the continuum model
        Returns:
        ----------
        loglamL : array
            log10(\lambda L_{\lambda}), where \lambda = 1350, 3000, 5100 A 
            \lambda L_{\lambda}(1350) is also known as L1350, etc.
        """
        conti_flux = pp[6]*(wave/self.pl_pivot)**pp[7]+self.F_poly_conti(wave, pp[11:14])
        if self.broken_pl:
            conti_flux = broken_pl_model(wave, pp[7], pp[14], pp[6]) + self.F_poly_conti(wave, pp[11:14])
        # plt.plot(wave,conti_flux)
        lamL = []
        for lam in [1350., 3000., 5100.]:
            if wave.min() < lam < wave.max():
                lam_flux = conti_flux[(wave > lam - 5) & (wave < lam + 5)].mean()
                lamL.append(lam * flux_to_luminosity(lam_flux, self.z))
            else:
                lamL.append(0.1)
        loglamL = np.log10(np.array(lamL))
        return loglamL
    

    def Fe_flux_mgii(self, xval, pp):
        "Fit the UV Fe compoent on the continuum from 1200 to 3500 A based on the Boroson & Green 1992."
        fe_uv = self.fe_uv
        yval = np.zeros_like(xval)
        wave_Fe_mgii = 10**fe_uv[:, 0]
        flux_Fe_mgii = fe_uv[:, 1]*10**15
        Fe_FWHM = pp[1]
        xval_new = xval*(1.0+pp[2])
        
        ind = np.where((xval_new > 1200.) & (xval_new < 3500.), True, False)
        if np.sum(ind) > 100:
            if Fe_FWHM < 900.0:
                sig_conv = np.sqrt(910.0**2-900.0**2)/2./np.sqrt(2.*np.log(2.))
            else:
                sig_conv = np.sqrt(Fe_FWHM**2-900.0**2)/2./np.sqrt(2.*np.log(2.))  # in km/s
            # Get sigma in pixel space
            sig_pix = sig_conv/106.3  # 106.3 km/s is the dispersion for the BG92 FeII template
            khalfsz = np.round(4*sig_pix+1, 0)
            xx = np.arange(0, khalfsz*2, 1)-khalfsz
            kernel = np.exp(-xx**2/(2*sig_pix**2))
            kernel = kernel/np.sum(kernel)
            
            flux_Fe_conv = np.convolve(flux_Fe_mgii, kernel, 'same')
            tck = interpolate.splrep(wave_Fe_mgii, flux_Fe_conv)
            yval[ind] = pp[0]*interpolate.splev(xval_new[ind], tck)
        return yval
    
    def Fe_flux_balmer(self, xval, pp):
        "Fit the optical FeII on the continuum from 3686 to 7484 A based on Vestergaard & Wilkes 2001"
        fe_op = self.fe_op
        yval = np.zeros_like(xval)
        
        wave_Fe_balmer = 10**fe_op[:, 0]
        flux_Fe_balmer = fe_op[:, 1]*10**15
        ind = np.where((wave_Fe_balmer > 3686.) & (wave_Fe_balmer < 7484.), True, False)
        wave_Fe_balmer = wave_Fe_balmer[ind]
        flux_Fe_balmer = flux_Fe_balmer[ind]
        Fe_FWHM = pp[1]
        xval_new = xval*(1.0+pp[2])
        ind = np.where((xval_new > 3686.) & (xval_new < 7484.), True, False)
        if np.sum(ind) > 100:
            if Fe_FWHM < 900.0:
                sig_conv = np.sqrt(910.0**2-900.0**2)/2./np.sqrt(2.*np.log(2.))
            else:
                sig_conv = np.sqrt(Fe_FWHM**2-900.0**2)/2./np.sqrt(2.*np.log(2.))  # in km/s
            # Get sigma in pixel space
            sig_pix = sig_conv/106.3  # 106.3 km/s is the dispersion for the BG92 FeII template
            khalfsz = np.round(4*sig_pix+1, 0)
            xx = np.arange(0, khalfsz*2, 1)-khalfsz
            kernel = np.exp(-xx**2/(2*sig_pix**2))
            kernel = kernel/np.sum(kernel)
            flux_Fe_conv = np.convolve(flux_Fe_balmer, kernel, 'same')
            tck = interpolate.splrep(wave_Fe_balmer, flux_Fe_conv)
            yval[ind] = pp[0]*interpolate.splev(xval_new[ind], tck)
        return yval


    def Fe_flux_verner(self, xval, pp):
        "Fit the FeII on the continuum from 2000 to 10000 A based on Verner et al. (2009)"
        fe_verner = self.fe_verner
        yval = np.zeros_like(xval)
        wave_Fe = fe_verner[:, 0]
        flux_Fe = fe_verner[:, 1]*8e-7
        
        # Restrict to template range
        ind = np.where((wave_Fe > 2000.) & (wave_Fe < 10000.), True, False)
        wave_Fe = wave_Fe[ind]
        flux_Fe = flux_Fe[ind]
        
        Fe_FWHM = pp[1]
        xval_new = xval*(1.0+pp[2])
        ind = np.where((xval_new > 2000.) & (xval_new < 10000.), True, False)
        
        if np.sum(ind) > 100:
            # Calculate actual dispersion from the template
            dwave = np.diff(wave_Fe)
            median_dwave = np.median(dwave)
            # Convert to velocity dispersion at middle wavelength
            median_wave = np.median(wave_Fe)
            template_dispersion = median_dwave / median_wave * 299792.458  # km/s
            
            # Verner template native resolution (typically ~900 km/s)
            template_native_fwhm = 900.0  # km/s
            
            if Fe_FWHM < template_native_fwhm:
                sig_conv = np.sqrt((template_native_fwhm + 10)**2 - template_native_fwhm**2) / (2. * np.sqrt(2.*np.log(2.)))
            else:
                sig_conv = np.sqrt(Fe_FWHM**2 - template_native_fwhm**2) / (2. * np.sqrt(2.*np.log(2.)))
            
            # For very high resolution templates, rebin first for efficiency
            if template_dispersion < 20:  # Very high resolution
                rebin_factor = max(1, int(20 / template_dispersion))
                if rebin_factor > 1:
                    # Rebin the template
                    n_new = len(wave_Fe) // rebin_factor
                    wave_Fe_rebin = np.zeros(n_new)
                    flux_Fe_rebin = np.zeros(n_new)
                    for i in range(n_new):
                        start_idx = i * rebin_factor
                        end_idx = min((i + 1) * rebin_factor, len(wave_Fe))
                        wave_Fe_rebin[i] = np.mean(wave_Fe[start_idx:end_idx])
                        flux_Fe_rebin[i] = np.mean(flux_Fe[start_idx:end_idx])
                    wave_Fe, flux_Fe = wave_Fe_rebin, flux_Fe_rebin
                    effective_dispersion = 20.0  # Updated dispersion after rebinning
                else:
                    effective_dispersion = template_dispersion
            else:
                effective_dispersion = template_dispersion
            
            # Calculate convolution kernel with actual dispersion
            sig_pix = sig_conv / effective_dispersion
            khalfsz = int(np.round(4*sig_pix + 1))
            xx = np.arange(0, khalfsz*2) - khalfsz
            kernel = np.exp(-xx**2 / (2.*sig_pix**2))
            kernel = kernel / np.sum(kernel)
            
            # Apply convolution
            flux_Fe_conv = np.convolve(flux_Fe, kernel, 'same')
            tck = interpolate.splrep(wave_Fe, flux_Fe_conv)
            yval[ind] = pp[0]*interpolate.splev(xval_new[ind], tck)
        return yval

    def Fe_flux_g12(self, xval, pp):
        "Fit the FeII on the continuum from 8100 to 11500 A based on Garcia-Rissmann+12."
        fe_nir = self.fe_nir
        yval = np.zeros_like(xval)

        # raw template
        wave_Fe = fe_nir[:, 0]
        flux_Fe = fe_nir[:, 1] * 100 # normalization to 1 

        # apply velocity shift
        Fe_FWHM = pp[1]
        xval_new = xval * (1.0 + pp[2])

        # select model range
        ind_wave = (wave_Fe >= 8100.) & (wave_Fe <= 11500.)
        wave_Fe = wave_Fe[ind_wave]
        flux_Fe = flux_Fe[ind_wave]

        ind = (xval_new >= 8100.) & (xval_new <= 11500.)
        if np.sum(ind) > 100:
            if Fe_FWHM < 750.0:
                sig_conv = np.sqrt(760.0**2 - 750.0**2) / (2. * np.sqrt(2.*np.log(2.)))
            else:
                sig_conv = np.sqrt(Fe_FWHM**2 - 750.0**2) / (2. * np.sqrt(2.*np.log(2.)))
            sig_pix = sig_conv / 100 
            khalfsz = int(np.round(4*sig_pix + 1))
            xx = np.arange(0, khalfsz*2) - khalfsz
            kernel = np.exp(-xx**2 / (2.*sig_pix**2))
            kernel /= kernel.sum()

            flux_conv = np.convolve(flux_Fe, kernel, mode='same')
            tck = interpolate.splrep(wave_Fe, flux_conv)
            yval[ind] = pp[0] * interpolate.splev(xval_new[ind], tck)

        return yval

    def Balmer_conti(self, xval, pp):
        """Fit the Balmer continuum from the model of Dietrich+02"""
        # xval = input wavelength, in units of A
        # pp=norm 
        xval=xval*u.AA
        lambda_BE = 3646.  # A
        Te = 1.5e4
        tau_BE = 1.
        bb_lam = BlackBody(Te*u.K, scale=1.0 * u.erg / (u.cm ** 2 * u.AA * u.s * u.sr))
        bbflux = bb_lam(xval).value*3.14   # in units of ergs/cm2/s/A
        tau = tau_BE * (xval.value/lambda_BE)**3
        y_BaC = pp * bbflux * (1 - np.exp(-tau))
        ind = ((xval.value < 2000) | (xval.value > lambda_BE))
        # ind = np.where(xval.value > lambda_BE, True, False)
        if ind.any() == True:
            y_BaC[ind] = 0
        return y_BaC


    def set_log10_electron_density(self, ne):
        """
        Parameters:
            ne : int / float
                log10 electron number density for Balmer line series. 
                Currently only support 9 or 10.
        """
        self.ne = ne


    def set_Balmer_fwhm(self, fwhm=None):
        """
        Parameters:
            fwhm : float
                FWHM of Balmer line series in km/s.
        """
        self.Balmer_fwhm = fwhm


    def Balmer_high_order(self, xval, pp):
        xx = xval
        df = self.df_balmer_series
        y = np.zeros_like(xx)
        wave_bs = df.wave.values
        flux_bs = df.flux_norm.values
        try:
            fwhm = self.Balmer_fwhm
            self.conti_fit.params[9] = fwhm
        except AttributeError:
            fwhm = pp[0]
        for i in range(len(wave_bs)):
            s = (fwhm / 3e5) * wave_bs[i] / 2 / np.sqrt(2*np.log(2))
            exp = ((xx-wave_bs[i]) / s)**2. / 2.
            y += flux_bs[i] * np.exp( -exp ) / s
        ind = ((xx < 3760.) & (xx > 3646.))
        ind1 = (xx < 3646.)
        ind2 = (xx >= 3760.)
        if ((len(y[ind]) != 0) & (len(y[ind2]) != 0)):
            ind_ref = ((xx > 3760.) & (xx < 3850.))
            f_interp = interpolate.interp1d(xx[ind_ref], y[ind_ref], fill_value='extrapolate')
            yi = f_interp(xx[ind])
            y[ind] = yi
            y[ind1] = 0.
        # Ensure non-negative scaling and cap if a cap is passed via pp
        amp = max(0.0, float(pp[1]))
        y_scaled = y * amp
        return y_scaled

    # ---------------- lmfit migration stubs -----------------
    def _fit_continuum_lmfit(self, wave, flux, err, pp0, param_bounds):
        """
        lmfit-based continuum fitting using existing residuals and parameter order.
        Builds Parameters named p0..pN mapped to pp0, applies bounds from param_bounds,
        and minimizes weighted residuals with Levenberg–Marquardt ('leastsq').
        Returns an object exposing .params (list), .result (lmfit result), and .success.
        """
        try:
            from lmfit import minimize, Parameters
        except Exception:
            raise ImportError("lmfit is required for lmfit continuum path but is not available.")

        # Build lmfit Parameters
        params = Parameters()
        n_params = len(pp0)
        for i in range(n_params):
            name = f"p{i}"
            init_val = float(pp0[i])
            params.add(name, value=init_val)
            # Apply bounds if provided
            if param_bounds is not None and i < len(param_bounds):
                b = param_bounds[i]
                if isinstance(b, dict) and b.get('limits') is not None:
                    lo, hi = b['limits']
                    if lo is not None:
                        params[name].set(min=float(lo))
                    if hi is not None:
                        params[name].set(max=float(hi))
        # No additional coupled constraints for speed

        # Define residual function mapping Parameters -> ordered array
        def residual_func(pars, x, y, w):
            pp = np.array([pars[f"p{j}"].value for j in range(n_params)], dtype=float)
            model = self._f_conti_all(x, pp)
            return (y - model) / w

        # Optimize
        result = minimize(residual_func, params, args=(wave, flux, err), method='leastsq')

        # Build a lightweight result compatible with downstream usage
        class FitResult:
            pass

        fitres = FitResult()
        fitres.params = [result.params[f"p{i}"].value for i in range(n_params)]
        fitres.result = result
        fitres.success = bool(getattr(result, 'success', True))
        # Populate attributes used by kmpfit in some branches to avoid attribute errors
        fitres.parinfo = param_bounds
        return fitres

    def _conti_mc_lmfit(self, x, y, err, pp0, pp_limits, n_trails):
        """
        lmfit-based Monte Carlo error estimation for continuum.
        Returns (param_std, L_std) where L_std is for [1350, 3000, 5100].
        """
        try:
            from lmfit import minimize, Parameters
        except Exception:
            raise ImportError("lmfit is required for lmfit MC path but is not available.")

        # Prepare parameter template from pp0 and bounds
        n_params = len(pp0)
        def make_params(init):
            params = Parameters()
            for i in range(n_params):
                name = f"p{i}"
                params.add(name, value=float(init[i]))
                b = pp_limits[i] if i < len(pp_limits) else None
                if isinstance(b, dict) and b.get('limits') is not None:
                    lo, hi = b['limits']
                    params[name].set(min=float(lo), max=float(hi))
            return params

        def residual_func(pars, x, y, w):
            pp = np.array([pars[f"p{j}"].value for j in range(n_params)], dtype=float)
            return (y - self._f_conti_all(x, pp)) / w

        samples = []
        Lsamples = []
        # Prepare Fe flux/ew collection if requested
        collect_fe = self.Fe_flux_range is not None and hasattr(self, 'Fe_range_list') and hasattr(self, 'Fe_wave_keys')
        if collect_fe:
            n_Fe_flux = np.array(self.Fe_flux_range).flatten().shape[0] // 2
            all_Fe_flux = np.zeros((n_Fe_flux, int(max(1, n_trails))))
            all_Fe_ew = np.zeros_like(all_Fe_flux)
        base_init = np.array(pp0, dtype=float)
        for ti in range(int(max(1, n_trails))):
            flux_noisy = y + np.random.randn(len(y)) * err
            params = make_params(base_init)
            try:
                res = minimize(residual_func, params, args=(x, flux_noisy, err), method='leastsq')
                if getattr(res, 'success', True):
                    pbest = np.array([res.params[f"p{i}"].value for i in range(n_params)], dtype=float)
                    samples.append(pbest)
                    Lsamples.append(self._L_conti(x, pbest))
                    if collect_fe:
                        for i, wrange in enumerate(self.Fe_range_list):
                            flux_tmp, ew_tmp = self.Fe_line_prop(pbest, wrange)
                            all_Fe_flux[i, ti] = flux_tmp
                            all_Fe_ew[i, ti] = ew_tmp
            except Exception:
                continue

        if len(samples) == 0:
            return np.zeros(n_params, dtype=float), np.zeros(3, dtype=float)
        samples = np.array(samples)
        Lsamples = np.array(Lsamples)
        if collect_fe:
            # Store per-range distributions for later error calculation
            try:
                df_fe_flux = pd.DataFrame(all_Fe_flux.T, columns=self.Fe_wave_keys)
                df_fe_ew = pd.DataFrame(all_Fe_ew.T, columns=self.Fe_wave_keys)
                self.df_fe_flux = df_fe_flux
                self.df_fe_ew = df_fe_ew
            except Exception:
                pass
        return samples.std(axis=0), Lsamples.std(axis=0)

    def _do_line_lmfit(self, linelist, line_flux, ind_line, ind_n, nline_fit, ngauss_fit):
        """
        lmfit-based line fitting using Manygauss on log-wavelength.
        Mirrors kmpfit parameter order: [amp, log(center), sigma] for each Gaussian.
        Applies tying logic inside residual (parity with kmpfit path).
        Returns a wrapper with params (best-fit vector with ties applied), chisqr, dof, etc.
        """
        try:
            from lmfit import minimize, Parameters
        except Exception:
            raise ImportError("lmfit is required for lmfit line path but is not available.")

        # Build initial parameter vector and bounds as in kmpfit
        line_fit_ini = np.array([])
        line_fit_par = np.array([])
        for n in range(nline_fit):
            for nn in range(int(ngauss_fit[n])):
                # initial values
                line_fit_ini0 = [
                    0.0,
                    float(np.log(linelist['lambda'][ind_line][n])),
                    float(linelist['inisig'][ind_line][n]),
                ]
                line_fit_ini = np.concatenate([line_fit_ini, line_fit_ini0])
                # bounds
                lambda_center = float(np.log(linelist['lambda'][ind_line][n]))
                v_off = float(linelist['voff'][ind_line][n])
                sig_low = float(linelist['minsig'][ind_line][n])
                sig_up = float(linelist['maxsig'][ind_line][n])
                line_fit_par0 = [
                    {'limits': (0.0, 1e10)},
                    {'limits': (lambda_center - v_off, lambda_center + v_off)},
                    {'limits': (sig_low, sig_up)},
                ]
                line_fit_par = np.concatenate([line_fit_par, line_fit_par0])

        # Save for downstream code
        self.line_fit_ini = line_fit_ini
        self.line_fit_par = line_fit_par

        # Build lmfit Parameters p0..pN
        params = Parameters()
        n_params = len(line_fit_ini)
        for i in range(n_params):
            name = f"p{i}"
            params.add(name, value=float(line_fit_ini[i]))
            b = line_fit_par[i] if i < len(line_fit_par) else None
            if isinstance(b, dict) and b.get('limits') is not None:
                lo, hi = b['limits']
                params[name].set(min=float(lo), max=float(hi))

        x = np.log(self.wave[ind_n])
        y = line_flux[ind_n]
        w = self.err[ind_n]

        def residual_func(pars, x, y, w):
            # Reconstruct parameter vector and apply ties
            pp = np.array([pars[f"p{j}"].value for j in range(n_params)], dtype=float)
            # Tie logic (same as _residuals_line)
            if self.tie_lambda:
                if len(self.ind_tie_vindex1) > 1:
                    for xx in range(len(self.ind_tie_vindex1) - 1):
                        pp[int(self.ind_tie_vindex1[xx + 1])] = pp[int(self.ind_tie_vindex1[0])] + self.delta_lambda1[xx]
                if len(self.ind_tie_vindex2) > 1:
                    for xx in range(len(self.ind_tie_vindex2) - 1):
                        pp[int(self.ind_tie_vindex2[xx + 1])] = pp[int(self.ind_tie_vindex2[0])] + self.delta_lambda2[xx]
            if self.tie_width:
                if len(self.ind_tie_windex1) > 1:
                    for xx in range(len(self.ind_tie_windex1) - 1):
                        pp[int(self.ind_tie_windex1[xx + 1])] = pp[int(self.ind_tie_windex1[0])]
                if len(self.ind_tie_windex2) > 1:
                    for xx in range(len(self.ind_tie_windex2) - 1):
                        pp[int(self.ind_tie_windex2[xx + 1])] = pp[int(self.ind_tie_windex2[0])]
            if len(self.ind_tie_findex1) > 0 and self.tie_flux_1:
                pp[int(self.ind_tie_findex1[1])] = pp[int(self.ind_tie_findex1[0])] * self.fvalue_factor_1
            if len(self.ind_tie_findex2) > 0 and self.tie_flux_2:
                pp[int(self.ind_tie_findex2[1])] = pp[int(self.ind_tie_findex2[0])] * self.fvalue_factor_2
            self.newpp = pp.copy()
            return (y - self.Manygauss(x, pp)) / w

        result = minimize(residual_func, params, args=(x, y, w), method='leastsq')

        # Build wrapper matching kmpfit attributes used by downstream code
        class FitResult:
            pass

        fitres = FitResult()
        # Final best-fit vector with ties applied
        pp_best = np.array([result.params[f"p{i}"].value for i in range(n_params)], dtype=float)
        # Apply tie logic once more to ensure vector consistency
        if self.tie_lambda:
            if len(self.ind_tie_vindex1) > 1:
                for xx in range(len(self.ind_tie_vindex1) - 1):
                    pp_best[int(self.ind_tie_vindex1[xx + 1])] = pp_best[int(self.ind_tie_vindex1[0])] + self.delta_lambda1[xx]
            if len(self.ind_tie_vindex2) > 1:
                for xx in range(len(self.ind_tie_vindex2) - 1):
                    pp_best[int(self.ind_tie_vindex2[xx + 1])] = pp_best[int(self.ind_tie_vindex2[0])] + self.delta_lambda2[xx]
        if self.tie_width:
            if len(self.ind_tie_windex1) > 1:
                for xx in range(len(self.ind_tie_windex1) - 1):
                    pp_best[int(self.ind_tie_windex1[xx + 1])] = pp_best[int(self.ind_tie_windex1[0])]
            if len(self.ind_tie_windex2) > 1:
                for xx in range(len(self.ind_tie_windex2) - 1):
                    pp_best[int(self.ind_tie_windex2[xx + 1])] = pp_best[int(self.ind_tie_windex2[0])]
        if len(self.ind_tie_findex1) > 0 and self.tie_flux_1:
            pp_best[int(self.ind_tie_findex1[1])] = pp_best[int(self.ind_tie_findex1[0])] * self.fvalue_factor_1
        if len(self.ind_tie_findex2) > 0 and self.tie_flux_2:
            pp_best[int(self.ind_tie_findex2[1])] = pp_best[int(self.ind_tie_findex2[0])] * self.fvalue_factor_2

        fitres.params = pp_best
        fitres.result = result
        fitres.success = bool(getattr(result, 'success', True))
        fitres.status = 1 if fitres.success else 0
        # kmpfit-like attributes
        fitres.chi2_min = float(getattr(result, 'chisqr', np.sum(result.residual**2)))
        ndata = getattr(result, 'ndata', len(x))
        nvarys = getattr(result, 'nvarys', len([p for p in result.params.values() if not p.vary is False]))
        fitres.dof = int(ndata - nvarys)
        fitres.niter = int(getattr(result, 'nfev', 0))
        return fitres


    def Fit(self, name=None, nsmooth=1, and_or_mask=True, reject_badpix=True, deredden=True, wave_range=None,
            wave_mask=None, decomposition_host=True, BC03=False, Mi=None, npca_gal=5, npca_qso=20,
            broken_pl=False, include_iron=None, Fe_uv_op=True, Fe_verner09=False,
            iron_temp_name=None,
            Fe_flux_range=None, poly=False, BC=False, rej_abs=False, initial_guess=None, MC=True, n_trails=1,
            linefit=True, tie_lambda=True, tie_width=True, tie_flux_1=True, tie_flux_2=True, save_result=True,
            plot_fig=True, save_fig=True, plot_line_name=True, plot_legend=True, dustmap_path=None, 
            save_fig_path=None, save_fits_path=None, save_fits_name=None, mask_compname=None):
        self.mask_compname = mask_compname

        # deprecate Fe_uv_op in favor of include_iron
        if include_iron is None:
            include_iron = Fe_uv_op
            warnings.warn("'Fe_uv_op' is deprecated; please use 'include_iron' instead.",
                          DeprecationWarning, stacklevel=2)
        else:
            if Fe_uv_op != include_iron:
                warnings.warn("'Fe_uv_op' is deprecated and ignored when 'include_iron' is set.",
                              DeprecationWarning, stacklevel=2)

        # deprecate Fe_verner09 in favor of iron_temp_name
        if iron_temp_name is None:
            if Fe_verner09:
                iron_temp_name = "V09"
                warnings.warn("'Fe_verner09' is deprecated; please use 'iron_temp_name' instead.",
                              DeprecationWarning, stacklevel=2)
            else:
                iron_temp_name = "BG92-VW01"
        else:
            if Fe_verner09:
                warnings.warn("'Fe_verner09' is deprecated and ignored when 'iron_temp_name' is set.",
                              DeprecationWarning, stacklevel=2)

        self.broken_pl = broken_pl
        if name is None:
            name = self.name
        self.name = name
        self.wave_range = wave_range
        self.wave_mask = wave_mask
        self.BC03 = BC03
        self.Mi = Mi
        self.npca_gal = npca_gal
        self.npca_qso = npca_qso
        self.initial_guess = initial_guess
        self.include_iron = include_iron
        self.Fe_uv_op = Fe_uv_op           # for backward‐compat
        self.Fe_verner09 = Fe_verner09     # for backward‐compat
        self.iron_temp_name = iron_temp_name
        self.Fe_flux_range = Fe_flux_range
        self.poly = poly
        self.BC = BC
        self.rej_abs = rej_abs
        self.MC = MC
        self.n_trails = n_trails
        self.tie_lambda = tie_lambda
        self.tie_width = tie_width
        self.tie_flux_1 = tie_flux_1
        self.tie_flux_2 = tie_flux_2
        self.plot_line_name = plot_line_name
        self.plot_legend = plot_legend
        self.save_fig = save_fig
        if name is None and save_fits_name is not None:
            name = save_fits_name
            print("Name is now {}.".format(name))
        elif name is None and save_fits_name is None:
            name = self.name
            print("Name is now {}.".format(name))
        else:
            pass
        if self.is_sdss == False and name is None:
            print("Bad figure name!")
        if self.z > 2.5:
            poly = False
        if Fe_verner09 == True:
            Fe_uv_op=True
        
        # get the source name in plate-mjd-fiber, if no then None
        if self.name is None and (self.plateid is not None or self.mjd is not None or self.fiberid is not None):
            self.sdss_name = str(self.plateid).zfill(4)+'-'+str(self.mjd)+'-'+str(self.fiberid).zfill(4)
            self.name = self.sdss_name
        
        # set default path for figure and fits
        if save_result == True and save_fits_path == None:
            save_fits_path = self.path
        if save_fig == True and save_fig_path == None:
            save_fig_path = self.path
        if save_fits_name == None:
            save_fits_name = f'res_qsofitmore_{self.name}.fits'        
        
        # deal with pixels with error equal 0 or inifity
        ind_gooderror = np.where((self.err != 0) & ~np.isinf(self.err), True, False)
        err_good = self.err[ind_gooderror]
        flux_good = self.flux[ind_gooderror]
        lam_good = self.lam[ind_gooderror]
        
        if (self.and_mask is not None) & (self.or_mask is not None):
            and_mask_good = self.and_mask[ind_gooderror]
            or_mask_good = self.or_mask[ind_gooderror]
            del self.and_mask, self.or_mask
            self.and_mask = and_mask_good
            self.or_mask = or_mask_good
        del self.err, self.flux, self.lam
        self.err = err_good
        self.flux = flux_good
        self.lam = lam_good
        
        if nsmooth is not None:
            self.flux = self.Smooth(self.flux, nsmooth)
            self.err = self.Smooth(self.err, nsmooth)
        if (and_or_mask == True) and (self.and_mask is not None or self.or_mask is not None):
            self._MaskSdssAndOr(self.lam, self.flux, self.err, self.and_mask, self.or_mask)
        if reject_badpix == True:
            self._RejectBadPix(self.lam, self.flux, self.err)
        if wave_range is not None:
            self._WaveTrim(self.lam, self.flux, self.err, self.z)
        if wave_mask is not None:
            self._WaveMsk(self.lam, self.flux, self.err, self.z)
        if self.in_rest_frame == False:
            self._RestFrame(self.lam, self.flux, self.err, self.z)
        elif self.in_rest_frame == True:
            if deredden == True:
                deredden = False
                print('The input spectrum is in the rest frame, so deredden is set to False!')
            self._RestFrame(self.lam, self.flux, self.err, 0.0)
        if deredden == True and self.ra != -999. and self.dec != -999.:
            self._DeRedden(self.lam, self.flux, self.err, self.ra, self.dec, dustmap_path)

        self._CalculateSN(self.wave, self.flux)
        self._OrignialSpec(self.wave, self.flux, self.err)
        
        # do host decomposition --------------
        if self.z < 1.16 and decomposition_host == True:
            self._DoDecomposition(self.wave, self.flux, self.err, self.path)
        else:
            self.decomposed = False
            if self.z > 1.16 and decomposition_host == True:
                print('redshift larger than 1.16 is not allowed for host '
                      'decomposion!')
        
        # fit continuum --------------------
        self._DoContiFit(self.wave, self.flux, self.err, self.ra, self.dec, self.plateid, self.mjd, self.fiberid)
        # fit line
        if linefit == True:
            
            self._DoLineFit(self.wave, self.line_flux, self.err, self.conti_fit)
        else:
            self.ncomp = 0
        # save data -------
        if save_result == True:
            if linefit == False:
                self.line_result = np.array([])
                self.line_result_type = np.array([])
                self.line_result_name = np.array([])
            self._SaveResult(self.conti_result, self.conti_result_type, self.conti_result_name, self.line_result,
                             self.line_result_type, self.line_result_name, save_fits_path, save_fits_name)
        
        # plot fig and save ------
        if plot_fig == True:
            if linefit == False:
                self.gauss_result = np.array([])
                self.all_comp_range = np.array([])
                self.uniq_linecomp_sort = np.array([])
            self._PlotFig(self.ra, self.dec, self.z, self.wave, self.flux, self.err, decomposition_host, linefit,
                          self.tmp_all, self.gauss_result, self.f_conti_model, self.conti_fit, self.all_comp_range,
                          self.uniq_linecomp_sort, self.line_flux, save_fig_path)


    def _SaveResult(self, conti_result, conti_result_type, conti_result_name, line_result, line_result_type,
                    line_result_name, save_fits_path, save_fits_name):
        """Save all data to fits"""
        self.all_result = np.concatenate([conti_result, line_result])
        self.all_result_type = np.concatenate([conti_result_type, line_result_type])
        self.all_result_name = np.concatenate([conti_result_name, line_result_name])
        all_result = self.all_result.astype(float)
        t = Table(all_result, names=(self.all_result_name), dtype=self.all_result_type)
        if 'Ha_whole_br_area' in self.all_result_name:
            t['LOGLHA'] = np.log10(flux_to_luminosity(t['Ha_whole_br_area'], self.z))
        if 'Ha_whole_br_area_err' in self.all_result_name:
            t['LOGLHA_ERR'] = np.abs(t['Ha_whole_br_area_err'] / (t['Ha_whole_br_area'] * np.log(10)))
        self.result_table = t
        t.write(save_fits_path+save_fits_name, format='fits', overwrite=True)


    def _PlotFig(self, ra, dec, z, wave, flux, err, decomposition_host, linefit, tmp_all, gauss_result, f_conti_model,
                 conti_fit, all_comp_range, uniq_linecomp_sort, line_flux, save_fig_path):
        """Plot the results"""
        if self.broken_pl == True:
            self.PL_poly = broken_pl_model(wave, 
                                           conti_fit.params[7],
                                           conti_fit.params[14],
                                           conti_fit.params[6]) + self.F_poly_conti(wave, conti_fit.params[11:14])
        else:
            self.PL_poly = conti_fit.params[6]*(wave/self.pl_pivot)**conti_fit.params[7]+self.F_poly_conti(wave, 
                                                                                                    conti_fit.params[11:14])
        
        matplotlib.rc('xtick', labelsize=20)
        matplotlib.rc('ytick', labelsize=20)
        
        if linefit == True:
            nclos = np.max([self.ncomp, 1])
            fig, axn = plt.subplots(nrows=2, ncols=nclos, figsize=(15, 8),
                                    squeeze=False)  # prepare for the emission line subplots in the second row
            gs = axn[0, 0].get_gridspec()
            for axi in axn[0, :]:
                axi.remove()
            ax = fig.add_subplot(gs[0, :])
            if self.MC == True:
                mc_flag = 2
            else:
                mc_flag = 1
            
            lines_total = np.zeros_like(wave)
            na_lines_total = np.zeros_like(wave)
            line_order = {'r': 3, 'g': 7}  # to make the narrow line plot above the broad line
            
            temp_gauss_result = gauss_result
            for p in range(int(len(temp_gauss_result)/mc_flag/3)):
                # warn that the width used to separate narrow from broad is not exact 1200 km s-1 which would lead to wrong judgement
                # if self.CalFWHM(temp_gauss_result[(2+p*3)*mc_flag]) < 1200.:
                line_single = self.Onegauss(np.log(wave), temp_gauss_result[p*3*mc_flag:(p+1)*3*mc_flag:mc_flag])
                if temp_gauss_result[(2+p*3)*mc_flag] - 0.0017 <= 1e-10:    
                    color = 'g'
                    na_lines_total += line_single
                else:
                    color = 'r'
                ax.plot(wave, line_single+f_conti_model, color=color, zorder=5)
                for c in range(self.ncomp):
                    axn[1][c].plot(wave, line_single, color=color, zorder=line_order[color])
                lines_total += line_single
            
            ax.plot(wave, lines_total+f_conti_model, 'b', label='line',
                    zorder=6)  # supplement the emission lines in the first subplot
            self.lines_total = lines_total
            self.na_lines_total = na_lines_total
            for c, linecompname in enumerate(uniq_linecomp_sort):
                tname = texlinename(linecompname)
                axn[1][c].plot(wave, lines_total, color='b', zorder=10)
                axn[1][c].plot(wave, self.line_flux, 'k', zorder=0)
                
                axn[1][c].set_xlim(all_comp_range[2*c:2*c+2])
                line_flux_c = line_flux[
                    np.where((wave > all_comp_range[2*c]) & (wave < all_comp_range[2*c+1]), True, False)]
                f_max = line_flux_c.max()
                f_min = line_flux_c.min()
                axn[1][c].set_ylim(-0.1*f_max, f_max*1.1)
                axn[1][c].set_xticks([all_comp_range[2*c], np.round((all_comp_range[2*c]+all_comp_range[2*c+1])/2, -1),
                                      all_comp_range[2*c+1]])
                axn[1][c].text(0.02, 0.9, tname, fontsize=20, transform=axn[1][c].transAxes)
                rchi2 = float(self.comp_result[np.where(self.comp_result_name==linecompname+'_line_red_chi2')])
                axn[1][c].text(0.02, 0.80, r'$\chi ^2_r=$'+str(np.round(float(rchi2), 2)),
                               fontsize=16, transform=axn[1][c].transAxes)
        else:
            fig, ax = plt.subplots(nrows=1, ncols=1,
                                   figsize=(15, 8))  # if no lines are fitted, there would be only one row
        
        if self.ra == -999. or self.dec == -999.:
            ax.set_title(str(self.name)+'   z = '+str(np.round(z, 4)), fontsize=20)
        else:
            # ax.set_title('ra,dec = ('+str(ra)+','+str(dec)+')   '+str(self.name)+'   $z$ = '+str(np.round(z, 4)),
                        #  fontsize=20)
            ax.set_title('ra, dec = ({:.6f}, {:.6f})   {}    $z$ = {:.4f}'.format(ra, dec, self.name.replace('_',' '), z),
                         fontsize=20)
        
        ax.plot(self.wave_prereduced, self.flux_prereduced, 'k', label='data', zorder=2)
        
        if decomposition_host == True and self.decomposed == True:
            ax.plot(wave, self.qso+self.host, 'pink', label='host+qso temp', zorder=3)
            ax.plot(wave, flux, 'grey', label='data-host', zorder=1)
            ax.plot(wave, self.host, 'purple', label='host', zorder=4)
        else:
            host = self.flux_prereduced.min()
        
        ax.scatter(wave[tmp_all], np.repeat(self.flux_prereduced.max()*1.05, len(wave[tmp_all])), color='grey',
                   marker='o')  # plot continuum region
        
        ax.plot([0, 0], [0, 0], 'r', label='line br', zorder=5)
        ax.plot([0, 0], [0, 0], 'g', label='line na', zorder=5)
        ax.plot(wave, f_conti_model, 'c', lw=2, label='FeII', zorder=7)
        if self.BC == True:
            ax.plot(wave, self.f_pl_model+self.f_poly_model+self.f_bc_model, 'y', lw=2, label='BC', zorder=8)
        ax.plot(wave, self.f_pl_model+self.f_poly_model, color='orange', lw=2, label='conti', zorder=9)
        if self.decomposed == False:
            plot_bottom = flux.min()
        else:
            plot_bottom = min(self.host.min(), flux.min())
        
        ax.set_ylim(plot_bottom*0.9, self.flux_prereduced.max()*1.1)
        
        if self.plot_legend == True:
            ax.legend(loc='best', frameon=False, ncol=2)
        
        # plot line name--------
        if self.plot_line_name == True:
            line_cen = np.array(
                [6564.60, 6549.85, 6585.27, 6718.29, 6732.66, 4862.68, 5008.24, 4687.02, 4341.68, 3934.78, 3728.47,
                 3426.84, 2798.75, 1908.72, 1816.97, 1750.26, 1718.55, 1549.06, 1640.42, 1402.06, 1396.76, 1335.30, \
                 1215.67])
            
            line_name = np.array(
                ['', '', r'H$\alpha$+[NII]', '', '[SII]6718,6732', r'H$\beta$', '[OIII]', 'HeII4687', r'H$\gamma$', 
                 'CaII3934', '[OII]3728',
                 'NeV3426', 'MgII', 'CIII]', 'SiII1816', 'NIII]1750', 'NIV]1718', 'CIV', 'HeII1640', '', 'SiIV+OIV',
                 'CII1335', r'Ly$\alpha$'])
            
            for ll in range(len(line_cen)):
                if wave.min() < line_cen[ll] < wave.max():
                    ax.plot([line_cen[ll], line_cen[ll]], [plot_bottom*0.9, self.flux_prereduced.max()*1.1], 'k:')
                    ax.text(line_cen[ll]+7, 1.08*self.flux_prereduced.max(), line_name[ll], rotation=90, fontsize=16,
                            va='top')
        
        ax.set_xlim(wave.min(), wave.max())
        ax.minorticks_on()
        
        if linefit == True:
            ax.text(0.5, -1.45, r'Rest-frame wavelength ($\rm \AA$)', fontsize=22, transform=ax.transAxes,
                    ha='center')
            ax.text(-0.07, -0.1, r'$F_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)', fontsize=22,
                    transform=ax.transAxes, rotation=90, ha='center', rotation_mode='anchor')
        else:
            plt.xlabel(r'Rest-frame wavelength ($\rm \AA$)', fontsize=22)
            plt.ylabel(r'$F_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)', fontsize=22)

        if self.save_fig == True:
            plt.savefig(save_fig_path+f'plot_fit_{self.name}.pdf', bbox_inches='tight')
            plt.savefig(save_fig_path+f'plot_fit_{self.name}.jpg', dpi=300, bbox_inches='tight')
        # plt.show()
        # plt.close()
    
    # ---------MC error for continuum parameters-------------------
    def _conti_mc(self, x, y, err, pp0, pp_limits, n_trails):
        """Calculate the continual parameters' Monte carlo errrors"""
        all_para = np.zeros(len(pp0)*n_trails).reshape(len(pp0), n_trails)
        all_L = np.zeros(3*n_trails).reshape(3, n_trails)
        n_Fe_flux = np.array(self.Fe_flux_range).flatten().shape[0]//2
        all_Fe_flux = np.zeros(n_Fe_flux*n_trails).reshape(n_Fe_flux, n_trails)
        all_Fe_ew = np.zeros_like(all_Fe_flux)
        for tra in range(n_trails):
            flux = y+np.random.randn(len(y))*err
            conti_fit = kmpfit.Fitter(residuals=self._residuals, data=(x, flux, err), maxiter=50)
            conti_fit.parinfo = pp_limits
            conti_fit.fit(params0=pp0)
            all_para[:, tra] = conti_fit.params
            all_L[:, tra] = np.asarray(self._L_conti(x, conti_fit.params))
            if self.Fe_flux_range is not None:
                for i, wrange in enumerate(self.Fe_range_list):
                    flux_tmp, ew_tmp = self.Fe_line_prop(conti_fit.params, wrange)
                    all_Fe_flux[i, tra] = flux_tmp
                    all_Fe_ew[i, tra] = ew_tmp
                self.all_Fe_flux = all_Fe_flux
                self.all_Fe_ew = all_Fe_ew
        if self.Fe_flux_range is not None:
            df_fe_flux = pd.DataFrame(all_Fe_flux.T, columns=self.Fe_wave_keys)
            df_fe_ew = pd.DataFrame(all_Fe_ew.T, columns=self.Fe_wave_keys)
            self.df_fe_flux = df_fe_flux
            self.df_fe_ew = df_fe_ew
        all_para_std = all_para.std(axis=1)
        all_L_std = all_L.std(axis=1)
        return all_para_std, all_L_std

    def Fe_line_prop(self, pp, subrange=None):
        wave = self.wave
        if subrange is None:
            subrange = [4434, 4684]
        f_fe_mgii_model = self.Fe_flux_mgii(wave, pp[0:3])
        f_fe_balmer_model = self.Fe_flux_balmer(wave, pp[3:6])
        f_fe_verner_model = self.Fe_flux_verner(wave, pp[3:6])
        f_pl_model = pp[6]*(wave/self.pl_pivot)**pp[7]
        if self.broken_pl == True:
            f_pl_model = broken_pl_model(wave, pp[7], pp[14], pp[6])
        f_bc_model = self.Balmer_conti(wave, pp[8]) + self.Balmer_high_order(wave, pp[9:11])
        f_poly_model = self.F_poly_conti(wave, pp[11:14])
        f_conti_no_fe = f_pl_model+f_poly_model+f_bc_model
        if self.Fe_verner09 == True:
            f_fe = f_fe_verner_model
        else:
            f_fe = f_fe_mgii_model + f_fe_balmer_model 
        range_low = subrange[0]
        range_high = subrange[1]
        ind = ((wave <= range_high) & (wave >= range_low))
        wave_in = wave[ind]
        contiflux = f_conti_no_fe[ind]
        Fe_flux = integrate.trapezoid(f_fe[ind], wave_in)
        Fe_ew = integrate.trapezoid(f_fe[ind]/contiflux, wave_in)
        return Fe_flux, Fe_ew

    def cal_Fe_line_res(self):
        Fe_line_result = {}
        keys = self.Fe_wave_keys
        print(repr(keys))
        for i, key in enumerate(keys):
            pp=self.conti_fit.params
            subrange=self.Fe_range_list[i]
            res = self.Fe_line_prop(pp=pp, subrange=subrange)
            res_name_flux = 'Fe_{}_flux'.format(key)
            res_name_ew = 'Fe_{}_ew'.format(key)
            Fe_line_result.update({res_name_flux:res[0]})
            Fe_line_result.update({res_name_ew:res[1]})
        if self.MC == True:
            for i, key in enumerate(keys):
                err_name_flux = 'Fe_{}_flux_err'.format(key)
                err_tmp_flux = self.df_fe_flux[key].std()
                err_name_ew = 'Fe_{}_ew_err'.format(key)
                err_tmp_ew = self.df_fe_ew[key].std()
                Fe_line_result.update({err_name_flux:err_tmp_flux})
                Fe_line_result.update({err_name_ew:err_tmp_ew})            
        self.Fe_line_result = Fe_line_result

    # line function-----------
    def _DoLineFit(self, wave, line_flux, err, f):
        """Fit the emission lines with Gaussian profile """
        
        # remove abosorbtion line in emission line region
        # remove the pixels below continuum 
        ind_neg_line = ~np.where(((((wave > 2700.) & (wave < 2900.)) | ((wave > 1700.) & (wave < 1970.)) | (
                (wave > 1500.) & (wave < 1700.)) | ((wave > 1290.) & (wave < 1450.)) | (
                                           (wave > 1150.) & (wave < 1290.))) & (line_flux < -err)), True, False)
        
        # read line parameter
        linepara = fits.open(self.path+'qsopar.fits')
        linelist = linepara[1].data
        mask_compname = self.mask_compname
        if mask_compname is not None:
            linelist = linelist[linelist['compname']!=mask_compname]
        self.linelist = linelist
        
        ind_kind_line = np.where((linelist['lambda'] > wave.min()) & (linelist['lambda'] < wave.max()), True, False)
        if ind_kind_line.any() == True:
            # sort complex name with line wavelength
            uniq_linecomp, uniq_ind = np.unique(linelist['compname'][ind_kind_line], return_index=True)
            uniq_linecomp_sort = uniq_linecomp[linelist['lambda'][ind_kind_line][uniq_ind].argsort()]
            ncomp = len(uniq_linecomp_sort)
            compname = linelist['compname']
            allcompcenter = np.sort(linelist['lambda'][ind_kind_line][uniq_ind])
            
            # loop over each complex and fit n lines simutaneously
            
            comp_result = np.array([])
            comp_result_type = np.array([])
            comp_result_name = np.array([])
            gauss_result = np.array([])
            gauss_result_type = np.array([])
            gauss_result_name = np.array([])
            all_comp_range = np.array([])
            fur_result = np.array([])
            fur_result_type = np.array([])
            fur_result_name = np.array([])
            self.na_all_dict = {}
            
            for ii in range(ncomp):
                compcenter = allcompcenter[ii]
                ind_line = np.where(linelist['compname'] == uniq_linecomp_sort[ii], True, False)  # get line index
                linecompname = uniq_linecomp_sort[ii]
                nline_fit = np.sum(ind_line)  # n line in one complex
                linelist_fit = linelist[ind_line]
                # n gauss in each line
                ngauss_fit = np.asarray(linelist_fit['ngauss'], dtype=int)
                
                # for iitmp in range(nline_fit):   # line fit together
                comp_range = [linelist_fit[0]['minwav'], linelist_fit[0]['maxwav']]  # read complex range from table
                all_comp_range = np.concatenate([all_comp_range, comp_range])
                
                # ----tie lines--------
                self._do_tie_line(linelist, ind_line)
                
                # get the pixel index in complex region and remove negtive abs in line region
                ind_n = np.where((wave > comp_range[0]) & (wave < comp_range[1]) & (ind_neg_line == True), True, False)
                ind_n_all = np.where((wave > comp_range[0]) & (wave < comp_range[1]))
                num_good_pix = np.sum(ind_n)
                comp_name = linelist['compname'][ind_line][0]
                # print('Number of good pixels in line complex {}: {}.'.format(comp_name, num_good_pix))
                if np.sum(ind_n) > 10:
                    # call kmpfit or lmfit for lines (controlled by migration flag)
                    if getattr(migration_config, 'use_lmfit_lines', False):
                        line_fit = self._do_line_lmfit(linelist, line_flux, ind_line, ind_n, nline_fit, ngauss_fit)
                    else:
                        if _kmpfit is None:
                            raise ImportError("kapteyn is required for kmpfit path. Install kapteyn or enable lmfit via migration_config.use_lmfit_lines=True")
                        line_fit = self._do_line_kmpfit(linelist, line_flux, ind_line, ind_n, nline_fit, ngauss_fit)
                    if comp_name == 'Hb':
                        na_dict = self.na_line_nomc(line_fit, linecompname, ind_line, nline_fit, ngauss_fit)
                        wing_status = check_wings(na_dict)
                        self.wing_status = wing_status
                        if None in wing_status:
                            pass
                        elif np.sum(wing_status)<2:
                            linelist = linelist[linelist['linename']!='OIII4959w']
                            linelist = linelist[linelist['linename']!='OIII5007w']
                            self.linelist = linelist
                            ind_kind_line = np.where((linelist['lambda'] > wave.min()) & (linelist['lambda'] < wave.max()), True, False)
                            uniq_linecomp, uniq_ind = np.unique(linelist['compname'][ind_kind_line], return_index=True)
                            uniq_linecomp_sort = uniq_linecomp[linelist['lambda'][ind_kind_line][uniq_ind].argsort()]
                            ind_line = np.where(linelist['compname'] == uniq_linecomp_sort[ii], True, False) 
                            linecompname = uniq_linecomp_sort[ii]
                            nline_fit = np.sum(ind_line)  # n line in one complex
                            linelist_fit = linelist[ind_line]
                            ngauss_fit = np.asarray(linelist_fit['ngauss'], dtype=int)
                            self._do_tie_line(linelist, ind_line)
                            if getattr(migration_config, 'use_lmfit_lines', False):
                                line_fit = self._do_line_lmfit(linelist, line_flux, ind_line, ind_n, nline_fit, ngauss_fit)
                            else:
                                if _kmpfit is None:
                                    raise ImportError("kapteyn is required for kmpfit path. Install kapteyn or enable lmfit via migration_config.use_lmfit_lines=True")
                                line_fit = self._do_line_kmpfit(linelist, line_flux, ind_line, ind_n, nline_fit, ngauss_fit)
                   
                    # calculate uncertainties (always save errors)
                    if getattr(migration_config, 'use_lmfit_lines', False):
                        # lmfit: always use stderr/covariance propagation
                        try:
                            n_params = len(line_fit.params)
                            param_stderr = []
                            for i in range(n_params):
                                par = line_fit.result.params.get(f"p{i}")
                                se = getattr(par, 'stderr', None) if par is not None else None
                                if se is None or (isinstance(se, float) and (np.isnan(se))):
                                    param_stderr.append(0.0)
                                else:
                                    param_stderr.append(float(se))
                            all_para_std = np.array(param_stderr)
                        except Exception:
                            all_para_std = np.zeros(len(line_fit.params))

                        # Compute metric uncertainties (analytic) for lmfit even when MC=False
                        try:
                            res = line_fit.result
                            var_names = list(getattr(res, 'var_names', []))
                            cov = getattr(res, 'covar', None)
                            n_all = len(line_fit.params)
                            Cfull = np.zeros((n_all, n_all), dtype=float)
                            if cov is not None and len(var_names) > 0:
                                for a, name_a in enumerate(var_names):
                                    ia = int(name_a[1:]) if name_a.startswith('p') else None
                                    if ia is None:
                                        continue
                                    for b, name_b in enumerate(var_names):
                                        ib = int(name_b[1:]) if name_b.startswith('p') else None
                                        if ib is None:
                                            continue
                                        Cfull[ia, ib] = cov[a, b]
                            else:
                                for j in range(n_all):
                                    Cfull[j, j] = all_para_std[j] * all_para_std[j]
                        except Exception:
                            n_all = len(line_fit.params)
                            Cfull = np.diag(all_para_std ** 2)

                        def metrics_from_p(pvec):
                            pp = pvec.copy()
                            if self.tie_lambda:
                                if len(self.ind_tie_vindex1) > 1:
                                    for xx in range(len(self.ind_tie_vindex1) - 1):
                                        pp[int(self.ind_tie_vindex1[xx + 1])] = pp[int(self.ind_tie_vindex1[0])] + self.delta_lambda1[xx]
                                if len(self.ind_tie_vindex2) > 1:
                                    for xx in range(len(self.ind_tie_vindex2) - 1):
                                        pp[int(self.ind_tie_vindex2[xx + 1])] = pp[int(self.ind_tie_vindex2[0])] + self.delta_lambda2[xx]
                            if self.tie_width:
                                if len(self.ind_tie_windex1) > 1:
                                    for xx in range(len(self.ind_tie_windex1) - 1):
                                        pp[int(self.ind_tie_windex1[xx + 1])] = pp[int(self.ind_tie_windex1[0])]
                                if len(self.ind_tie_windex2) > 1:
                                    for xx in range(len(self.ind_tie_windex2) - 1):
                                        pp[int(self.ind_tie_windex2[xx + 1])] = pp[int(self.ind_tie_windex2[0])]
                            if len(self.ind_tie_findex1) > 0 and self.tie_flux_1:
                                pp[int(self.ind_tie_findex1[1])] = pp[int(self.ind_tie_findex1[0])] * self.fvalue_factor_1
                            if len(self.ind_tie_findex2) > 0 and self.tie_flux_2:
                                pp[int(self.ind_tie_findex2[1])] = pp[int(self.ind_tie_findex2[0])] * self.fvalue_factor_2
                            m = self.line_prop(compcenter, pp, 'broad')
                            return np.array(m, dtype=float)

                        base_p = np.array([line_fit.result.params[f"p{i}"].value for i in range(n_all)], dtype=float)
                        Jm = np.zeros((5, n_all), dtype=float)
                        for j in range(n_all):
                            pj = base_p[j]
                            step = max(1e-6, 1e-2 * (abs(pj) if np.isfinite(pj) else 1.0))
                            p_plus = base_p.copy(); p_minus = base_p.copy()
                            p_plus[j] = pj + step
                            p_minus[j] = pj - step
                            mp = metrics_from_p(p_plus)
                            mm = metrics_from_p(p_minus)
                            Jm[:, j] = (mp - mm) / (2.0 * step)
                        covM = Jm @ Cfull @ Jm.T
                        stdM = np.sqrt(np.clip(np.diag(covM), 0, np.inf))
                        fwhm_std, sigma_std, ew_std, peak_std, area_std = [float(x) for x in stdM]
                        # compute na_dict deterministically without MC
                        na_dict = self.na_line_nomc(line_fit, linecompname, ind_line, nline_fit, ngauss_fit)
                        self.na_all_dict.update(na_dict)
                    else:
                        if self.MC == True and self.n_trails > 0:
                            all_para_std, fwhm_std, sigma_std, ew_std, peak_std, area_std, na_dict = self.new_line_mc(
                                np.log(wave[ind_n]), line_flux[ind_n], err[ind_n], self.line_fit_ini, self.line_fit_par,
                                self.n_trails, compcenter, linecompname, ind_line, nline_fit, linelist_fit, ngauss_fit)
                            self.na_all_dict.update(na_dict)
                        else:
                            # no lmfit and no MC: set zeros for stds
                            all_para_std = np.zeros(len(line_fit.params))
                            fwhm_std = sigma_std = ew_std = peak_std = area_std = 0.0
                            na_dict = self.na_line_nomc(line_fit, linecompname, ind_line, nline_fit, ngauss_fit)
                            self.na_all_dict.update(na_dict)
                    
                    # ----------------------get line fitting results----------------------
                    # complex parameters
                    
                    # tie lines would reduce the number of parameters increasing the dof
                    dof_fix = 0
                    if self.tie_lambda == True:
                        dof_fix += np.max((len(self.ind_tie_vindex1), 1))-1
                        dof_fix += np.max((len(self.ind_tie_vindex2), 1))-1
                    if self.tie_width == True:
                        dof_fix += np.max((len(self.ind_tie_windex1), 1))-1
                        dof_fix += np.max((len(self.ind_tie_windex2), 1))-1
                    if self.tie_flux_1 == True:
                        dof_fix += np.max((len(self.ind_tie_findex1), 1))-1
                        dof_fix += np.max((len(self.ind_tie_findex2), 1))-1
                    med_sn = np.nanmedian(self.sn_obs[ind_n_all])
                    comp_result_tmp = np.array(
                        [[num_good_pix], [line_fit.status], [line_fit.chi2_min],
                         [line_fit.chi2_min/(line_fit.dof+dof_fix)], [line_fit.niter],
                         [line_fit.dof+dof_fix], [med_sn]]).flatten()
                    comp_result_type_tmp = np.array(['int', 'int', 'float', 'float', 'int', 'int', 'float'])
                    comp_result_name_tmp = np.array(
                        ['LINE_NPIX_'+comp_name, comp_name+'_line_status', comp_name+'_line_min_chi2',
                         comp_name+'_line_red_chi2', comp_name+'_niter', comp_name+'_ndof', 
                         'LINE_MED_SN_'+comp_name])
                    comp_result = np.concatenate([comp_result, comp_result_tmp])
                    comp_result_name = np.concatenate([comp_result_name, comp_result_name_tmp])
                    comp_result_type = np.concatenate([comp_result_type, comp_result_type_tmp])
                    
                    # gauss result -------------
                    
                    gauss_tmp = np.array([])
                    gauss_type_tmp = np.array([])
                    gauss_name_tmp = np.array([])
                    
                    for gg in range(len(line_fit.params)):
                        gauss_tmp = np.concatenate([gauss_tmp, np.array([line_fit.params[gg]])])
                        if self.MC == True and self.n_trails > 0:
                            gauss_tmp = np.concatenate([gauss_tmp, np.array([all_para_std[gg]])])
                    gauss_result = np.concatenate([gauss_result, gauss_tmp])
                    
                    # gauss result name -----------------
                    for n in range(nline_fit):
                        for nn in range(int(ngauss_fit[n])):
                            line_name = linelist['linename'][ind_line][n]+'_'+str(nn+1)
                            if self.MC == True and self.n_trails > 0:
                                gauss_type_tmp_tmp = ['float', 'float', 'float', 'float', 'float', 'float']
                                gauss_name_tmp_tmp = [line_name+'_scale', line_name+'_scale_err',
                                                      line_name+'_centerwave', line_name+'_centerwave_err',
                                                      line_name+'_sigma', line_name+'_sigma_err']
                            else:
                                gauss_type_tmp_tmp = ['float', 'float', 'float']
                                gauss_name_tmp_tmp = [line_name+'_scale', line_name+'_centerwave', line_name+'_sigma']
                            gauss_name_tmp = np.concatenate([gauss_name_tmp, gauss_name_tmp_tmp])
                            gauss_type_tmp = np.concatenate([gauss_type_tmp, gauss_type_tmp_tmp])
                    gauss_result_type = np.concatenate([gauss_result_type, gauss_type_tmp])
                    gauss_result_name = np.concatenate([gauss_result_name, gauss_name_tmp])
                    
                    # further line parameters ----------
                    fur_result_tmp = np.array([])
                    fur_result_type_tmp = np.array([])
                    fur_result_name_tmp = np.array([])
                    # if comp_name == 'CIV':
                    #     fwhm, sigma, ew, peak, area = self.comb_line_prop(compcenter, line_fit.params)
                    # else:
                    fwhm, sigma, ew, peak, area = self.line_prop(compcenter, line_fit.params, 'broad')
                    br_name = uniq_linecomp_sort[ii]
                    
                    # Always save value + error pairs
                    fur_result_tmp = np.array(
                        [fwhm, fwhm_std, sigma, sigma_std, ew, ew_std, peak, peak_std, area, area_std])
                    fur_result_type_tmp = np.concatenate([fur_result_type_tmp,
                                                          ['float', 'float', 'float', 'float', 'float', 'float',
                                                           'float', 'float', 'float', 'float']])
                    fur_result_name_tmp = np.array(
                        [br_name+'_whole_br_fwhm', br_name+'_whole_br_fwhm_err', br_name+'_whole_br_sigma',
                         br_name+'_whole_br_sigma_err', br_name+'_whole_br_ew', br_name+'_whole_br_ew_err',
                         br_name+'_whole_br_peak', br_name+'_whole_br_peak_err', br_name+'_whole_br_area',
                         br_name+'_whole_br_area_err'])
                    fur_result = np.concatenate([fur_result, fur_result_tmp])
                    fur_result_type = np.concatenate([fur_result_type, fur_result_type_tmp])
                    fur_result_name = np.concatenate([fur_result_name, fur_result_name_tmp])
                    if comp_name == 'CIV':
                        civ_fwhm, civ_sigma, civ_ew, civ_peak, civ_area = self.comb_line_prop(compcenter, line_fit.params)
                        fur_result_civ = np.array(
                            [civ_fwhm, civ_sigma, civ_ew, civ_peak, civ_area])
                        fur_result_type_civ = np.array(
                            ['float', 'float', 'float', 'float', 'float'])
                        fur_result_name_civ = np.array(
                            ['CIV_whole_fwhm', 'CIV_whole_sigma', 'CIV_whole_ew',
                            'CIV_whole_peak', 'CIV_whole_area'])
                        fur_result = np.concatenate([fur_result, fur_result_civ])
                        fur_result_type = np.concatenate([fur_result_type, fur_result_type_civ])
                        fur_result_name = np.concatenate([fur_result_name, fur_result_name_civ])                
                else:
                    print("less than 10 pixels in line fitting!")
            
            # line_result = np.concatenate([comp_result, gauss_result, fur_result])
            # line_result_type = np.concatenate([comp_result_type, gauss_result_type, fur_result_type])
            # line_result_name = np.concatenate([comp_result_name, gauss_result_name, fur_result_name])
            line_result = np.concatenate([comp_result, fur_result])
            line_result_type = np.concatenate([comp_result_type, fur_result_type])
            line_result_name = np.concatenate([comp_result_name, fur_result_name])        
        else:
            line_result = np.array([])
            line_result_name = np.array([])
            comp_result = np.array([])
            gauss_result = np.array([])
            gauss_result_name = np.array([])
            line_result_type = np.array([])
            ncomp = 0
            all_comp_range = np.array([])
            uniq_linecomp_sort = np.array([])
            print("No line to fit! Pleasse set Line_fit to FALSE or enlarge wave_range!")

        self.comp_result = comp_result
        self.comp_result_name = comp_result_name
        self.gauss_result = gauss_result
        self.gauss_result_name = gauss_result_name
        # if self.MC == True and self.na_all_dict:
        self.cal_na_line_res()
        na_line_result = np.array(list(self.na_line_result.values()))
        na_line_result_type = np.full_like(na_line_result,'float',dtype=object)
        na_line_result_name = np.asarray(list(self.na_line_result.keys()))
        line_result = np.concatenate([line_result, na_line_result])
        line_result_type = np.concatenate([line_result_type, na_line_result_type])
        line_result_name = np.concatenate([line_result_name, na_line_result_name])
        self.line_result = line_result
        self.line_result_type = line_result_type
        self.line_result_name = line_result_name
        self.ncomp = ncomp
        self.line_flux = line_flux
        self.all_comp_range = all_comp_range
        self.uniq_linecomp_sort = uniq_linecomp_sort
        return self.line_result, self.line_result_name


    def na_line_nomc(self, line_fit, linecompname,
                    ind_line, nline_fit, ngauss_fit):
        linelist = self.linelist
        linenames = linelist[linelist['compname']==linecompname]['linename']
        na_all_dict = {}
        for line in linenames: 
            # if ('br' not in line and 'na' not in line) or ('Ha_na' in line) or ('Hb_na' in line):
            if 'br' not in line:
                emp_dict = {'fwhm': [],
                            'sigma' : [],
                            'ew' : [],
                            'peak' : [],
                            'area' : []}
                na_all_dict.setdefault(line, emp_dict)  
        all_line_name = []
        for n in range(nline_fit):
            for nn in range(int(ngauss_fit[n])):
                line_name = linelist['linename'][ind_line][n]
                all_line_name.append(line_name)
                # print('Line {} Gaussian component {} added.'.format(line_name, nn+1))
        all_line_name = np.asarray(all_line_name)
        # print('All line names: {}'.format(all_line_name))

        for line in linenames: 
            # if ('br' not in line and 'na' not in line) or ('Ha_na' in line) or ('Hb_na' in line) or ('CIV' in line):
            if 'br' not in line:
                try:
                    par_ind = np.where(all_line_name==line)[0][0]*3
                    linecenter = float(linelist[linelist['linename']==line]['lambda'][0])
                    na_tmp = self.line_prop(linecenter, line_fit.params[par_ind:par_ind+3], 'narrow')
                    if na_tmp[0] == 0:
                        na_tmp = self.line_prop(linecenter, line_fit.params[par_ind:par_ind+3], 'broad')
                    na_all_dict[line]['fwhm'].append(na_tmp[0])
                    na_all_dict[line]['sigma'].append(na_tmp[1])
                    na_all_dict[line]['ew'].append(na_tmp[2])
                    na_all_dict[line]['peak'].append(na_tmp[3])
                    na_all_dict[line]['area'].append(na_tmp[4])
                except:
                    warnings.warn('Line {} parameters unavailable.'.format(line))
                    pass
                    
        for line in linenames: 
            # if ('br' not in line and 'na' not in line) or ('Ha_na' in line) or ('Hb_na' in line) or ('CIV_na' in line):
            if 'br' not in line:    
                na_all_dict[line]['fwhm'] = getnonzeroarr(np.asarray(na_all_dict[line]['fwhm']))
                na_all_dict[line]['sigma'] = getnonzeroarr(np.asarray(na_all_dict[line]['sigma']))
                na_all_dict[line]['ew'] = getnonzeroarr(np.asarray(na_all_dict[line]['ew']))
                na_all_dict[line]['peak'] = getnonzeroarr(np.asarray(na_all_dict[line]['peak']))
                na_all_dict[line]['area'] = getnonzeroarr(np.asarray(na_all_dict[line]['area']))
        
        return na_all_dict


    # ---------MC error for emission line parameters-------------------
    def new_line_mc(self, x, y, err, pp0, pp_limits, n_trails, compcenter, linecompname,
                    ind_line, nline_fit, linelist_fit, ngauss_fit):
        """calculate the Monte Carlo errror of line parameters"""
        linelist = self.linelist
        linenames = linelist[linelist['compname']==linecompname]['linename']
        self.linenames_mc = linenames
        all_para_1comp = np.zeros(len(pp0)*n_trails).reshape(len(pp0), n_trails)
        all_para_std = np.zeros(len(pp0))
        all_fwhm = np.zeros(n_trails)
        all_sigma = np.zeros(n_trails)
        all_ew = np.zeros(n_trails)
        all_peak = np.zeros(n_trails)
        all_area = np.zeros(n_trails)
        na_all_dict = {}
        if 'OIII4959w' in linenames and 'OIII5007w' in linenames:
            linenames = np.append(linenames, ['OIII4959_whole', 'OIII5007_whole'])  
        if 'CIV_br' in linenames and 'CIV_na' in linenames:
            linenames = np.append(linenames, ['CIV_whole'])  
        for line in linenames: 
            # if ('br' not in line and 'na' not in line) or ('Ha_na' in line) or ('Hb_na' in line) or ('CIV_na' in line) or ('CIV_whole' in line):
            if 'br' not in line:
                emp_dict = {'fwhm': [],
                            'sigma' : [],
                            'ew' : [],
                            'peak' : [],
                            'area' : []}
                na_all_dict.setdefault(line, emp_dict)

        for tra in range(n_trails):
            flux = y+np.random.randn(len(y))*err
            line_fit = kmpfit.Fitter(residuals=self._residuals_line, data=(x, flux, err), maxiter=50)
            line_fit.parinfo = pp_limits
            line_fit.fit(params0=pp0)
            line_fit.params = self.newpp
            all_para_1comp[:, tra] = line_fit.params
            
            # further line properties
            all_fwhm[tra], all_sigma[tra], all_ew[tra], all_peak[tra], all_area[tra] 
            broad_all = self.line_prop(compcenter, line_fit.params, 'broad')
            all_fwhm[tra] = broad_all[0]
            all_sigma[tra] =  broad_all[1]
            all_ew[tra] = broad_all[2]
            all_peak[tra] = broad_all[3]
            all_area[tra] = broad_all[4]     
            all_line_name = []
            for n in range(nline_fit):
                for nn in range(int(ngauss_fit[n])):
                    line_name = linelist['linename'][ind_line][n]
                    all_line_name.append(line_name)
            all_line_name = np.asarray(all_line_name)

            for line in linenames: 
                if ('br' not in line) and ('whole' not in line):
                    try:
                        par_ind = np.where(all_line_name==line)[0][0]*3
                        linecenter = float(linelist[linelist['linename']==line]['lambda'][0])
                        na_tmp = self.line_prop(linecenter, line_fit.params[par_ind:par_ind+3], 'narrow')
                        if line_fit.params[par_ind+2] > 0.0017:
                            na_tmp = self.line_prop(linecenter, line_fit.params[par_ind:par_ind+3], 'broad')
                        na_all_dict[line]['fwhm'].append(na_tmp[0])
                        na_all_dict[line]['sigma'].append(na_tmp[1])
                        na_all_dict[line]['ew'].append(na_tmp[2])
                        na_all_dict[line]['peak'].append(na_tmp[3])
                        na_all_dict[line]['area'].append(na_tmp[4])
                    except:
                        print('Line {} parameters mismatch.'.format(line))
                        pass
                elif ('whole' in line) and ('CIV_whole' not in line):
                    linec = line.split('_')[0]
                    linew = linec+'w'
                    # print('Line: {}. Core: {}. Wing: {}.'.format(line, linec, linew))
                    par_ind1 = np.where(all_line_name==linec)[0][0]*3
                    par_ind2 = np.where(all_line_name==linew)[0][0]*3
                    inds1 = np.concatenate([np.arange(par_ind1, par_ind1+3),
                                            np.arange(par_ind2, par_ind2+3)])
                    linecenter = float(linelist[linelist['linename']==linec]['lambda'][0])
                    na_tmp = self.comb_line_prop(linecenter, line_fit.params[inds1])
                    na_all_dict[line]['fwhm'].append(na_tmp[0])
                    na_all_dict[line]['sigma'].append(na_tmp[1])
                    na_all_dict[line]['ew'].append(na_tmp[2])
                    na_all_dict[line]['peak'].append(na_tmp[3])
                    na_all_dict[line]['area'].append(na_tmp[4])  
                elif 'CIV_whole' in line:
                    all_civ = self.comb_line_prop(compcenter, line_fit.params)
                    na_all_dict[line]['fwhm'].append(all_civ[0])
                    na_all_dict[line]['sigma'].append(all_civ[1])
                    na_all_dict[line]['ew'].append(all_civ[2])
                    na_all_dict[line]['peak'].append(all_civ[3])
                    na_all_dict[line]['area'].append(all_civ[4]) 
        for line in linenames: 
            # if ('br' not in line and 'na' not in line) or ('Ha_na' in line) or ('Hb_na' in line) or ('CIV_na' in line) or ('CIV_whole' in line):
            if 'br' not in line:
                na_all_dict[line]['fwhm'] = getnonzeroarr(np.asarray(na_all_dict[line]['fwhm']))
                na_all_dict[line]['sigma'] = getnonzeroarr(np.asarray(na_all_dict[line]['sigma']))
                na_all_dict[line]['ew'] = getnonzeroarr(np.asarray(na_all_dict[line]['ew']))
                na_all_dict[line]['peak'] = getnonzeroarr(np.asarray(na_all_dict[line]['peak']))
                na_all_dict[line]['area'] = getnonzeroarr(np.asarray(na_all_dict[line]['area']))
        for st in range(len(pp0)):
            all_para_std[st] = all_para_1comp[st, :].std()
        
        return all_para_std, all_fwhm.std(), all_sigma.std(), all_ew.std(), all_peak.std(), all_area.std(), na_all_dict


    def cal_na_line_res(self):
        linelist = self.linelist
        na_line_result = {}
        df_gauss = pd.DataFrame(data=np.array([self.gauss_result]),
                                columns=self.gauss_result_name)
        #caution: dtypes are all object for df_gauss, conversion needed.
        keys = list(self.na_all_dict.keys())
        try:
            keys.remove('OIII4959_whole')
            keys.remove('OIII5007_whole')
        except:
            pass
        try:
            keys.remove('CIV_whole')
        except:
            pass
        print(repr(keys))
        if self.MC == True and self.na_all_dict:
            if 'CIV_whole' in self.na_all_dict.keys():
                par_list = list(self.na_all_dict['CIV_whole'].keys())
                for ii in range(len(par_list)):
                    par = par_list[ii]
                    # res_name_tmp = comp_tmp+'_'+par
                    # res_tmp = na_tmp[ii]
                    err_name_tmp = 'CIV_whole_'+par+'_err'
                    err_tmp = self.na_all_dict['CIV_whole'][par].std()
                    if err_tmp == 0:
                        # res_tmp = 0.0
                        err_tmp = 0.0
                    # na_line_result.update({res_name_tmp:res_tmp})
                    na_line_result.update({err_name_tmp:err_tmp})
            for line in keys:
                linecenter = float(linelist[linelist['linename']==line]['lambda'].item())
                line_scale = float(df_gauss[line+'_1_scale'][0])
                line_centerwave = float(df_gauss[line+'_1_centerwave'][0])
                line_sigma = float(df_gauss[line+'_1_sigma'][0])
                line_param = np.array([line_scale,line_centerwave,line_sigma])
                na_tmp = self.line_prop(linecenter, line_param, 'narrow')
                if line_sigma > 0.0017:
                    na_tmp = self.line_prop(linecenter, line_param, 'broad')
                par_list = list(self.na_all_dict[line].keys())
                for i in range(len(par_list)):
                    par = par_list[i]
                    res_name_tmp = line+'_'+par
                    res_tmp = na_tmp[i]
                    err_name_tmp = line+'_'+par+'_err'
                    err_tmp = self.na_all_dict[line][par].std()
                    if res_tmp == 0:
                        res_tmp = 0.0
                        err_tmp = 0.0
                    na_line_result.update({res_name_tmp:res_tmp})
                    na_line_result.update({err_name_tmp:err_tmp})
            if 'OIII4959_whole' in self.na_all_dict.keys() and 'OIII5007_whole' in self.na_all_dict.keys():
                for i, comp_tmp in enumerate(['OIII4959_whole', 'OIII5007_whole']):
                    linec = comp_tmp.split('_')[0]
                    linew = linec+'w'
                    linecenter = float(linelist[linelist['linename']==linec]['lambda'].item())
                    line_scale1 = float(df_gauss[linec+'_1_scale'][0])
                    line_centerwave1 = float(df_gauss[linec+'_1_centerwave'][0])
                    line_sigma1 = float(df_gauss[linec+'_1_sigma'][0])
                    line_scale2 = float(df_gauss[linew+'_1_scale'][0])
                    line_centerwave2 = float(df_gauss[linew+'_1_centerwave'][0])
                    line_sigma2 = float(df_gauss[linew+'_1_sigma'][0])
                    line_param = np.array([line_scale1,line_centerwave1,line_sigma1,
                                           line_scale2,line_centerwave2,line_sigma2])
                    na_tmp = self.comb_line_prop(linecenter, line_param)
                    par_list = list(self.na_all_dict[comp_tmp].keys())
                    for ii in range(len(par_list)):
                        par = par_list[ii]
                        res_name_tmp = comp_tmp+'_'+par
                        res_tmp = na_tmp[ii]
                        err_name_tmp = comp_tmp+'_'+par+'_err'
                        err_tmp = self.na_all_dict[comp_tmp][par].std()
                        if res_tmp == 0:
                            res_tmp = 0.0
                            err_tmp = 0.0
                        na_line_result.update({res_name_tmp:res_tmp})
                        na_line_result.update({err_name_tmp:err_tmp})
        elif self.MC == False and self.na_all_dict:
            for line in self.na_all_dict.keys():
                par_list = list(self.na_all_dict[line].keys())
                for i in range(len(par_list)):
                    par = par_list[i]
                    res_name_tmp = line+'_'+par
                    res_list = self.na_all_dict[line][par]
                    if np.asarray(res_list).size > 0:
                        res_tmp = res_list[0]
                    else:
                        res_tmp = 0.0
                    if res_tmp == 0:
                        res_tmp = 0.0
                    na_line_result.update({res_name_tmp:res_tmp})
        self.na_line_result = na_line_result


    # -----line properties calculation function--------
    def line_prop(self, compcenter, pp, linetype, save_luminosity=False):
        """
        Calculate the further results for the broad component in emission lines, e.g., FWHM, sigma, peak, line flux
        The compcenter is the theortical vacuum wavelength for the broad compoenet.
        """
        pp = pp.astype(float)
        if linetype == 'broad':
            ind_br = np.repeat(np.where(pp[2::3] > 0.0017, True, False), 3)
        
        elif linetype == 'narrow':
            ind_br = np.repeat(np.where(pp[2::3] <= 0.0017, True, False), 3)
        
        else:
            raise RuntimeError("line type should be 'broad' or 'narrow'!")
        
        ind_br[9:] = False  # to exclude the broad OIII and broad He II
        
        p = pp[ind_br]
        del pp
        pp = p
        
        c = 299792.458  # km/s
        n_gauss = int(len(pp)/3)
        if n_gauss == 0:
            fwhm, sigma, ew, peak, area = 0., 0., 0., 0., 0.
        else:
            cen = np.zeros(n_gauss)
            sig = np.zeros(n_gauss)
            
            for i in range(n_gauss):
                cen[i] = pp[3*i+1]
                sig[i] = pp[3*i+2]
            
            # print cen,sig,area
            left = min(cen-3*sig)
            right = max(cen+3*sig)
            disp = 1.e-4*np.log(10.)
            npix = int((right-left)/disp)
            
            xx = np.linspace(left, right, npix)
            yy = self.Manygauss(xx, pp)
        
            # here I directly use the continuum model to avoid the inf bug of EW when the spectrum range passed in is too short
            contiflux = self.conti_fit.params[6]*(np.exp(xx)/self.pl_pivot)**self.conti_fit.params[7]+self.F_poly_conti(
                np.exp(xx), self.conti_fit.params[11:])+self.Balmer_conti(np.exp(xx), self.conti_fit.params[8]) + self.Balmer_high_order(np.exp(xx), self.conti_fit.params[9:11])            
            if self.broken_pl == True:
                f = interpolate.InterpolatedUnivariateSpline(
                    self.wave, 
                    self.f_conti_model)
                contiflux = f(np.exp(xx))
            
            # find the line peak location
            ypeak = yy.max()
            ypeak_ind = np.argmax(yy)
            peak = np.exp(xx[ypeak_ind])
            
            # find the FWHM in km/s
            # take the broad line we focus and ignore other broad components such as [OIII], HeII
            
            if n_gauss > 3:
                spline = interpolate.UnivariateSpline(xx,
                                                      self.Manygauss(xx, pp[0:9])-np.max(self.Manygauss(xx, pp[0:9]))/2,
                                                      s=0)
            else:
                spline = interpolate.UnivariateSpline(xx, yy-np.max(yy)/2, s=0)
            if len(spline.roots()) > 0:
                fwhm_left, fwhm_right = spline.roots().min(), spline.roots().max()
                fwhm = abs(np.exp(fwhm_left)-np.exp(fwhm_right))/compcenter*c
                
                # calculate the line sigma and EW in normal wavelength
                line_flux = self.Manygauss(xx, pp)
                line_wave = np.exp(xx)
                lambda0 = integrate.trapezoid(line_flux, line_wave)  # calculate the total broad line flux
                lambda1 = integrate.trapezoid(line_flux*line_wave, line_wave)
                lambda2 = integrate.trapezoid(line_flux*line_wave*line_wave, line_wave)
                ew = integrate.trapezoid(np.abs(line_flux/contiflux), line_wave)
                area = lambda0
                
                sigma = np.sqrt(lambda2/lambda0-(lambda1/lambda0)**2)/compcenter*c
            else:
                fwhm, sigma, ew, peak, area = 0., 0., 0., 0., 0.
        
        return fwhm, sigma, ew, peak, area


    def comb_line_prop(self, compcenter, pp):
        """
        Calculate the further results for a combined component of emission lines, e.g., FWHM, sigma, peak, line flux.
        The compcenter is the theortical vacuum wavelength for the broad compoenet.
        """
        pp = pp.astype(float)
        ind_br = np.repeat(np.where(pp[2::3] > 0., True, False), 3)
        ind_br[9:] = False  # to exclude the broad OIII and broad He II
        
        p = pp[ind_br]
        del pp
        pp = p
        
        c = 299792.458  # km/s
        n_gauss = int(len(pp)/3)
        if n_gauss == 0:
            fwhm, sigma, ew, peak, area = 0., 0., 0., 0., 0.
        else:
            cen = np.zeros(n_gauss)
            sig = np.zeros(n_gauss)
            
            for i in range(n_gauss):
                cen[i] = pp[3*i+1]
                sig[i] = pp[3*i+2]
            
            # print cen,sig,area
            left = min(cen-3*sig)
            right = max(cen+3*sig)
            disp = 1.e-4*np.log(10.)
            npix = int((right-left)/disp)
            
            xx = np.linspace(left, right, npix)
            yy = self.Manygauss(xx, pp)
        
            # here I directly use the continuum model to avoid the inf bug of EW when the spectrum range passed in is too short
            contiflux = self.conti_fit.params[6]*(np.exp(xx)/self.pl_pivot)**self.conti_fit.params[7]+self.F_poly_conti(
                np.exp(xx), self.conti_fit.params[11:])+self.Balmer_conti(np.exp(xx), self.conti_fit.params[8]) + self.Balmer_high_order(np.exp(xx), self.conti_fit.params[9:11])            
            if self.broken_pl == True:
                f = interpolate.InterpolatedUnivariateSpline(
                    self.wave, 
                    self.f_conti_model)
                contiflux = f(np.exp(xx))
            
            # find the line peak location
            ypeak = yy.max()
            ypeak_ind = np.argmax(yy)
            peak = np.exp(xx[ypeak_ind])
            
            # find the FWHM in km/s
            # take the broad line we focus and ignore other broad components such as [OIII], HeII
            
            if n_gauss > 3:
                spline = interpolate.UnivariateSpline(xx,
                                                      self.Manygauss(xx, pp[0:9])-np.max(self.Manygauss(xx, pp[0:9]))/2,
                                                      s=0)
            else:
                spline = interpolate.UnivariateSpline(xx, yy-np.max(yy)/2, s=0)
            if len(spline.roots()) > 0:
                fwhm_left, fwhm_right = spline.roots().min(), spline.roots().max()
                fwhm = abs(np.exp(fwhm_left)-np.exp(fwhm_right))/compcenter*c
                
                # calculate the line sigma and EW in normal wavelength
                line_flux = self.Manygauss(xx, pp)
                line_wave = np.exp(xx)
                lambda0 = integrate.trapezoid(line_flux, line_wave)  # calculate the total broad line flux
                lambda1 = integrate.trapezoid(line_flux*line_wave, line_wave)
                lambda2 = integrate.trapezoid(line_flux*line_wave*line_wave, line_wave)
                ew = integrate.trapezoid(np.abs(line_flux/contiflux), line_wave)
                area = lambda0
                
                sigma = np.sqrt(lambda2/lambda0-(lambda1/lambda0)**2)/compcenter*c
            else:
                fwhm, sigma, ew, peak, area = 0., 0., 0., 0., 0.
        
        return fwhm, sigma, ew, peak, area

    def CalFWHM(self, logsigma):
        """transfer the logFWHM to normal frame"""
        return 2*np.sqrt(2*np.log(2))*(np.exp(logsigma)-1)*ac.c.to(u.Unit('km/s')).value

    def F_poly_conti(self, xval, pp):
        """Fit the continuum with a polynomial component account for the dust reddening with a*X+b*X^2+c*X^3"""
        xval2 = xval-self.pl_pivot
        yval = 0.*xval2
        for i in range(len(pp)):
            yval = yval+pp[i]*xval2**(i+1)
        return yval
    
    def Onegauss(self, xval, pp):
        """The single Gaussian model used to fit the emission lines 
        Parameter: the scale factor, central wavelength in logwave, line FWHM in logwave
        """
        
        term1 = np.exp(- (xval-pp[1])**2/(2.*pp[2]**2))
        yval = pp[0]*term1/(np.sqrt(2.*np.pi)*pp[2])
        return yval
    
    def Manygauss(self, xval, pp):
        """The multi-Gaussian model used to fit the emission lines, it will call the onegauss function"""
        ngauss = int(pp.shape[0]/3)
        if ngauss != 0:
            yval = 0.
            for i in range(ngauss):
                yval = yval+self.Onegauss(xval, pp[i*3:(i+1)*3])
            return yval
        else:
            return np.zeros_like(xval)

    def Get_Fe_flux(self, ranges, pp=None):
        """Calculate the flux of fitted FeII template within given wavelength ranges.
        ranges: 1-D array, 2-D array
            if 1-D array was given, it should contain two parameters contain a range of wavelength. FeII flux within this range would be calculate and documented in the result fits file.
            if 2-D array was given, it should contain a series of ranges. FeII flux within these ranges would be documented respectively.
        pp: 1-D array with 3 or 6 items.
            If 3 parameters were given, function will choose a proper template (MgII or balmer) according to the range.
            If the range give excess either template, an error would be arose.
            If 6 parameters were given (recommended), function would adopt the first three for the MgII template and the last three for the balmer."""
        if pp is None:
            pp = self.conti_fit.params[:6]
        
        Fe_flux_result = np.array([])
        Fe_flux_type = np.array([])
        Fe_flux_name = np.array([])
        if np.array(ranges).ndim == 1:
            Fe_flux_result = np.append(Fe_flux_result, self._calculate_Fe_flux(ranges, pp))
            Fe_flux_name = np.append(Fe_flux_name, 'Fe_flux_'+str(int(np.min(ranges)))+'_'+str(int(np.max(ranges))))
            Fe_flux_type = np.append(Fe_flux_type, 'float')
        
        elif np.array(ranges).ndim == 2:
            for iii in range(np.array(self.Fe_flux_range).shape[0]):
                Fe_flux_result = np.append(Fe_flux_result, self._calculate_Fe_flux(ranges[iii], pp))
                Fe_flux_name = np.append(Fe_flux_name,
                                         'Fe_flux_'+str(int(np.min(ranges[iii])))+'_'+str(int(np.max(ranges[iii]))))
                Fe_flux_type = np.append(Fe_flux_type, 'float')
        else:
            raise IndexError('The parameter ranges only adopts arrays with 1 or 2 dimensions.')
        
        return Fe_flux_result, Fe_flux_type, Fe_flux_name

    def Smooth(self, y, box_pts):
        "Smooth the flux with n pixels"
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth
    
    def _CalculateSN(self, wave, flux, alter=True):
        """
        Calculate the spectral SN ratio for 1350, 3000, 5100A, return the mean value of Three spots
        This function will automatically check if the 50A vicinity of at the default three wavelength contain more than
        10 pixels. If so, this function will calculate the continuum SN ratio from available regions. If not, it may
        imply that the give spectrum are very low resolution or have frequent gaps in their wavelength coverage. We
        provide another algorithm to calculate the SNR regardless of the continuum.
        
        Parameters
        ----------
        wave: 1-D array
            wavelength of the spectrum
        flux: 1-D array
            flux of the spectrum
        alter: bool
            if True, the function will calculate the SNR regardless of the continuum. If False, the function will return
            -1 if the continuum SNR is not available.
            
        Returns
        -------
        SN_ratio_conti: float
            the SNR of the continuum
        """
        ind5100 = (wave > 5080) & (wave < 5130)
        ind3000 = (wave > 3000) & (wave < 3050)
        ind1350 = (wave > 1325) & (wave < 1375)

        if np.all(np.array([np.sum(ind5100), np.sum(ind3000), np.sum(ind1350)]) < 10):

            if alter is False:
                self.SN_ratio_conti = -1.
                return self.SN_ratio_conti

            # referencing: www.stecf.org/software/ASTROsoft/DER_SNR/
            input_data = np.array(flux)
            # Values that are exactly zero (padded) are skipped
            input_data = np.array(input_data[np.where(input_data != 0.0)])
            n = len(input_data)
            # For spectra shorter than this, no value can be returned
            if (n > 4):
                signal = np.median(input_data)
                noise = 0.6052697 * np.median(np.abs(2.0 * input_data[2:n - 2] - input_data[0:n - 4] - input_data[4:n]))
                self.SN_ratio_conti = float(signal / noise)
            else:
                self.SN_ratio_conti = -1.

        else:
            tmp_SN = np.array([flux[ind5100].mean() / flux[ind5100].std() if np.sum(ind5100) > 0 else np.nan,
                               flux[ind3000].mean() / flux[ind3000].std() if np.sum(ind3000) > 0 else np.nan,
                               flux[ind1350].mean() / flux[ind1350].std() if np.sum(ind1350) > 0 else np.nan])
            tmp_SN = tmp_SN[np.array([np.sum(ind5100), np.sum(ind3000), np.sum(ind1350)]) > 10]
            if not np.all(np.isnan(tmp_SN)):
                self.SN_ratio_conti = np.nanmean(tmp_SN)
            else:
                self.SN_ratio_conti = -1.

        return self.SN_ratio_conti
    
    def _DoDecomposition(self, wave, flux, err, path):
        """Decompose the host galaxy from QSO"""
        datacube = self._HostDecompose(self.wave, self.flux, self.err, self.z, self.Mi, self.npca_gal, self.npca_qso,
                                       path)
        
        # for some negtive host templete, we do not do the decomposition
        if np.sum(np.where(datacube[3, :] < 0., True, False)) > 100:
            self.host = np.zeros(len(wave))
            self.decomposed = False
            print('Get negtive host galaxy flux larger than 100 pixels, '
                  'decomposition is not applied!')
        else:
            self.decomposed = True
            del self.wave, self.flux, self.err
            self.wave = datacube[0, :]
            # block OIII, ha,NII,SII,OII,Ha,Hb,Hr,hdelta
            
            line_mask = np.where(
                (self.wave < 4970.) & (self.wave > 4950.) | (self.wave < 5020.) & (self.wave > 5000.) | (
                        self.wave < 6590.) & (self.wave > 6540.) | (self.wave < 6740.) & (self.wave > 6710.) | (
                        self.wave < 3737.) & (self.wave > 3717.) | (self.wave < 4872.) & (self.wave > 4852.) | (
                        self.wave < 4350.) & (self.wave > 4330.) | (self.wave < 4111.) & (self.wave > 4091.), True,
                False)
            
            f = interpolate.interp1d(self.wave[~line_mask], datacube[3, :][~line_mask], bounds_error=False,
                                     fill_value=0)
            masked_host = f(self.wave)
            self.masked_host = masked_host
            self.flux = datacube[1, :]-masked_host  # QSO flux without host
            self.err = datacube[2, :]
            self.host = datacube[3, :]
            self.qso = datacube[4, :]
            self.host_data = datacube[1, :]-self.qso
        return self.wave, self.flux, self.err

    def _HostDecompose(self, wave, flux, err, z, Mi, npca_gal, npca_qso, path):
        path = datapath
        """
        core function to do host decomposition
        #Wave is the obs frame wavelength, n_gal and n_qso are the number of eigenspectra used to fit
        #If Mi is None then the qso use the globle ones to fit. If not then use the redshift-luminoisty binded ones to fit
        #See details: 
        #Yip, C. W., Connolly, A. J., Szalay, A. S., et al. 2004a, AJ, 128, 585
        #Yip, C. W., Connolly, A. J., Vanden Berk, D. E., et al. 2004b, AJ, 128, 2603
        """
        
        # read galaxy and qso eigenspectra -----------------------------------
        if self.BC03 == False:
            galaxy = fits.open(os.path.join(path, 'pca', 'Yip_pca_templates', 'gal_eigenspec_Yip2004.fits'))
            gal = galaxy[1].data
            wave_gal = gal['wave'].flatten()
            flux_gal = gal['pca'].reshape(gal['pca'].shape[1], gal['pca'].shape[2])
        if self.BC03 == True:
            cc = 0
            flux03 = np.array([])
            for i in glob.glob(os.path.join(path, 'bc03', '*.gz')):
                cc = cc+1
                gal_temp = np.genfromtxt(i)
                wave_gal = gal_temp[:, 0]
                flux03 = np.concatenate((flux03, gal_temp[:, 1]))
            flux_gal = np.array(flux03).reshape(cc, -1)
        
        if Mi is None:
            quasar = fits.open(os.path.join(path, 'pca', 'Yip_pca_templates', 'qso_eigenspec_Yip2004_global.fits'))
        else:
            if -24 < Mi <= -22 and 0.08 <= z < 0.53:
                quasar = fits.open(os.path.join(path, 'pca', 'Yip_pca_templates', 'qso_eigenspec_Yip2004_CZBIN1.fits'))
            elif -26 < Mi <= -24 and 0.08 <= z < 0.53:
                quasar = fits.open(os.path.join(path, 'pca', 'Yip_pca_templates', 'qso_eigenspec_Yip2004_DZBIN1.fits'))
            elif -24 < Mi <= -22 and 0.53 <= z < 1.16:
                quasar = fits.open(os.path.join(path, 'pca', 'Yip_pca_templates', 'qso_eigenspec_Yip2004_BZBIN2.fits'))
            elif -26 < Mi <= -24 and 0.53 <= z < 1.16:
                quasar = fits.open(os.path.join(path, 'pca', 'Yip_pca_templates', 'qso_eigenspec_Yip2004_CZBIN2.fits'))
            elif -28 < Mi <= -26 and 0.53 <= z < 1.16:
                quasar = fits.open(os.path.join(path, 'pca', 'Yip_pca_templates', 'qso_eigenspec_Yip2004_DZBIN2.fits'))
            else:
                raise RuntimeError('Host galaxy template is not available for this redshift and Magnitude!')
        
        qso = quasar[1].data
        wave_qso = qso['wave'].flatten()
        flux_qso = qso['pca'].reshape(qso['pca'].shape[1], qso['pca'].shape[2])
        
        # get the shortest wavelength range
        wave_min = min(wave.min(), wave_gal.min(), wave_qso.min())
        wave_max = max(wave.max(), wave_gal.max(), wave_qso.max())
        
        ind_data = np.where((wave > wave_min) & (wave < wave_max), True, False)
        ind_gal = np.where((wave_gal > wave_min-1.) & (wave_gal < wave_max+1.), True, False)
        ind_qso = np.where((wave_qso > wave_min-1.) & (wave_qso < wave_max+1.), True, False)
        
        flux_gal_new = np.zeros(flux_gal.shape[0]*flux[ind_data].shape[0]).reshape(flux_gal.shape[0],
                                                                                   flux[ind_data].shape[0])
        flux_qso_new = np.zeros(flux_qso.shape[0]*flux[ind_data].shape[0]).reshape(flux_qso.shape[0],
                                                                                   flux[ind_data].shape[0])
        for i in range(flux_gal.shape[0]):
            fgal = interpolate.interp1d(wave_gal[ind_gal], flux_gal[i, ind_gal], bounds_error=False, fill_value=0)
            flux_gal_new[i, :] = fgal(wave[ind_data])
        for i in range(flux_qso.shape[0]):
            fqso = interpolate.interp1d(wave_qso[ind_qso], flux_qso[i, ind_qso], bounds_error=False, fill_value=0)
            flux_qso_new[i, :] = fqso(wave[ind_data])
        
        wave_new = wave[ind_data]
        flux_new = flux[ind_data]
        err_new = err[ind_data]
        
        flux_temp = np.vstack((flux_gal_new[0:npca_gal, :], flux_qso_new[0:npca_qso, :]))
        res = np.linalg.lstsq(flux_temp.T, flux_new, rcond=None)[0]
        
        host_flux = np.dot(res[0:npca_gal], flux_temp[0:npca_gal])
        qso_flux = np.dot(res[npca_gal:], flux_temp[npca_gal:])
        
        data_cube = np.vstack((wave_new, flux_new, err_new, host_flux, qso_flux))
        
        ind_f4200 = np.where((wave_new > 4160.) & (wave_new < 4210.), True, False)
        frac_host_4200 = np.sum(host_flux[ind_f4200])/np.sum(flux_new[ind_f4200])
        ind_f5100 = np.where((wave_new > 5080.) & (wave_new < 5130.), True, False)
        frac_host_5100 = np.sum(host_flux[ind_f5100])/np.sum(flux_new[ind_f5100])
        
        return data_cube  # ,frac_host_4200,frac_host_5100
    
    def _MaskSdssAndOr(self, lam, flux, err, and_mask, or_mask):
        """
        Remove SDSS and_mask and or_mask points are not zero
        Parameter:
        ----------
        lam: wavelength
        flux: flux
        err: 1 sigma error
        and_mask: SDSS flag "and_mask", mask out all non-zero pixels
        or_mask: SDSS flag "or_mask", mask out all npn-zero pixels
        
        Retrun:
        ---------
        return the same size array of wavelength, flux, error
        """
        ind_and_or = np.where((and_mask == 0) & (or_mask == 0), True, False)
        del self.lam, self.flux, self.err
        self.lam, self.flux, self.err = lam[ind_and_or], flux[ind_and_or], err[ind_and_or]
    
    def _RejectBadPix(self, lam, flux, err):
        """
        Reject 10 most possiable outliers, input wavelength, flux and error. Return a different size wavelength,
        flux, and error.
        """
        # -----remove bad pixels, but not for high SN spectrum------------
        # Alternative outlier detection using modified z-score method
        median_flux = np.nanmedian(flux)
        mad = np.nanmedian(np.abs(flux - median_flux))
        modified_z_scores = 0.6745 * (flux - median_flux) / mad
        outlier_threshold = 3.5
        outlier_indices = np.where(np.abs(modified_z_scores) > outlier_threshold)[0]
        
        # Limit to 10 most extreme outliers (similar to original GESD behavior)
        if len(outlier_indices) > 10:
            outlier_scores = np.abs(modified_z_scores[outlier_indices])
            top_outliers = np.argsort(outlier_scores)[-10:]
            outlier_indices = outlier_indices[top_outliers]
        
        ind_bad = ([], outlier_indices)
        wv = np.asarray([i for j, i in enumerate(lam) if j not in ind_bad[1]], dtype=np.float64)
        fx = np.asarray([i for j, i in enumerate(flux) if j not in ind_bad[1]], dtype=np.float64)
        er = np.asarray([i for j, i in enumerate(err) if j not in ind_bad[1]], dtype=np.float64)
        del self.lam, self.flux, self.err
        self.lam, self.flux, self.err = wv, fx, er
        return self.lam, self.flux, self.err
    
    def _WaveTrim(self, lam, flux, err, z):
        """
        Trim spectrum with a range in the rest frame. 
        """
        # trim spectrum e.g., local fit emiision lines
        ind_trim = np.where((lam/(1.+z) > self.wave_range[0]) & (lam/(1.+z) < self.wave_range[1]), True, False)
        del self.lam, self.flux, self.err
        self.lam, self.flux, self.err = lam[ind_trim], flux[ind_trim], err[ind_trim]
        if len(self.lam) < 100:
            raise RuntimeError("No enough pixels in the input wave_range!")
        return self.lam, self.flux, self.err
    
    def _WaveMsk(self, lam, flux, err, z):
        """Block the bad pixels or absorption lines in spectrum."""
        
        for msk in range(len(self.wave_mask)):
            try:
                ind_not_mask = ~np.where((lam/(1.+z) > self.wave_mask[msk, 0]) & (lam/(1.+z) < self.wave_mask[msk, 1]),
                                         True, False)
            except IndexError:
                raise RuntimeError("Wave_mask should be 2D array,e.g., np.array([[2000,3000],[3100,4000]]).")
            
            del self.lam, self.flux, self.err
            self.lam, self.flux, self.err = lam[ind_not_mask], flux[ind_not_mask], err[ind_not_mask]
            lam, flux, err = self.lam, self.flux, self.err
        return self.lam, self.flux, self.err
    
    def _RestFrame(self, lam, flux, err, z):
        """Move wavelenth and flux to rest frame"""
        self.wave = lam/(1.+z)
        self.flux = flux*(1.+z)
        self.err = err*(1.+z)
        return self.wave, self.flux, self.err
    
    def _OrignialSpec(self, wave, flux, err):
        """save the orignial spectrum before host galaxy decompsition"""
        self.wave_prereduced = wave
        self.flux_prereduced = flux
        self.err_prereduced = err

    def _calculate_Fe_flux(self, range, pp):
        """Calculate the flux of fitted FeII template within one given wavelength range.
        The pp could be an array with a length of 3 or 6. If 3 parameters were give, function will choose a
        proper template (MgII or balmer) according to the range. If the range give excess both template, an
        error would be arose. If 6 parameters were give, function would adopt the first three for the MgII
        template and the last three for the balmer."""
        
        balmer_range = np.array([3686., 7484.])
        mgii_range = np.array([1200., 3500.])
        upper = np.min([np.max(range), np.max(self.wave)])
        lower = np.max([np.min(range), np.min(self.wave)])
        if upper < np.max(range) or lower > np.min(range):
            print('Warning: The range given to calculate the flux of FeII pseudocontiuum (partially) exceeded '
                  'the boundary of spectrum wavelength range. The excess part would be set to zero!')
        disp = 1.e-4*np.log(10.)
        xval = np.exp(np.arange(np.log(lower), np.log(upper), disp))
        if len(xval) < 10:
            print('Warning: Available part in range '+str(range)+' is less than 10 pixel. Flux = -999 would be given!')
            return -999
        
        if len(pp) == 3:
            if upper <= mgii_range[1] and lower >= mgii_range[0]:
                yval = self.Fe_flux_mgii(xval, pp)
            elif upper <= balmer_range[1] and lower >= balmer_range[0]:
                yval = self.Fe_flux_balmer(xval, pp)
            else:
                raise OverflowError('Only 3 parameters are given in this function. \
                Make sure the range is within [1200., 3500.] or [3686., 7484.]!')
        
        elif len(pp) == 6:
            yval = self.Fe_flux_mgii(xval, pp[:3])+self.Fe_flux_balmer(xval, pp[3:])
            if upper > balmer_range[1] or lower < mgii_range[0]:
                print('Warning: The range given to calculate the flux of FeII pseudocontiuum (partially) '
                      'exceeded the template range [1200., 7478.]. The excess part would be set to zero!')
            elif upper > mgii_range[1] and lower < balmer_range[0]:
                print('Warning: The range given to calculate the flux of FeII pseudocontiuum (partially) '
                      'contained range [3500., 3686.] which is the gap between FeII templates and would be set to zero!')
        
        else:
            raise IndexError('The parameter pp only adopts a list of 3 or 6.')
        
        flux = integrate.trapezoid(yval[(xval >= lower) & (xval <= upper)], xval[(xval >= lower) & (xval <= upper)])
        return flux
    
    def _do_line_kmpfit(self, linelist, line_flux, ind_line, ind_n, nline_fit, ngauss_fit):
        """The key function to do the line fit with kmpfit"""
        line_fit = kmpfit.Fitter(self._residuals_line, data=(
            np.log(self.wave[ind_n]), line_flux[ind_n], self.err[ind_n]))  # fitting wavelength in ln space
        line_fit_ini = np.array([])
        line_fit_par = np.array([])
        for n in range(nline_fit):
            for nn in range(ngauss_fit[n]):
                # set up initial parameter guess
                line_fit_ini0 = [0., np.log(linelist['lambda'][ind_line][n]), linelist['inisig'][ind_line][n]]
                line_fit_ini = np.concatenate([line_fit_ini, line_fit_ini0])
                # set up parameter limits
                lambda_low = np.log(linelist['lambda'][ind_line][n])-linelist['voff'][ind_line][n]
                lambda_up = np.log(linelist['lambda'][ind_line][n])+linelist['voff'][ind_line][n]
                sig_low = linelist['minsig'][ind_line][n]
                sig_up = linelist['maxsig'][ind_line][n]
                line_fit_par0 = [{'limits': (0., 10.**10)}, {'limits': (lambda_low, lambda_up)},
                                 {'limits': (sig_low, sig_up)}]
                line_fit_par = np.concatenate([line_fit_par, line_fit_par0])
        
        line_fit.parinfo = line_fit_par
        line_fit.fit(params0=line_fit_ini)
        line_fit.params = self.newpp
        self.line_fit = line_fit
        self.line_fit_ini = line_fit_ini
        self.line_fit_par = line_fit_par
        return line_fit
    
    def _do_tie_line(self, linelist, ind_line):
        """Tie line's central"""
        # --------------- tie parameter-----------
        # so far, only two groups of each properties are support for tying
        ind_tie_v1 = np.where(linelist['vindex'][ind_line] == 1., True, False)
        ind_tie_v2 = np.where(linelist['vindex'][ind_line] == 2., True, False)
        ind_tie_w1 = np.where(linelist['windex'][ind_line] == 1., True, False)
        ind_tie_w2 = np.where(linelist['windex'][ind_line] == 2., True, False)
        ind_tie_f1 = np.where(linelist['findex'][ind_line] == 1., True, False)
        ind_tie_f2 = np.where(linelist['findex'][ind_line] == 2., True, False)
        
        ind_tie_vindex1 = np.array([])
        ind_tie_vindex2 = np.array([])
        ind_tie_windex1 = np.array([])
        ind_tie_windex2 = np.array([])
        ind_tie_findex1 = np.array([])
        ind_tie_findex2 = np.array([])
        
        # get index of vindex windex in initial parameters
        for iii in range(len(ind_tie_v1)):
            if ind_tie_v1[iii] == True:
                ind_tie_vindex1 = np.concatenate(
                    [ind_tie_vindex1, np.array([int(np.sum(linelist['ngauss'][ind_line][0:iii])*3+1)])])
        if np.any(ind_tie_v1):
            self.delta_lambda1 = (np.log(linelist['lambda'][ind_line][ind_tie_v1])-np.log(
                linelist['lambda'][ind_line][ind_tie_v1][0]))[1:]
        else:
            self.delta_lambda1 = np.array([])
        self.ind_tie_vindex1 = ind_tie_vindex1
        
        for iii in range(len(ind_tie_v2)):
            if ind_tie_v2[iii] == True:
                ind_tie_vindex2 = np.concatenate(
                    [ind_tie_vindex2, np.array([int(np.sum(linelist['ngauss'][ind_line][0:iii])*3+1)])])
        if np.any(ind_tie_v2):
            self.delta_lambda2 = (np.log(linelist['lambda'][ind_line][ind_tie_v2])-np.log(
                linelist['lambda'][ind_line][ind_tie_v2][0]))[1:]
        else:
            self.delta_lambda2 = np.array([])
        self.ind_tie_vindex2 = ind_tie_vindex2
        
        for iii in range(len(ind_tie_w1)):
            if ind_tie_w1[iii] == True:
                ind_tie_windex1 = np.concatenate(
                    [ind_tie_windex1, np.array([int(np.sum(linelist['ngauss'][ind_line][0:iii])*3+2)])])
        self.ind_tie_windex1 = ind_tie_windex1
        
        for iii in range(len(ind_tie_w2)):
            if ind_tie_w2[iii] == True:
                ind_tie_windex2 = np.concatenate(
                    [ind_tie_windex2, np.array([int(np.sum(linelist['ngauss'][ind_line][0:iii])*3+2)])])
        self.ind_tie_windex2 = ind_tie_windex2
        
        # get index of findex for 1&2 case in initial parameters
        for iii_1 in range(len(ind_tie_f1)):
            if ind_tie_f1[iii_1] == True:
                ind_tie_findex1 = np.concatenate(
                    [ind_tie_findex1, np.array([int(np.sum(linelist['ngauss'][ind_line][0:iii_1])*3)])])
        
        for iii_2 in range(len(ind_tie_f2)):
            if ind_tie_f2[iii_2] == True:
                ind_tie_findex2 = np.concatenate(
                    [ind_tie_findex2, np.array([int(np.sum(linelist['ngauss'][ind_line][0:iii_2])*3)])])
        
        # get tied fvalue for case 1 and case 2
        if np.sum(ind_tie_f1) > 0:
            self.fvalue_factor_1 = linelist['fvalue'][ind_line][ind_tie_f1][1]/linelist['fvalue'][ind_line][ind_tie_f1][
                0]
        else:
            self.fvalue_factor_1 = np.array([])
        if np.sum(ind_tie_f2) > 0:
            self.fvalue_factor_2 = linelist['fvalue'][ind_line][ind_tie_f2][1]/linelist['fvalue'][ind_line][ind_tie_f2][
                0]
        else:
            self.fvalue_factor_2 = np.array([])
        
        self.ind_tie_findex1 = ind_tie_findex1
        self.ind_tie_findex2 = ind_tie_findex2
    
    # ---------MC error for emission line parameters-------------------
    def _line_mc(self, x, y, err, pp0, pp_limits, n_trails, compcenter):
        """calculate the Monte Carlo errror of line parameters"""
        all_para_1comp = np.zeros(len(pp0)*n_trails).reshape(len(pp0), n_trails)
        all_para_std = np.zeros(len(pp0))
        all_fwhm = np.zeros(n_trails)
        all_sigma = np.zeros(n_trails)
        all_ew = np.zeros(n_trails)
        all_peak = np.zeros(n_trails)
        all_area = np.zeros(n_trails)
        
        for tra in range(n_trails):
            flux = y+np.random.randn(len(y))*err
            line_fit = kmpfit.Fitter(residuals=self._residuals_line, data=(x, flux, err), maxiter=50)
            line_fit.parinfo = pp_limits
            line_fit.fit(params0=pp0)
            line_fit.params = self.newpp
            all_para_1comp[:, tra] = line_fit.params
            
            # further line properties
            all_fwhm[tra], all_sigma[tra], all_ew[tra], all_peak[tra], all_area[tra] = self.line_prop(compcenter,
                                                                                                      line_fit.params,
                                                                                                      'broad')
        
        for st in range(len(pp0)):
            all_para_std[st] = all_para_1comp[st, :].std()
        
        return all_para_std, all_fwhm.std(), all_sigma.std(), all_ew.std(), all_peak.std(), all_area.std()

    def _residuals_line(self, pp, data):
        "The line residual function used in kmpfit"
        xval, yval, weight = data
        
        # ------tie parameter------------
        if self.tie_lambda == True:
            if len(self.ind_tie_vindex1) > 1:
                for xx in range(len(self.ind_tie_vindex1)-1):
                    pp[int(self.ind_tie_vindex1[xx+1])] = pp[int(self.ind_tie_vindex1[0])]+self.delta_lambda1[xx]
            
            if len(self.ind_tie_vindex2) > 1:
                for xx in range(len(self.ind_tie_vindex2)-1):
                    pp[int(self.ind_tie_vindex2[xx+1])] = pp[int(self.ind_tie_vindex2[0])]+self.delta_lambda2[xx]
        
        if self.tie_width == True:
            if len(self.ind_tie_windex1) > 1:
                for xx in range(len(self.ind_tie_windex1)-1):
                    pp[int(self.ind_tie_windex1[xx+1])] = pp[int(self.ind_tie_windex1[0])]
            
            if len(self.ind_tie_windex2) > 1:
                for xx in range(len(self.ind_tie_windex2)-1):
                    pp[int(self.ind_tie_windex2[xx+1])] = pp[int(self.ind_tie_windex2[0])]
        
        if len(self.ind_tie_findex1) > 0 and self.tie_flux_1 == True:
            pp[int(self.ind_tie_findex1[1])] = pp[int(self.ind_tie_findex1[0])]*self.fvalue_factor_1
        if len(self.ind_tie_findex2) > 0 and self.tie_flux_2 == True:
            pp[int(self.ind_tie_findex2[1])] = pp[int(self.ind_tie_findex2[0])]*self.fvalue_factor_2
        # ---------------------------------
        
        # restore parameters
        self.newpp = pp.copy()
        return (yval-self.Manygauss(xval, pp))/weight
    
    def _residuals(self, pp, data):
        """Continual residual function used in kmpfit"""
        xval, yval, weight = data
        return (yval-self._f_conti_all(xval, pp))/weight
