#!/usr/bin/env python
# from os import name
import sys
import glob
import warnings
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sfdmap
from scipy import interpolate
from scipy import integrate
from kapteyn import kmpfit
from PyAstronomy import pyasl
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
# from astropy.modeling.blackbody import blackbody_lambda
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy import constants as ac
from PyQSOFit import QSOFit
from .extinction import *
from .auxmodule import *
import pkg_resources
import pandas as pd
import astropy
from packaging import version
if version.parse(astropy.__version__) < version.parse("4.3.0"):
    from astropy.modeling.blackbody import blackbody_lambda
else:
    from astropy.modeling.models import BlackBody


datapath = pkg_resources.resource_filename('PyQSOFit', '/')
dustmap_path = pkg_resources.resource_filename('PyQSOFit', '/sfddata/')
new_datapath = pkg_resources.resource_filename('qsofitmore', '/ext_data/')

__all__ = ['QSOFitNew']

getnonzeroarr = lambda x: x[x != 0]
sciplotstyle()

class QSOFitNew(QSOFit):

    def __init__(self, lam, flux, err, z, ra=- 999., dec=-999., name=None, plateid=None, mjd=None, fiberid=None, 
                 path=None, and_mask=None, or_mask=None, is_sdss=True):
        """
        Get the input data perpared for the QSO spectral fitting
        
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
            If the source is SDSS object, they have the plate ID, MJD and Fiber ID in their file herader.
            
        path: str
            the path of the input data
            
        and_mask, or_mask: 1-D array with Npix, optional
            the bad pixels defined from SDSS data, which can be got from SDSS datacube.
        """
        
        self.lam = np.asarray(lam, dtype=np.float64)
        self.flux = np.asarray(flux, dtype=np.float64)
        self.err = np.asarray(err, dtype=np.float64)
        self.sn_obs = self.flux/self.err
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


    @classmethod
    def fromsdss(cls, fname, redshift=None, path=None, plateid=None, mjd=None, fiberid=None, 
                 ra=None, dec=None, telescope=None):
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
                   mjd=mjd, fiberid=fiberid, path=path, is_sdss=True)


    @classmethod
    def fromiraf(cls, fname, redshift=None, path=None, plateid=None, mjd=None, fiberid=None, 
                 ra=None, dec=None, telescope=None):
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
                   mjd=mjd, fiberid=fiberid, path=path, is_sdss=False)

    @classmethod
    def fromcomb1d(cls, fname, redshift=None, path=None, plateid=None, mjd=None, fiberid=None, 
                 ra=None, dec=None, telescope=None):
        """
        Initialize QSOFit object from a combined spectra
        with two extensions, the first and second 
        extensions of which are flux and flux errors
        respectively.
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
        q = QSOFitNew.fromcomb1d("custom_iraf_spectrum.fits", redshift=0.01, path=path)
        """
        hdu = fits.open(fname)
        header = hdu[0].header
        objname = header['object']
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
        try:
            CD1_1 = float(header['CD1_1'])
        except:
            CD1_1 = float(header['CDELT1'])
        CRPIX1 = float(header['CRPIX1'])
        W1 = (1-CRPIX1) * CD1_1 + CRVAL1
        data = hdu[0].data
        num_pt = len(data)
        wave = np.linspace(W1, 
                           W1 + (num_pt - 1) * CD1_1, 
                           num=num_pt)
        flux = data
        err = hdu[1].data
        hdu.close() 
        flux *= 1e17
        err *= 1e17
        return cls(lam=wave, flux=flux, err=err, z=redshift, ra=ra, dec=dec, name=name, plateid=plateid, 
                   mjd=mjd, fiberid=fiberid, path=path, is_sdss=False)

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
            m = sfdmap.SFDMap(dustmap_path)
            zero_flux = np.where(flux == 0, True, False)
            flux[zero_flux] = 1e-10
            flux_unred = pyasl.unred(lam, flux, m.ebv(ra, dec))
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


    def _HostDecompose(self, wave, flux, err, z, Mi, npca_gal, npca_qso, path):
        path = datapath
        return super()._HostDecompose(wave, flux, err, z, Mi, npca_gal, npca_qso, path)

    
    def _DoContiFit(self, wave, flux, err, ra, dec, plateid, mjd, fiberid):
        """Fit the continuum with PL, Polynomial, UV/optical FeII, Balmer continuum"""
        if self.plateid is None:
            plateid = 0
        if self.plateid is None:
            mjd = 0
        if self.plateid is None:
            fiberid = 0
        # tmp_selfpath = self.path
        # self.path = datapath
        # global fe_uv, fe_op
        self.fe_uv = np.genfromtxt(datapath+'fe_uv.txt')
        self.fe_op = np.genfromtxt(datapath+'fe_optical.txt')
        self.fe_verner = np.genfromtxt(new_datapath+'Fe_Verner.txt')
        if self.BC == True:
            try:
                print("N_e = 1E{}.".format(self.ne))
            except AttributeError:
                print('N_e for Balmer line series not set.\nSetting N_e = 1E09. (q.set_log10_electron_density(9))')
                ne = 9
                self.ne = ne
            if self.ne == 9:
                balmer_file = new_datapath+'balmer_n6_n50_em_NE09.csv'
            elif self.ne == 10:
                balmer_file = new_datapath+'balmer_n6_n50_em_NE10.csv'
            self.df_balmer_series = pd.read_csv(balmer_file)
        else:
            self.df_balmer_series = pd.read_csv(new_datapath+'balmer_n6_n50_em_NE09.csv')
        
        # do continuum fit--------------------------
        window_all = np.array(
            [[1150., 1170.], [1275., 1290.], [1350., 1360.], [1445., 1465.], [1690., 1705.], [1770., 1810.],
             [1970., 2400.], [2480., 2675.], [2925., 3400.], [3500., 3600.], [3600., 4260.],
            #  [3775., 3832.], [3833., 3860.], [3890., 3960.],
            #  [4000., 4090.], [4115., 4260.],
             [4435., 4640.], [5100., 5535.], [6005., 6035.], [6110., 6250.], [6800., 7000.], [7160., 7180.],
             [7500., 7800.], [8050., 8150.], ])
        
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
            pp0 = np.array([0., 3000., 0., 0., 3000., 0., 1., -1.5, 0., 5e3, 0., 0., 0., 0.])
        if self.broken_pl == True:
            pp0 = np.array([0., 3000., 0., 0., 3000., 0., 1., -1.5, 0., 5e3, 0., 0., 0., 0., -0.35])
        conti_fit = kmpfit.Fitter(residuals=self._residuals, data=(wave[tmp_all], flux[tmp_all], err[tmp_all]))
        tmp_parinfo = [{'limits': (0., 10.**10)}, {'limits': (1200., 10000.)}, {'limits': (-0.01, 0.01)},
                       {'limits': (0., 10.**10)}, {'limits': (1200., 10000.)}, {'limits': (-0.01, 0.01)},
                       {'limits': (0., 10.**10)}, {'limits': (-5., 3.)}, 
                       {'limits': (0., 10.**10)}, {'limits': (2e3, 9e3)}, {'limits': (0., 10.**10)}, 
                       None, None, None, ]
        if self.broken_pl == True:
            conti_fit = kmpfit.Fitter(residuals=self._residuals, data=(wave[tmp_all], flux[tmp_all], err[tmp_all]))
            tmp_parinfo = [{'limits': (0., 10.**10)}, {'limits': (1200., 10000.)}, {'limits': (-0.01, 0.01)},
                           {'limits': (0., 10.**10)}, {'limits': (1200., 10000.)}, {'limits': (-0.01, 0.01)},
                           {'limits': (0., 10.**10)}, {'limits': (-5., 3.)}, 
                           {'limits': (0., 10.**10)}, {'limits': (2e3, 9e3)}, {'limits': (0., 10.**10)}, 
                           None, None, None, 
                           {'limits': (-5., 3.)},]
        conti_fit.parinfo = tmp_parinfo
        conti_fit.fit(params0=pp0)
        
        # Perform one iteration to remove 3sigma pixel below the first continuum fit
        # to avoid the continuum windows falls within a BAL trough
        if self.rej_abs == True:
            if self.poly == True and self.broken_pl == False:
                tmp_conti = (conti_fit.params[6]*(wave[tmp_all]/3000.0)**conti_fit.params[7]+self.F_poly_conti(
                    wave[tmp_all], conti_fit.params[11:14]))
            elif self.poly ==False and self.broken_pl == False:
                tmp_conti = (conti_fit.params[6]*(wave[tmp_all]/3000.0)**conti_fit.params[7])
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
            f = kmpfit.Fitter(residuals=self._residuals, data=(
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
            # calculate MC err
            conti_para_std, all_L_std = self._conti_mc(self.wave[tmp_all], self.flux[tmp_all],
                                                       self.err[tmp_all], pp0, conti_fit.parinfo,
                                                       self.n_trails)
            
            self.conti_result = np.array(
                [ra, dec, str(plateid), str(mjd), str(fiberid), self.z, SNR_SPEC, self.SN_ratio_conti, conti_fit.params[0],
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
                 'BalmerS_norm_err', 'POLY_a', 'POLY_a_err', 'POLY_b', 'POLY_b_err', 'POLY_c', 'POLY_c_err', 'L1350',
                 'L1350_err', 'L3000', 'L3000_err', 'L5100', 'L5100_err'])
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
            # for iii in range(Fe_flux_result.shape[0]):
            #     self.conti_result = np.append(self.conti_result, [Fe_flux_result[iii], Fe_flux_std[iii]])
            #     self.conti_result_type = np.append(self.conti_result_type, [Fe_flux_type[iii], Fe_flux_type[iii]])
            #     self.conti_result_name = np.append(self.conti_result_name,
            #                                        [Fe_flux_name[iii], Fe_flux_name[iii]+'_err'])
        else:
            self.conti_result = np.array(
                [ra, dec, str(plateid), str(mjd), str(fiberid), self.z, SNR_SPEC, self.SN_ratio_conti, conti_fit.params[0],
                 conti_fit.params[1], conti_fit.params[2], conti_fit.params[3], conti_fit.params[4],
                 conti_fit.params[5], conti_fit.params[6], conti_fit.params[7], conti_fit.params[8],
                 conti_fit.params[9], conti_fit.params[10], conti_fit.params[11], conti_fit.params[12],
                 conti_fit.params[13], L[0], L[1], L[2]])
            self.conti_result_type = np.array(
                ['float', 'float', 'int', 'int', 'int', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float',
                 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float',
                 'float'])
            self.conti_result_name = np.array(
                ['ra', 'dec', 'plateid', 'MJD', 'fiberid', 'redshift', 'SNR_SPEC', 'SN_ratio_conti', 'Fe_uv_norm', 'Fe_uv_FWHM',
                 'Fe_uv_shift', 'Fe_op_norm', 'Fe_op_FWHM', 'Fe_op_shift', 'PL_norm', 'PL_slope', 'BalmerC_norm',
                 'BalmerS_FWHM', 'BalmerS_norm', 'POLY_a', 'POLY_b', 'POLY_c', 'L1350', 'L3000', 'L5100'])
            if self.broken_pl == True:
                self.conti_result = np.concatenate((self.conti_result, 
                                                    conti_fit.params[14]), axis=None)
                self.conti_result_type = np.concatenate((self.conti_result_type, 
                                                         'float'), axis=None)
                self.conti_result_name = np.concatenate((self.conti_result_name, 
                                                         'PL_slope2'), axis=None)
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
        f_pl_model = conti_fit.params[6]*(wave/3000.0)**conti_fit.params[7]
        if self.broken_pl == True:
            f_pl_model = broken_pl_model(wave, conti_fit.params[7], conti_fit.params[14], conti_fit.params[6])
        f_bc_model = self.Balmer_conti(wave, conti_fit.params[8]) + self.Balmer_high_order(wave, conti_fit.params[9:11])
        f_poly_model = self.F_poly_conti(wave, conti_fit.params[11:14])
        if self.Fe_verner09 == True:
            f_conti_model = (f_pl_model+f_fe_verner_model+f_poly_model+f_bc_model)
        else:
            f_conti_model = (f_pl_model+f_fe_mgii_model+f_fe_balmer_model+f_poly_model+f_bc_model)
        line_flux = flux-f_conti_model

        self.f_conti_model = f_conti_model
        self.f_bc_model = f_bc_model
        self.f_fe_uv_model = f_fe_mgii_model
        self.f_fe_op_model = f_fe_balmer_model
        self.f_fe_verner_model = f_fe_verner_model
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
        # power-law continuum
        f_pl = pp[6]*(xval/3000.0)**pp[7]
        # Balmer continuum
        f_conti_BC = self.Balmer_conti(xval, pp[8]) + self.Balmer_high_order(xval, pp[9:11])
        # polynormal conponent for reddened spectra
        f_poly = self.F_poly_conti(xval, pp[11:14])
        if self.broken_pl == True:
            f_pl = broken_pl_model(xval, pp[7], pp[14], pp[6])
        
        if self.Fe_uv_op == True and self.Fe_verner09 == False:
            f_Fe_all = f_Fe_MgII+f_Fe_Balmer
        elif self.Fe_uv_op == True and self.Fe_verner09 == True:
            f_Fe_all = f_Fe_verner
        
        if self.Fe_uv_op == True and self.poly == False and self.BC == False:
            yval = f_pl+f_Fe_all
        elif self.Fe_uv_op == True and self.poly == True and self.BC == False:
            yval = f_pl+f_Fe_all+f_poly
        elif self.Fe_uv_op == True and self.poly == False and self.BC == True:
            yval = f_pl+f_Fe_all+f_conti_BC
        elif self.Fe_uv_op == False and self.poly == True and self.BC == False:
            yval = f_pl+f_poly
        elif self.Fe_uv_op == False and self.poly == False and self.BC == False:
            yval = f_pl
        elif self.Fe_uv_op == False and self.poly == False and self.BC == True:
            yval = f_pl+f_conti_BC
        elif self.Fe_uv_op == True and self.poly == True and self.BC == True:
            yval = f_pl+f_Fe_all+f_poly+f_conti_BC
        elif self.Fe_uv_op == False and self.poly == True and self.BC == True:
            yval = f_pl+f_Fe_Balmer+f_poly+f_conti_BC
        else:
            raise RuntimeError('No this option for Fe_uv_op, poly and BC!')
        return yval


    def _L_conti(self, wave, pp):
        """Calculate continuum Luminoisity at 1350,3000,5100A"""
        conti_flux = pp[6]*(wave/3000.0)**pp[7]+self.F_poly_conti(wave, pp[11:14])
        if self.broken_pl == True:
            conti_flux = broken_pl_model(wave, pp[7], pp[14], pp[6]) + self.F_poly_conti(wave, pp[11:14])
        # plt.plot(wave,conti_flux)
        L = np.array([])
        for LL in zip([1350., 3000., 5100.]):
            if wave.max() > LL[0] and wave.min() < LL[0]:
                L_tmp = np.asarray([np.log10(
                    LL[0]*self.Flux2L(conti_flux[np.where(abs(wave-LL[0]) < 5., True, False)].mean(), self.z))])
            else:
                L_tmp = np.array([-1.])
            L = np.concatenate([L, L_tmp])  # save log10(L1350,L3000,L5100)
        return L


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
        "Fit the FeII on the continuum from 2000 to 12000 A based on Verner et al. (2009)"
        fe_verner = self.fe_verner
        yval = np.zeros_like(xval)
        wave_Fe = fe_verner[:, 0]
        flux_Fe = fe_verner[:, 1]*8e-7
        ind = np.where((wave_Fe > 2000.) & (wave_Fe < 12000.), True, False)
        wave_Fe = wave_Fe[ind]
        flux_Fe = flux_Fe[ind]
        Fe_FWHM = pp[1]
        xval_new = xval*(1.0+pp[2])
        ind = np.where((xval_new > 2000.) & (xval_new < 12000.), True, False)
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
            flux_Fe_conv = np.convolve(flux_Fe, kernel, 'same')
            tck = interpolate.splrep(wave_Fe, flux_Fe_conv)
            yval[ind] = pp[0]*interpolate.splev(xval_new[ind], tck)
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
        y_scaled = y * pp[1]
        return y_scaled


    def Fit(self, name=None, nsmooth=1, and_or_mask=True, reject_badpix=True, deredden=True, wave_range=None,
            wave_mask=None, decomposition_host=True, BC03=False, Mi=None, npca_gal=5, npca_qso=20, 
            broken_pl=False, Fe_uv_op=True, Fe_verner09=False,
            Fe_flux_range=None, poly=False, BC=False, rej_abs=False, initial_guess=None, MC=True, n_trails=1,
            linefit=True, tie_lambda=True, tie_width=True, tie_flux_1=True, tie_flux_2=True, save_result=True,
            plot_fig=True, save_fig=True, plot_line_name=True, plot_legend=True, dustmap_path=dustmap_path, 
            save_fig_path=None, save_fits_path=None, save_fits_name=None, mask_compname=None):
        self.mask_compname = mask_compname
        self.broken_pl = broken_pl
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
        return super().Fit(name=name, nsmooth=nsmooth, and_or_mask=and_or_mask, reject_badpix=reject_badpix, 
                           deredden=deredden, wave_range=wave_range, wave_mask=wave_mask, 
                           decomposition_host=decomposition_host, BC03=BC03, Mi=Mi, npca_gal=npca_gal, 
                           npca_qso=npca_qso, Fe_uv_op=Fe_uv_op, Fe_verner09=Fe_verner09,
                           Fe_flux_range=Fe_flux_range, poly=poly, 
                           BC=BC, rej_abs=rej_abs, initial_guess=initial_guess, MC=MC, n_trails=n_trails, 
                           linefit=linefit, tie_lambda=tie_lambda, tie_width=tie_width, tie_flux_1=tie_flux_1, 
                           tie_flux_2=tie_flux_2, save_result=save_result, plot_fig=plot_fig, 
                           save_fig=save_fig, plot_line_name=plot_line_name, plot_legend=plot_legend, 
                           dustmap_path=dustmap_path, save_fig_path=save_fig_path, save_fits_path=save_fits_path, 
                           save_fits_name=save_fits_name)


    def _SaveResult(self, conti_result, conti_result_type, conti_result_name, line_result, line_result_type,
                    line_result_name, save_fits_path, save_fits_name):
        """Save all data to fits"""
        self.all_result = np.concatenate([conti_result, line_result])
        self.all_result_type = np.concatenate([conti_result_type, line_result_type])
        self.all_result_name = np.concatenate([conti_result_name, line_result_name])
        all_result = self.all_result.astype(float)
        t = Table(all_result, names=(self.all_result_name), dtype=self.all_result_type)
        self.result_table = t
        t.write(save_fits_path+save_fits_name+'.fits', format='fits', overwrite=True)


    def _PlotFig(self, ra, dec, z, wave, flux, err, decomposition_host, linefit, tmp_all, gauss_result, f_conti_model,
                 conti_fit, all_comp_range, uniq_linecomp_sort, line_flux, save_fig_path):
        """Plot the results"""
        if self.broken_pl == True:
            self.PL_poly = broken_pl_model(wave, 
                                           conti_fit.params[7],
                                           conti_fit.params[14],
                                           conti_fit.params[6]) + self.F_poly_conti(wave, conti_fit.params[11:14])
        else:
            self.PL_poly = conti_fit.params[6]*(wave/3000.0)**conti_fit.params[7]+self.F_poly_conti(wave, 
                                                                                                    conti_fit.params[11:14])
        
        matplotlib.rc('xtick', labelsize=20)
        matplotlib.rc('ytick', labelsize=20)
        
        if linefit == True:
            fig, axn = plt.subplots(nrows=2, ncols=np.max([self.ncomp, 1]), figsize=(15, 8),
                                    squeeze=False)  # prepare for the emission line subplots in the second row
            ax = plt.subplot(2, 1, 1)  # plot the first subplot occupying the whole first row
            if self.MC == True:
                mc_flag = 2
            else:
                mc_flag = 1
            
            lines_total = np.zeros_like(wave)
            line_order = {'r': 3, 'g': 7}  # to make the narrow line plot above the broad line
            
            temp_gauss_result = gauss_result
            for p in range(int(len(temp_gauss_result)/mc_flag/3)):
                # warn that the width used to separate narrow from broad is not exact 1200 km s-1 which would lead to wrong judgement
                # if self.CalFWHM(temp_gauss_result[(2+p*3)*mc_flag]) < 1200.:
                if temp_gauss_result[(2+p*3)*mc_flag] - 0.0017 <= 1e-10:    
                    color = 'g'
                else:
                    color = 'r'
                
                line_single = self.Onegauss(np.log(wave), temp_gauss_result[p*3*mc_flag:(p+1)*3*mc_flag:mc_flag])
                
                ax.plot(wave, line_single+f_conti_model, color=color, zorder=5)
                for c in range(self.ncomp):
                    axn[1][c].plot(wave, line_single, color=color, zorder=line_order[color])
                lines_total += line_single
            
            ax.plot(wave, lines_total+f_conti_model, 'b', label='line',
                    zorder=6)  # supplement the emission lines in the first subplot
            self.lines_total = lines_total
            for c, linecompname in enumerate(uniq_linecomp_sort):
                tname = texlinename(linecompname)
                axn[1][c].plot(wave, lines_total, color='b', zorder=10)
                axn[1][c].plot(wave, self.line_flux, 'k', zorder=0)
                
                axn[1][c].set_xlim(all_comp_range[2*c:2*c+2])
                f_max = line_flux[
                    np.where((wave > all_comp_range[2*c]) & (wave < all_comp_range[2*c+1]), True, False)].max()
                f_min = line_flux[
                    np.where((wave > all_comp_range[2*c]) & (wave < all_comp_range[2*c+1]), True, False)].min()
                axn[1][c].set_ylim(f_min*0.9, f_max*1.1)
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
            ax.set_title(str(self.sdss_name)+'   z = '+str(np.round(z, 4)), fontsize=20)
        else:
            # ax.set_title('ra,dec = ('+str(ra)+','+str(dec)+')   '+str(self.sdss_name)+'   $z$ = '+str(np.round(z, 4)),
                        #  fontsize=20)
            ax.set_title('ra, dec = ({:.6f}, {:.6f})   {}    $z$ = {:.4f}'.format(ra, dec, self.sdss_name.replace('_',' '), z),
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
        
        if linefit == True:
            ax.text(0.5, -1.45, r'Rest-frame wavelength ($\rm \AA$)', fontsize=22, transform=ax.transAxes,
                    ha='center')
            ax.text(-0.07, -0.1, r'$f_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)', fontsize=22,
                    transform=ax.transAxes, rotation=90, ha='center', rotation_mode='anchor')
        else:
            plt.xlabel(r'Rest-frame wavelength ($\rm \AA$)', fontsize=22)
            plt.ylabel(r'$f_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)', fontsize=22)
        
        if self.save_fig == True:
            plt.savefig(save_fig_path+self.sdss_name+'.pdf', bbox_inches='tight')
            plt.savefig(save_fig_path+self.sdss_name+'.jpg', dpi=300, bbox_inches='tight')
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
            subrange = [4435, 4685]
        f_fe_mgii_model = self.Fe_flux_mgii(wave, pp[0:3])
        f_fe_balmer_model = self.Fe_flux_balmer(wave, pp[3:6])
        f_fe_verner_model = self.Fe_flux_verner(wave, pp[3:6])
        f_pl_model = pp[6]*(wave/3000.0)**pp[7]
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
        Fe_flux = integrate.trapz(f_fe[ind], wave_in)
        Fe_ew = integrate.trapz(f_fe[ind]/contiflux, wave_in)
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
                    # call kmpfit for lines
                    
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
                            line_fit = self._do_line_kmpfit(linelist, line_flux, ind_line, ind_n, nline_fit, ngauss_fit)
                   
                    # calculate MC err
                    if self.MC == True and self.n_trails > 0:
                        all_para_std, fwhm_std, sigma_std, ew_std, peak_std, area_std, na_dict = self.new_line_mc(
                            np.log(wave[ind_n]), line_flux[ind_n], err[ind_n], self.line_fit_ini, self.line_fit_par,
                            self.n_trails, compcenter, linecompname, ind_line, nline_fit, linelist_fit, ngauss_fit)
                        self.na_all_dict.update(na_dict)
                    else:
                        na_dict = self.na_line_nomc(line_fit, linecompname, ind_line, 
                                  nline_fit, ngauss_fit)
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
                    
                    if self.MC == True and self.n_trails > 0:
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
                    else:
                        fur_result_tmp = np.array([fwhm, sigma, ew, peak, area])
                        fur_result_type_tmp = np.concatenate(
                            [fur_result_type_tmp, ['float', 'float', 'float', 'float', 'float']])
                        fur_result_name_tmp = np.array(
                            [br_name+'_whole_br_fwhm', br_name+'_whole_br_sigma', br_name+'_whole_br_ew',
                             br_name+'_whole_br_peak', br_name+'_whole_br_area'])
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
                    linecenter = np.float(linelist[linelist['linename']==line]['lambda'][0])
                    na_tmp = self.line_prop(linecenter, line_fit.params[par_ind:par_ind+3], 'narrow')
                    if na_tmp[0] == 0:
                        na_tmp = self.line_prop(linecenter, line_fit.params[par_ind:par_ind+3], 'broad')
                    na_all_dict[line]['fwhm'].append(na_tmp[0])
                    na_all_dict[line]['sigma'].append(na_tmp[1])
                    na_all_dict[line]['ew'].append(na_tmp[2])
                    na_all_dict[line]['peak'].append(na_tmp[3])
                    na_all_dict[line]['area'].append(na_tmp[4])
                except:
                    print('Mismatch.')
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
                        linecenter = np.float(linelist[linelist['linename']==line]['lambda'][0])
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
                    linecenter = np.float(linelist[linelist['linename']==linec]['lambda'][0])
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
                linecenter = np.float(linelist[linelist['linename']==line]['lambda'].item())
                line_scale = np.float(df_gauss[line+'_1_scale'])
                line_centerwave = np.float(df_gauss[line+'_1_centerwave'])
                line_sigma = np.float(df_gauss[line+'_1_sigma'])
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
                    linecenter = np.float(linelist[linelist['linename']==linec]['lambda'].item())
                    line_scale1 = np.float(df_gauss[linec+'_1_scale'])
                    line_centerwave1 = np.float(df_gauss[linec+'_1_centerwave'])
                    line_sigma1 = np.float(df_gauss[linec+'_1_sigma'])
                    line_scale2 = np.float(df_gauss[linew+'_1_scale'])
                    line_centerwave2 = np.float(df_gauss[linew+'_1_centerwave'])
                    line_sigma2 = np.float(df_gauss[linew+'_1_sigma'])
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
                    if res_list:
                        res_tmp = res_list[0]
                    else:
                        res_tmp = 0.0
                    if res_tmp == 0:
                        res_tmp = 0.0
                    na_line_result.update({res_name_tmp:res_tmp})
        self.na_line_result = na_line_result


    # -----line properties calculation function--------
    def line_prop(self, compcenter, pp, linetype):
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
            contiflux = self.conti_fit.params[6]*(np.exp(xx)/3000.0)**self.conti_fit.params[7]+self.F_poly_conti(
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
                lambda0 = integrate.trapz(line_flux, line_wave)  # calculate the total broad line flux
                lambda1 = integrate.trapz(line_flux*line_wave, line_wave)
                lambda2 = integrate.trapz(line_flux*line_wave*line_wave, line_wave)
                ew = integrate.trapz(np.abs(line_flux/contiflux), line_wave)
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
            contiflux = self.conti_fit.params[6]*(np.exp(xx)/3000.0)**self.conti_fit.params[7]+self.F_poly_conti(
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
                lambda0 = integrate.trapz(line_flux, line_wave)  # calculate the total broad line flux
                lambda1 = integrate.trapz(line_flux*line_wave, line_wave)
                lambda2 = integrate.trapz(line_flux*line_wave*line_wave, line_wave)
                ew = integrate.trapz(np.abs(line_flux/contiflux), line_wave)
                area = lambda0
                
                sigma = np.sqrt(lambda2/lambda0-(lambda1/lambda0)**2)/compcenter*c
            else:
                fwhm, sigma, ew, peak, area = 0., 0., 0., 0., 0.
        
        return fwhm, sigma, ew, peak, area

    def CalFWHM(self, logsigma):
        """transfer the logFWHM to normal frame"""
        return 2*np.sqrt(2*np.log(2))*(np.exp(logsigma)-1)*ac.c.to(u.Unit('km/s')).value


