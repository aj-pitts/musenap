import numpy as np
from astropy.io import fits
import argparse
import warnings
import os
from tqdm import tqdm
from typing import Optional, TYPE_CHECKING
import time

from utils import util

from maps.musemap import MuseMAP
from maps.musemap import MuseMapBitMask
from musedap_data import MuseDAPData
from measurement_map import MeasurementMAP

if TYPE_CHECKING:
    from nai_analysis.engine.measurement_engine import MeasurementEngine

class WeqMAP(MeasurementMAP):

    name = "weq_nai"
    dependencies = ['redshift']

    def compute(self, engine: "MeasurementEngine") -> MuseMAP:
        start = time.time()
        DAP = self.dap_data

        flux = DAP.flux
        ivar = DAP.ivar
        mask = DAP.mask
        wave = DAP.wave
        model = DAP.model
        spatial_bins = DAP.spatial_bins

        c = 2.998e5 #speed of light km/s
        
        valid_bins = spatial_bins!=-1

        EW_map = MuseMAP.empty_from_binmap("EW_NAI", spatial_bins, default=-999, additional_bitdefs={"OVERMASKED":6})
        bm = MuseMapBitMask()

        redshifts = engine.get('redshift')

        redshift_map = redshifts.data

        ## define the Na D window bounds
        nad_region = 5885, 5905
        # continuum_lims = [(5850, 5870), (5910, 5930)]

        ## 3D rest wave cube
        rest_wave = wave[None, None, :] / (redshift_map + 1)[:, :, None]

        ## slice flux cube
        w = (rest_wave >= nad_region[0]) & (rest_wave <= nad_region[1])
        flux_slice = np.where(w, flux, np.nan)
        model_slice = np.where(w, model, np.nan)
        ivar_slice = np.where(w, ivar, np.nan)

        ## normalize flux
        normflux = flux_slice / model_slice
        normflux_error = 1 / np.sqrt(ivar_slice) / model_slice

        ## boxcar EW
        continuum = np.ones_like(normflux)
        dlambda = np.gradient(rest_wave, axis=0)
        W_eq = np.nansum( (continuum - normflux) * dlambda, axis=0 )
        W_eq_error = np.sqrt( np.nansum( (dlambda * normflux_error)**2 ) )

        bm.set_flag(EW_map.mask, ~valid_bins, ["NO_VALUE", "DO_NOT_USE"])
        bm.set_flag(EW_map.mask, ~np.isfinite(W_eq), ["MATH_ERROR", "DO_NOT_USE"])
        bm.set_flag(EW_map.mask, ~np.isfinite(W_eq_error), ["UNRELIABLE", "UNCERTAINTY_OOB"])

        EW_map.data = W_eq
        EW_map.error = W_eq_error

        dnu = bm.flagged(EW_map.mask, "DO_NOT_USE")
        EW_map.data[dnu] = -999
        EW_map.error[dnu] = -999
        return EW_map
    
    def plot(self, engine: "MeasurementEngine", directory: str, figname: str):
        map_data = engine.get(self.name)
        map_data.plot_map(directory=directory, figname=figname, title=r"$\mathrm{EW_{Na\ I}}$",
                          show = False, save = True, cmap='rainbow')
        map_data.plot_hist(directory=directory, figname=figname, title=r"$\mathrm{EW_{Na\ I}}$",
                           show=False, save=True)
    

class WeqAbsMAP(MeasurementMAP):

    name = "weq_abs_nai"
    dependencies = ['redshift']

    def compute(self, engine: "MeasurementEngine") -> MuseMAP:
        start = time.time()
        DAP = self.dap_data

        flux = DAP.flux
        ivar = DAP.ivar
        mask = DAP.mask
        wave = DAP.wave
        model = DAP.model
        spatial_bins = DAP.spatial_bins

        c = 2.998e5 #speed of light km/s
        
        valid_bins = spatial_bins!=-1

        EWabs_map = MuseMAP.empty_from_binmap("EW_ABS_NAI", spatial_bins, default=-999, additional_bitdefs={"OVERMASKED":6})
        bm = MuseMapBitMask()

        redshifts = engine.get('redshift')
        
        redshift_map = redshifts.data

        ## 3D rest wave cube
        rest_wave = wave[None, None, :] / (redshift_map + 1)[:, :, None]

        ## continuum region
        continuum_lims = [(5850, 5870), (5910, 5930)]
        blim = continuum_lims[0]
        rlim = continuum_lims[1]

        ## slice the continuum
        w = ( (rest_wave >= blim[0]) & (rest_wave <= blim[1]) ) | ( (rest_wave >= rlim[0]) & (rest_wave <= rlim[1]) )
        flux_continuum = np.where(w, flux, np.nan)
        model_continuum = np.where(w, model, np.nan)
        ivar_continuum = np.where(w, ivar, np.nan)

        normalized_continuum = flux_continuum / model_continuum

        median_2d = np.nanmedian(normalized_continuum, axis=0)
        std_2d = np.nanstd(normalized_continuum, axis=0)

        thresh = median_2d + std_2d

        ## define the Na D window bounds
        nad_region = 5885, 5905

        ## slice flux cube
        w = (rest_wave >= nad_region[0]) & (rest_wave <= nad_region[1])
        flux_slice = np.where(w, flux, np.nan)
        model_slice = np.where(w, model, np.nan)
        ivar_slice = np.where(w, ivar, np.nan)

        ## normalize flux
        normflux = flux_slice / model_slice
        normflux_error = 1 / np.sqrt(ivar_slice) / model_slice

        ## mask by the threshold
        w = normflux > thresh
        normflux_masked = np.where(w, normflux, np.nan)
        normflux_error_masked = np.where(w, normflux_error, np.nan)

        ## boxcar EW
        continuum = np.ones_like(normflux)
        dlambda = np.gradient(rest_wave, axis=0)
        W_eq = np.nansum( (continuum - normflux_masked) * dlambda, axis=0 )
        W_eq_error = np.sqrt( np.nansum( (dlambda * normflux_error_masked)**2 ) )

        bm.set_flag(EWabs_map.mask, ~valid_bins, ["NO_VALUE", "DO_NOT_USE"])
        bm.set_flag(EWabs_map.mask, ~np.isfinite(W_eq), ["MATH_ERROR", "DO_NOT_USE"])
        bm.set_flag(EWabs_map.mask, ~np.isfinite(W_eq_error), ["UNRELIABLE", "UNCERTAINTY_OOB"])

        EWabs_map.data = W_eq
        EWabs_map.error = W_eq_error

        dnu = bm.flagged(EWabs_map.mask, "DO_NOT_USE")
        EWabs_map.data[dnu] = -999
        EWabs_map.error[dnu] = -999
        return EWabs_map
    
    def plot(self, engine: "MeasurementEngine", directory: str, figname: str):
        map_data = engine.get(self.name)
        map_data.plot_map(directory=directory, figname=figname, title=r"$\mathrm{EW}_{abs}$",
                          show = False, save = True, cmap='rainbow')
        map_data.plot_hist(directory=directory, figname=figname, title=r"$\mathrm{EW}_{abs}$",
                           show=False, save=True)

class WeqEmMAP(MeasurementMAP):

    name = "weq_em_nai"
    dependencies = ['redshift']

    def compute(self, engine: "MeasurementEngine") -> MuseMAP:
        start = time.time()
        DAP = self.dap_data

        flux = DAP.flux
        ivar = DAP.ivar
        mask = DAP.mask
        wave = DAP.wave
        model = DAP.model
        spatial_bins = DAP.spatial_bins

        c = 2.998e5 #speed of light km/s
        
        valid_bins = spatial_bins!=-1

        EWem_map = MuseMAP.empty_from_binmap("EW_EM_NAI", spatial_bins, default=-999, additional_bitdefs={"OVERMASKED":6})
        bm = MuseMapBitMask()

        redshifts = engine.get('redshift')
        
        redshift_map = redshifts.data

        ## 3D rest wave cube
        rest_wave = wave[None, None, :] / (redshift_map + 1)[:, :, None]

        ## continuum region
        continuum_lims = [(5850, 5870), (5910, 5930)]
        blim = continuum_lims[0]
        rlim = continuum_lims[1]

        ## slice the continuum
        w = ( (rest_wave >= blim[0]) & (rest_wave <= blim[1]) ) | ( (rest_wave >= rlim[0]) & (rest_wave <= rlim[1]) )
        flux_continuum = np.where(w, flux, np.nan)
        model_continuum = np.where(w, model, np.nan)
        ivar_continuum = np.where(w, ivar, np.nan)

        normalized_continuum = flux_continuum / model_continuum

        median_2d = np.nanmedian(normalized_continuum, axis=0)
        std_2d = np.nanstd(normalized_continuum, axis=0)

        thresh = median_2d - std_2d

        ## define the Na D window bounds
        nad_region = 5885, 5905

        ## slice flux cube
        w = (rest_wave >= nad_region[0]) & (rest_wave <= nad_region[1])
        flux_slice = np.where(w, flux, np.nan)
        model_slice = np.where(w, model, np.nan)
        ivar_slice = np.where(w, ivar, np.nan)

        ## normalize flux
        normflux = flux_slice / model_slice
        normflux_error = 1 / np.sqrt(ivar_slice) / model_slice

        ## mask by the threshold
        w = normflux > thresh
        normflux_masked = np.where(w, normflux, np.nan)
        normflux_error_masked = np.where(w, normflux_error, np.nan)

        ## boxcar EW
        continuum = np.ones_like(normflux)
        dlambda = np.gradient(rest_wave, axis=0)
        W_eq = np.nansum( (continuum - normflux_masked) * dlambda, axis=0 )
        W_eq_error = np.sqrt( np.nansum( (dlambda * normflux_error_masked)**2 ) )

        bm.set_flag(EWem_map.mask, ~valid_bins, ["NO_VALUE", "DO_NOT_USE"])
        bm.set_flag(EWem_map.mask, ~np.isfinite(W_eq), ["MATH_ERROR", "DO_NOT_USE"])
        bm.set_flag(EWem_map.mask, ~np.isfinite(W_eq_error), ["UNRELIABLE", "UNCERTAINTY_OOB"])

        EWem_map.data = W_eq
        EWem_map.error = W_eq_error

        dnu = bm.flagged(EWem_map.mask, "DO_NOT_USE")
        EWem_map.data[dnu] = -999
        EWem_map.error[dnu] = -999
        return EWem_map
    
    def plot(self, engine: "MeasurementEngine", directory: str, figname: str):
        map_data = engine.get(self.name)
        map_data.plot_map(directory=directory, figname=figname, title=r"$\mathrm{EW}_{em}$",
                          show = False, save = True, cmap='rainbow')
        map_data.plot_hist(directory=directory, figname=figname, title=r"$\mathrm{EW}_{em}$",
                           show=False, save=True)