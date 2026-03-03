import numpy as np
from astropy.io import fits
from tqdm import tqdm
from musedap_data import MuseDAPData
from utils import util, bitmask
from typing import Optional, TYPE_CHECKING

from maps.musemap import MuseMAP
from maps.musemap import MuseMapBitMask
from musedap_data import MuseDAPData
from measurement_map import MeasurementMAP

if TYPE_CHECKING:
    from engine.measurement_engine import MeasurementEngine


class SnrMAP(MeasurementMAP):

    name = "snr_nai"
    dependencies = ['redshift']

    def compute(self, engine: "MeasurementEngine") -> MuseMAP:
        ## unpack DAP data
        DAP = self.dap_data

        spatial_bins = DAP.spatial_bins
        flux = DAP.flux
        ivar = DAP.ivar
        wave = DAP.wave

        ## create redshift MuseMAP if not input
        redshifts = engine.get('redshift')

        redshift_map = redshifts.data

        ## initialize MuseMAP for output and bitmask
        snr_MAP = MuseMAP.empty_from_binmap(spatial_bins, 'SNR_NAI')
        bm = MuseMapBitMask()

        ## continuum around Na I
        windows = [(5865, 5875), (5915, 5925)]
        blim = windows[0]
        rlim = windows[1]

        ## 3D rest wave cube
        rest_wave = wave[None, None, :] / (redshift_map + 1)[:, :, None]

        ## slice to the continuum windows
        w = ( (rest_wave >= blim[0]) & (rest_wave <= blim[1]) ) | ( (rest_wave >= rlim[0]) & (rest_wave <= rlim[1]) )
        flux_sliced = np.where(w, flux, np.nan)
        ivar_sliced = np.where(w, ivar, np.nan)

        s2n = flux_sliced * np.sqrt(ivar_sliced)
        s2n_med = np.nanmedian(s2n, axis=0)

        ## set bitmask values
        valid_bins = spatial_bins != -1
        bm.set_flag(snr_MAP.mask, ~valid_bins, ["NO_VALUE", "DO_NOT_USE"])
        bm.set_flag(snr_MAP.mask, ~np.isfinite(s2n_med), ["MATH_ERROR", "DO_NOT_USE"])

        ## assign the data
        snr_MAP.data = s2n_med

        ## overwrite DO NOT USE data with 0
        dnu = bm.flagged(snr_MAP.mask, "DO_NOT_USE")
        snr_MAP.data[dnu] = 0

        return snr_MAP

    def plot(self, engine: "MeasurementEngine", directory: str, figname: str):
        map_data = engine.get(self.name)
        map_data.plot_map(directory=directory, figname=figname, title=r"$S/N_{\mathrm{Na\ I}}$",
                          show = False, save = True, cmap='rainbow')
        map_data.plot_hist(directory=directory, figname=figname, title=r"$S/N_{\mathrm{Na\ I}}$",
                           show=False, save=True)