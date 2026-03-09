import numpy as np
import warnings
import time

from src.nai_analysis.measurements.measurement_map import MeasurementMAP
from src.nai_analysis.maps.musemap import MuseMAP
from src.nai_analysis.maps.bitmask import MuseMapBitMask
from src.nai_analysis.utils import util, progress

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.nai_analysis.engine.measurement_engine import MeasurementEngine

class SnrMAP(MeasurementMAP):

    name = "snr_nai"
    dependencies = ['redshift']

    def compute(self, engine: "MeasurementEngine") -> MuseMAP:
        with progress.ProgressWheel(f"Computing {self.name} MAP"):
            start = time.time()
            warnings.simplefilter("ignore", RuntimeWarning)
            ## unpack DAP data
            DAP = self.dap_data

            spatial_bins = DAP.spatial_bins
            flux = DAP.flux
            ivar = DAP.ivar
            wave = DAP.wave

            ## create redshift MuseMAP if not input
            redshifts_musemap = engine.get('redshift')

            redshift_map = redshifts_musemap.data

            ## initialize MuseMAP for output and bitmask
            snr_MAP = MuseMAP.empty_from_binmap(self.name, DAP.galname, DAP.bin_method, spatial_bins)
            bm = MuseMapBitMask()

            ## continuum around Na I
            windows = [(5865, 5875), (5915, 5925)]
            blim = windows[0]
            rlim = windows[1]

            ## 3D rest wave cube
            rest_wave = wave[:, None, None] / (redshift_map + 1)[None, :, :]

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

            end = time.time()
        util.sys_message(f"Constructed {self.name} MAP: time to complete {end-start:.3g} s", color='green', verbose=self.verbose)
        return snr_MAP