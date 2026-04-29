import numpy as np
import warnings

from src.nai_analysis.measurements.measurement_map import MeasurementMAP
from src.nai_analysis.maps.musemap import MuseMAP
from src.nai_analysis.maps.bitmask import MuseMapBitMask
from src.nai_analysis.utils import util, progress

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.nai_analysis.engine.measurement_engine import MeasurementEngine

import time

class RedshiftMAP(MeasurementMAP):

    name = "redshift"
    dependencies = []

    def compute(self, engine: "MeasurementEngine") -> MuseMAP:
        with progress.ProgressWheel(f"Computing {self.name} MAP"):
            start = time.time()
            warnings.simplefilter("ignore", RuntimeWarning)

            c = 2.998e5

            dap_data = self.dap_data

            spatial_bins = dap_data.spatial_bins

            z_sys = dap_data.redshift

            vref_data = dap_data.get_map_data("STELLAR_VEL") # dap_data.get_emline("EMLINE_GVEL", 'Ha-6564')
            vref = vref_data.data
            vref_ivar = vref_data.ivar
            vref_mask = vref_data.mask
            
            redshift = MuseMAP.empty_from_binmap(self.name, dap_data.galname, dap_data.bin_method, spatial_bins)
            bm = MuseMapBitMask()

            z = (vref * (1 + z_sys)) / c + z_sys
            z_error = ((1/np.sqrt(vref_ivar)) / c) * (1 + z_sys)


            dap_bad = util.DAP_pix_mask(vref_mask)

            bm.set_flag(redshift.mask, spatial_bins == -1, ["no value", "do not use"])
            bm.set_flag(redshift.mask, dap_bad, ['do not use'])
            bm.set_flag(redshift.mask, ~np.isfinite(z), ["math error", "do not use"])
            bm.set_flag(redshift.mask, ~np.isfinite(z_error), ["unreliable", "uncertainty_oob"])

            dnu = bm.flagged(redshift.mask, 'do not use')
            z[dnu] = -1
            z_error[dnu] = 0

            redshift.data = z
            redshift.error = z_error

            end = time.time()
        util.sys_message(f"Constructed {self.name} MAP: time to complete {end-start:.3g} s", color='green', verbose=self.verbose)
        return redshift