import numpy as np
from measurement_map import measurementMAP
from maps.musemap import MuseMAP
from maps.bitmask import MuseMapBitMask

from utils import util

import time

class RedshiftMAP(measurementMAP):
    def compute(self) -> MuseMAP:
        start = time.time()

        c = 2.998e5

        dap_data = self.dap_data

        spatial_bins = dap_data.spatial_bins
        valid_bins = spatial_bins!=-1

        z_sys = dap_data.redshift

        stellar_vel = dap_data.stellar_vel
        stellar_vel_ivar = dap_data.stellar_vel_ivar
        stellar_vel_mask = dap_data.stellar_vel_mask

        redshift = MuseMAP.empty_from_binmap(spatial_bins, 'redshift')
        bm = MuseMapBitMask()

        z = (stellar_vel * (1 + z_sys)) / c + z_sys
        z_error = ((1/np.sqrt(stellar_vel_ivar)) / c) * (1 + z_sys)

        dap_bad = util.DAP_pix_mask(stellar_vel_mask)
        error_inf = ~np.isfinite(z_error)
        z_inf = ~np.isfinite(z)

        bm.set_flag(redshift.mask, ~valid_bins, ["no value", "do not use"])
        bm.set_flag(redshift.mask, dap_bad, ['do not use'])
        bm.set_flag(redshift.mask, z_inf, ["math error", "do not use"])
        bm.set_flag(redshift.mask, error_inf, ["unreliable", "uncertainty_oob"])

        redshift.data[valid_bins] = z[valid_bins]
        redshift.error[valid_bins] = z_error[valid_bins]

        dnu = bm.flagged(redshift.mask, 'do not use')
        redshift.data[dnu] = 0
        
        end = time.time()
        util.sys_message(f"    Constructed Redshift MAP: time to complete {end-start:.3g} s", color='green', verbose=self.verbose)
        return redshift