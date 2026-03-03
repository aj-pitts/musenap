import numpy as np
from measurement_map import MeasurementMAP
from maps.musemap import MuseMAP
from maps.bitmask import MuseMapBitMask

from utils import util

import time

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from nai_analysis.engine.measurement_engine import MeasurementEngine

class RedshiftMAP(MeasurementMAP):

    name = "redshift"
    dependencies = []

    def compute(self) -> MuseMAP:
        start = time.time()

        c = 2.998e5

        dap_data = self.dap_data

        spatial_bins = dap_data.spatial_bins

        z_sys = dap_data.redshift

        stellar_vel = dap_data.stellar_vel
        stellar_vel_ivar = dap_data.stellar_vel_ivar
        stellar_vel_mask = dap_data.stellar_vel_mask

        redshift = MuseMAP.empty_from_binmap(spatial_bins, 'redshift')
        bm = MuseMapBitMask()

        z = (stellar_vel * (1 + z_sys)) / c + z_sys
        z_error = ((1/np.sqrt(stellar_vel_ivar)) / c) * (1 + z_sys)


        dap_bad = util.DAP_pix_mask(stellar_vel_mask)

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
        util.sys_message(f"    Constructed Redshift MAP: time to complete {end-start:.3g} s", color='green', verbose=self.verbose)
        return redshift
    
    def plot(self, engine: "MeasurementEngine", directory: str, figname: str):
        map_data = engine.get(self.name)
        map_data.plot_map(directory=directory, figname=figname, title=r"$z$",
                          show = False, save = True, cmap='coolwarm')
        map_data.plot_hist(directory=directory, figname=figname, title=r"$z$",
                           show=False, save=True)