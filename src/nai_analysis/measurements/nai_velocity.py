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

class VcenMAP(MeasurementMAP):

    name = "v_cen"
    dependencies = ["mcmc_table"]

    def compute(self, engine: "MeasurementEngine") -> MuseMAP:
        with progress.ProgressWheel(f"Computing {self.name} MAP"):
            start = time.time()
            warnings.simplefilter("ignore", RuntimeWarning)
            DAP = self.dap_data
            spatial_bins = DAP.spatial_bins

            mcmc_results = engine.get('mcmc_table')

            vcen_map = MuseMAP.empty_from_binmap(self.name, DAP.galname, DAP.bin_method, spatial_bins, default=-999)
            bm = MuseMapBitMask()

            data = np.zeros_like(spatial_bins) - 999
            error = np.zeros_like(spatial_bins) - 999

            lamrest = 5897.558
            c = 2.998e5

            bins = mcmc_results['bin']
            velocities = mcmc_results['velocities']
            err = np.mean(mcmc_results['percentiles'][:, 0, 1:], axis=1) * c / lamrest

            for binind, v, e in zip(bins, velocities, err):
                w = spatial_bins == binind
                data[w] = v
                error[w] = e

            bm.set_flag(vcen_map.mask, spatial_bins==-1, ["NO_VALUE", "DO_NOT_USE"])
            bm.set_flag(vcen_map.mask, data == -999, ["DO_NOT_USE"])
            bm.set_flag(vcen_map.mask, error == - 999, ["UNRELIABLE", "UNCERTAINTY_OOB"])

            data[data==-999] = 0
            error[error==-999] = 0

            vcen_map.data = data
            vcen_map.error = error

            end = time.time()
        util.sys_message(f"Constructed {self.name} MAP: time to complete {end-start:.3g} s", color='green', verbose=self.verbose)
        return vcen_map
    
class VfracMAP(MeasurementMAP):

    name = "v_frac"
    dependencies = ["mcmc_table"]
    
    def compute(self, engine: "MeasurementEngine") -> MuseMAP:
        with progress.ProgressWheel(f"Computing {self.name} MAP"):
            start = time.time()
            warnings.simplefilter("ignore", RuntimeWarning)
            DAP = self.dap_data
            spatial_bins = DAP.spatial_bins

            mcmc_results = engine.get('mcmc_table')

            vfrac_map = MuseMAP.empty_from_binmap(self.name, DAP.galname, DAP.bin_method, spatial_bins)

            data = np.zeros_like(spatial_bins)

            lamrest = 5897.558

            bins = mcmc_results['bin']
            velocities = mcmc_results['velocities']
            lambda_samples = mcmc_results['lambda samples']

            for binind, v, samples in zip(bins, velocities, lambda_samples):
                w = spatial_bins == binind
                data[w] = np.sign(v) * np.sum(np.sign(v) * (lamrest - samples) > 0) / samples.size

            vfrac_map.data = data

            end = time.time()
        util.sys_message(f"Constructed {self.name} MAP: time to complete {end-start:.3g} s", color='green', verbose=self.verbose)
        return vfrac_map
    
class VmaxMAP(MeasurementMAP):

    name = "v_max"
    dependencies = ["mcmc_table"]
    
    def compute(self, engine: "MeasurementEngine") -> MuseMAP:
        with progress.ProgressWheel(f"Computing {self.name} MAP"):
            start = time.time()
            warnings.simplefilter("ignore", RuntimeWarning)
            DAP = self.dap_data
            spatial_bins = DAP.spatial_bins

            mcmc_results = engine.get('mcmc_table')

            vmax_map = MuseMAP.empty_from_binmap(self.name, DAP.galname, DAP.bin_method, spatial_bins)
            bm = MuseMapBitMask()

            data = np.zeros_like(spatial_bins) - 999
            error = np.zeros_like(spatial_bins) - 999

            lamrest = 5897.558
            c = 2.998e5
            ln1 = abs(np.log(0.1))

            bins = mcmc_results['bin']
            velocities = mcmc_results['velocities']
            err = np.mean(mcmc_results['percentiles'][:, 0, 1:], axis=1) * c / lamrest
            bds = mcmc_results['percentiles'][:, 2, 0]
            bd_errs = np.mean(mcmc_results['percentiles'][:, 2, 1:], axis=1)

            for binind, v, e, bd, bde in zip(bins, velocities, err, bds, bd_errs):
                w = spatial_bins == binind

                data[w] = v - np.sqrt( ln1 * bd )
                error[w] = np.sqrt( e**2 + np.sqrt( ln1 * bde )**2 )

            if np.sum(~np.isfinite(data)) > 0:
                bm.set_flag(vmax_map.mask, ~np.isfinite(data), ["DO_NOT_USE", "MATH_ERROR"])
                data[~np.isfinite(data)] = -999

            if np.sum(~np.isfinite(error)):
                bm.set_flag(vmax_map.mask, ~np.isfinite(error), ["UNRELIABLE", "UNCERTAINTY_OOB", "MATH_ERROR"])
                error[~np.isfinite(error)] = -999

            bm.set_flag(vmax_map.mask, spatial_bins==-1, ["NO_VALUE", "DO_NOT_USE"])
            bm.set_flag(vmax_map.mask, data == -999, ["DO_NOT_USE"])
            bm.set_flag(vmax_map.mask, error == - 999, ["UNRELIABLE", "UNCERTAINTY_OOB"])

            data[data==-999] = 0
            error[error==-999] = 0

            vmax_map.data = data
            vmax_map.error = error

            end = time.time()
        util.sys_message(f"Constructed {self.name} MAP: time to complete {end-start:.3g} s", color='green', verbose=self.verbose)
        return vmax_map