import numpy as np

from measurement_map import MeasurementMAP
from maps.musemap import MuseMAP
from maps.bitmask import MuseMapBitMask

from typing import Optional, TYPE_CHECKING


if TYPE_CHECKING:
    from engine.measurement_engine import MeasurementEngine

class VcenMAP(MeasurementMAP):

    name = "vcen"
    dependencies = ["mcmc_table"]

    def compute(self, engine: "MeasurementEngine") -> MuseMAP:
        DAP = self.dap_data
        spatial_bins = DAP.spatial_bins

        mcmc_results = engine.get('mcmc_table')

        vcen_map = MuseMAP.empty_from_binmap("V_CEN", spatial_bins)
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
        bm.set_flag(vcen_map.mask, error == - 999, ["UNREALIABLE", "UNCERTAINTY_OOB"])

        data[data==-999] = 0
        error[error==-999] = 0

        vcen_map.data = data
        vcen_map.error = error

        return vcen_map
    
    def plot(self, engine: "MeasurementEngine", directory: str, figname: str):
        map_data = engine.get(self.name)
        map_data.plot_map(directory=directory, figname=figname, title=r"$v_{\mathrm{cen}}\ \left( \mathrm{km\ s^{-1}} \right)$",
                          show = False, save = True, cmap='rainbow')
        map_data.plot_hist(directory=directory, figname=figname, title=r"$v_{\mathrm{cen}}\ \left( \mathrm{km\ s^{-1}} \right)$",
                           show=False, save=True)
    
class VfracMAP(MeasurementMAP):

    name = "vfrac"
    dependencies = ["mcmc_table"]
    
    def compute(self, engine: "MeasurementEngine") -> MuseMAP:
        DAP = self.dap_data
        spatial_bins = DAP.spatial_bins

        mcmc_results = engine.get('mcmc_table')

        vfrac_map = MuseMAP.empty_from_binmap("V_FRAC", spatial_bins, nomask=True)

        data = np.zeros_like(spatial_bins)

        lamrest = 5897.558

        bins = mcmc_results['bin']
        velocities = mcmc_results['velocities']
        lambda_samples = mcmc_results['lambda samples']

        for binind, v, samples in zip(bins, velocities, lambda_samples):
            w = spatial_bins == binind
            data[w] = np.sign(v) * np.sum(np.sign(v) * (lamrest - samples) > 0) / samples.size

        vfrac_map.data = data

        return vfrac_map
    
    def plot(self, engine: "MeasurementEngine", directory: str, figname: str):
        map_data = engine.get(self.name)
        map_data.plot_map(directory=directory, figname=figname, title=r"$f_{\left| \Delta v \right| > 0}$",
                          show = False, save = True, cmap='rainbow')
        map_data.plot_hist(directory=directory, figname=figname, title=r"$f_{\left| \Delta v \right| > 0}$",
                           show=False, save=True)
    
class VmaxMAP(MeasurementMAP):

    name = "vmax"
    dependencies = ["mcmc_table"]
    
    def compute(self, engine: "MeasurementEngine") -> MuseMAP:
        DAP = self.dap_data
        spatial_bins = DAP.spatial_bins

        mcmc_results = engine.get('mcmc_table')

        vmax_map = MuseMAP.empty_from_binmap("V_MAX", spatial_bins)
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
            bm.set_flag(vmax_map.mask, ~np.isfinite(error), ["UNREALIABLE", "UNCERTAINTY_OOB", "MATH_ERROR"])
            error[~np.isfinite(error)] = -999

        bm.set_flag(vmax_map.mask, spatial_bins==-1, ["NO_VALUE", "DO_NOT_USE"])
        bm.set_flag(vmax_map.mask, data == -999, ["DO_NOT_USE"])
        bm.set_flag(vmax_map.mask, error == - 999, ["UNREALIABLE", "UNCERTAINTY_OOB"])

        data[data==-999] = 0
        error[error==-999] = 0

        vmax_map.data = data
        vmax_map.error = error
        return vmax_map
    
    def plot(self, engine: "MeasurementEngine", directory: str, figname: str):
        map_data = engine.get(self.name)
        map_data.plot_map(directory=directory, figname=figname, title=r"$v_{\mathrm{max\ out}}\ \left( \mathrm{km\ s^{-1}} \right)$",
                          show = False, save = True, cmap='rainbow')
        map_data.plot_hist(directory=directory, figname=figname, title=r"$v_{\mathrm{max\ out}}\ \left( \mathrm{km\ s^{-1}} \right)$",
                           show=False, save=True)