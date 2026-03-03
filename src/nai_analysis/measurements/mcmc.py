import numpy as np

from measurement_map import MeasurementMAP
from maps.musemap import MuseMAP
from maps.bitmask import MuseMapBitMask

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.measurement_engine import MeasurementEngine

class LambdaMAP(MeasurementMAP):

    name = 'lambda'
    dependencies = ['mcmc_table']

    def compute(self, engine: "MeasurementEngine") -> MuseMAP:
        DAP = self.dap_data
        spatial_bins = DAP.spatial_bins

        mcmc_results = engine.get('mcmc_table')
        
        lambda_map = MuseMAP.empty_from_binmap("Lambda0", spatial_bins)
        bm = MuseMapBitMask()

        data = np.zeros_like(spatial_bins)
        error = np.zeros_like(spatial_bins)

        for row in mcmc_results:
            binid = row['bin']
            w = spatial_bins == binid

            lam, err1, err2 = row['percentiles'][0,:]

            data[w] = lam
            error[w] = np.mean([err1, err2])

        lambda_map.data = data
        lambda_map.error = error

        bm.set_flag(lambda_map.mask, spatial_bins==-1, ["NO_VALUE", "DO_NOT_USE"])
        bm.set_flag(lambda_map.mask, data == 0, ["MATH_ERROR", "DO_NOT_USE"])
        bm.set_flag(lambda_map.mask, error == 0, ["UNRELIABLE", "UNCERTAINTY_OOB"])

        return lambda_map
    
class LogNMAP(MeasurementMAP):

    name = 'logn'
    dependencies = ['mcmc_table']

    def compute(self, engine: "MeasurementEngine") -> MuseMAP:
        DAP = self.dap_data
        spatial_bins = DAP.spatial_bins

        mcmc_results = engine.get('mcmc_table')
        
        logn_map = MuseMAP.empty_from_binmap("logN", spatial_bins)
        bm = MuseMapBitMask()

        data = np.zeros_like(spatial_bins)
        error = np.zeros_like(spatial_bins)

        for row in mcmc_results:
            binid = row['bin']
            w = spatial_bins == binid

            logn, err1, err2 = row['percentiles'][1,:]

            data[w] = logn
            error[w] = np.mean([err1, err2])

        logn_map.data = data
        logn_map.error = error

        bm.set_flag(logn_map.mask, spatial_bins==-1, ["NO_VALUE", "DO_NOT_USE"])
        bm.set_flag(logn_map.mask, data == 0, ["MATH_ERROR", "DO_NOT_USE"])
        bm.set_flag(logn_map.mask, error == 0, ["UNRELIABLE", "UNCERTAINTY_OOB"])

        return logn_map
    
class bDMAP(MeasurementMAP):

    name = 'bd'
    dependencies = ['mcmc_table']

    def compute(self, engine: "MeasurementEngine") -> MuseMAP:
        DAP = self.dap_data
        spatial_bins = DAP.spatial_bins

        mcmc_results = engine.get('mcmc_table')
        
        bd_map = MuseMAP.empty_from_binmap("bD", spatial_bins)
        bm = MuseMapBitMask()

        data = np.zeros_like(spatial_bins)
        error = np.zeros_like(spatial_bins)

        for row in mcmc_results:
            binid = row['bin']
            w = spatial_bins == binid

            lam, err1, err2 = row['percentiles'][2,:]

            data[w] = lam
            error[w] = np.mean([err1, err2])

        bd_map.data = data
        bd_map.error = error

        bm.set_flag(bd_map.mask, spatial_bins==-1, ["NO_VALUE", "DO_NOT_USE"])
        bm.set_flag(bd_map.mask, data == 0, ["MATH_ERROR", "DO_NOT_USE"])
        bm.set_flag(bd_map.mask, error == 0, ["UNRELIABLE", "UNCERTAINTY_OOB"])

        return bd_map
    
class CfMAP(MeasurementMAP):

    name = 'cf'
    dependencies = ['mcmc_table']

    def compute(self, engine: "MeasurementEngine") -> MuseMAP:
        DAP = self.dap_data
        spatial_bins = DAP.spatial_bins

        mcmc_results = engine.get('mcmc_table')
        
        cf_map = MuseMAP.empty_from_binmap("Cf", spatial_bins)
        bm = MuseMapBitMask()

        data = np.zeros_like(spatial_bins)
        error = np.zeros_like(spatial_bins)

        for row in mcmc_results:
            binid = row['bin']
            w = spatial_bins == binid

            lam, err1, err2 = row['percentiles'][3,:]

            data[w] = lam
            error[w] = np.mean([err1, err2])

        cf_map.data = data
        cf_map.error = error

        bm.set_flag(cf_map.mask, spatial_bins==-1, ["NO_VALUE", "DO_NOT_USE"])
        bm.set_flag(cf_map.mask, data == 0, ["MATH_ERROR", "DO_NOT_USE"])
        bm.set_flag(cf_map.mask, error == 0, ["UNRELIABLE", "UNCERTAINTY_OOB"])

        return cf_map