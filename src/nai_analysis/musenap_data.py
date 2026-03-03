import numpy as np
from astropy.io import fits
from utils import defaults, file_handler, util
from functools import cached_property
import os

from musedap_data import MuseDAPData

class MuseNAPData:
    def __init__(self, galaxy_name: str, binning_method: str, analysis_plans: str):
        self.galname = galaxy_name
        self.bin_method = binning_method
        self.analysisplan = analysis_plans

    @staticmethod
    def _validate():
        pass

    @classmethod
    def from_DAP_data(cls, DAP_data: MuseDAPData):
        return cls(DAP_data.galname, DAP_data.bin_method, DAP_data.analysisplan)
    
    def write_data(self) -> None:
        local_path = os.path.join(defaults.get_data_path(), 'local')
        analysis_plans = defaults.analysis_plans()
        corr_key = defaults.corr_key()
        
        fullpath = os.path.join(local_path, f"{galname}-{bin_method}", analysis_plans, corr_key)
        util.check_filepath(fullpath, verbose=verbose)
        
        filename = f"{galname}-{bin_method}-NAP-MAPs.fits"
        filepath = os.path.join(fullpath, filename)
        pass