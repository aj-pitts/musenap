import numpy as np
from astropy.io import fits
from functools import cached_property
from typing import Optional

from src.nai_analysis.utils import defaults, file_handler

emline_keys = {
    'OII-3727':1,
    'OII-3729':2,
    'H12-3751':3,
    'H11-3771':4,
    'Hthe-3798':5,
    'Heta-3836':6,
    'NeIII-3869':7,
    'HeI-3889':8,
    'Hzet-3890':9,
    'NeIII-3968':10,
    'Heps-3971':11,
    'Hdel-4102':12,
    'Hgam-4341':13,
    'HeII-4687':14,
    'Hb-4862':15,
    'OIII-4960':16,
    'OIII-5008':17,
    'NI-5199':18,
    'NI-5201':19,
    'HeI-5877':20,
    'OI-6302':21,
    'OI-6365':22,
    'NII-6549':23,
    'Ha-6564':24,
    'NII-6585':25,
    'SII-6718':26,
    'SII-6732':27,
    'HeI-7067':28,
    'ArIII-7137':29,
    'ArIII-7753':30,
    'Peta-9017':31,
    'SIII-9071':32,
    'Pzet-9231':33,
    'SIII-9533':34,
    'Peps-9548':35
}

class DAPMap:
    """A simple container class to store DAP Maps, and their mask and ivar if present"""
    def __init__(self, name: str, data: np.ndarray, mask: Optional[np.ndarray], ivar: Optional[np.ndarray]):
        self.name = name
        self.data = data
        self.mask = mask
        self.ivar = ivar


class MuseDAPData:
    """A class to handle unpacking and storing all of the data output by the MUSE DAP"""
    def __init__(self, galaxy_name: str, bin_method: str, analysis_plans: str, cube_filepath: str,
                logcube_filepath: str, maps_filepath: str, local_filepath: str, config_filepath: str, mcmc_dir: str, verbose = False):
        # galaxy info
        self.galname = galaxy_name
        self.bin_method = bin_method
        self.analysisplan = analysis_plans

        # flags
        self.verbose = verbose

        # get data paths
        self.cube_path = cube_filepath
        self.logcube_path = logcube_filepath
        self.maps_path = maps_filepath
        self.local_path = local_filepath
        self.config_path = config_filepath
        self.mcmc_dir = mcmc_dir

        # assign the config file to self
        self._open_config()

    @classmethod
    def from_name(cls, galaxy_name: str, binning_method: str, verbose = False):
        datapath_dict = file_handler.get_data_paths(galaxy_name, binning_method, verbose=verbose)
        plans = defaults.analysis_plans()
        return cls(galaxy_name, binning_method, plans, datapath_dict['CUBE'], datapath_dict['LOGCUBE'], datapath_dict['MAPS'], datapath_dict['LOCAL'],
                   datapath_dict['CONFIG'], datapath_dict['MCMC_DIR'], verbose = verbose)


    @staticmethod
    def _validate_files() -> None:
        """
        Validate that necessary files from the pipeline exist. Only raises a warning for local file
        """
        pass

    def _open_config(self):
        self.default_config = file_handler.parse_ini_file(self.config_path)

    def _open_logcube(self, HDU_name: str) -> np.ndarray:
        with fits.open(self.logcube_path) as hdul:
            return hdul[HDU_name].data.copy()
    
    def _open_maps(self, HDU_name: str) -> np.ndarray:
        with fits.open(self.maps_path) as hdul:
            return hdul[HDU_name].data.copy()
    
    def _open_local(self, HDU_name: str) -> np.ndarray:
        with fits.open(self.local_path) as hdul:
            return hdul[HDU_name].data.copy()

    def get_map_data(self, HDU_name: str) -> DAPMap:
        data = self._open_maps(HDU_name)
        try:
            mask = self._open_maps(f"{HDU_name}_MASK")
        except:
            mask = None
        
        try:
            ivar = self._open_maps(f"{HDU_name}_MASK")
        except:
            ivar = None

        return DAPMap(HDU_name, data, mask, ivar)
        
    
    def get_emline(self, HDU_name: str, emline_key: str) -> DAPMap:
        ind = emline_keys.get(emline_key, None)
        if ind is None:
            raise ValueError(f"{emline_key} not a valid key\nValid emlines are {list(emline_keys.keys())}")
        
        ind -= 1 # convert from channel no. to Python index

        emline_map = self.get_map_data(HDU_name)
        data = emline_map.data[ind]
        mask = emline_map.mask[ind] if emline_map.mask is not None else None
        ivar = emline_map.ivar[ind] if emline_map.ivar is not None else None

        return DAPMap(f"{HDU_name}_{emline_key}", data, mask, ivar)

        
    ## config file properties
    @property
    def redshift(self):
        return float(self.default_config['z'])
    
    @property
    def ra(self):
        return float(self.default_config['objra'])
    
    @property
    def dec(self):
        return float(self.default_config['objdec'])
    
    @property
    def inclination(self):
        return np.cos(np.radians(1 - float(self.default_config['ell'])))
    
    @property
    def pa(self):
        return float(self.default_config['pa'])
    
    @property
    def r_eff(self):
        return float(self.default_config['reff'])

    ## logcube HDU's
    @cached_property
    def flux(self):
        return self._open_logcube('FLUX')

    @cached_property
    def ivar(self):
        return self._open_logcube('IVAR')
        
    @cached_property
    def mask(self):
        return self._open_logcube('MASK')
    
    @cached_property
    def wave(self):
        return self._open_logcube('WAVE')

    @cached_property
    def model(self):
        return self._open_logcube('MODEL')

    @cached_property
    def model_mask(self):
        return self._open_logcube('MODEL_MASK')

    @cached_property
    def spec_emline(self):
        return self._open_logcube('EMLINE')
    
    @cached_property
    def spec_stellar(self):
        return self._open_logcube('STELLAR')
    
    @cached_property
    def spec_stellar_mask(self):
        return self._open_logcube('STELLAR_MASK')
    
    @cached_property
    def spatial_bins(self):
        return self._open_logcube('BINID')[0]
    
    @cached_property
    def binid(self):
        return self._open_logcube('BINID')

    ## relevant MAPS HDU's
    @cached_property
    def r_coords(self):
        return self._open_maps('SPX_ELLCOO')[1]
    
    @cached_property
    def r_coords_arcsec(self):
        return self._open_maps('SPX_ELLCOO')[0]
    
    def plot_maps(self, show: bool = False, save: bool = True) -> None:
        from src.nai_analysis.plotting.dap_maps import DAP_MAPS
        DAP_MAPS(self, show=show, save=save, verbose=self.verbose)