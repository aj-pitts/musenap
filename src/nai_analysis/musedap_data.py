import numpy as np
from astropy.io import fits
from functools import cached_property

from src.nai_analysis.utils import defaults, file_handler

class MuseDAPData:
    """A class to handle unpacking and storing all of the data output by the MUSE DAP"""
    def __init__(self, galaxy_name: str, bin_method: str, analysis_plans: str,
                logcube_filepath: str, maps_filepath: str, local_filepath: str, config_filepath: str, mcmc_dir: str, verbose = False):
        # galaxy info
        self.galname = galaxy_name
        self.bin_method = bin_method
        self.analysisplan = analysis_plans

        # flags
        self.verbose = verbose

        # get data paths
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
        return cls(galaxy_name, binning_method, plans, datapath_dict['LOGCUBE'], datapath_dict['MAPS'], datapath_dict['LOCAL'],
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

    # config file properties
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
    def stellar_vel(self):
        return self._open_maps('STELLAR_VEL')
    
    @cached_property
    def stellar_vel_ivar(self):
        return self._open_maps('STELLAR_VEL_IVAR')
    
    @cached_property
    def stellar_vel_mask(self):
        return self._open_maps('STELLAR_VEL_MASK')

    @cached_property
    def emline_gflux(self):
        return self._open_maps('EMLINE_GFLUX')
    
    @cached_property
    def emline_gflux_ivar(self):
        return self._open_maps('EMLINE_GFLUX_IVAR')
    
    @cached_property
    def emline_gflux_mask(self):
        return self._open_maps('EMLINE_GFLUX_MASK')