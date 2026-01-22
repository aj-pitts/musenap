import numpy as np
from astropy.io import fits
from utils import defaults, file_handler, util
from functools import cached_property

class MuseDAPData:
    """A class to handle unpacking and storing all of the data output by the MUSE DAP"""
    def __init__(self, galaxy_name, bin_method, logcube_filepath, maps_filepath, local_filepath, config_filepath, verbose = False):
        # galaxy info
        self.name = galaxy_name
        self.bin_method = bin_method

        # flags
        self.verbose = verbose

        # get data paths
        self.logcube_path = logcube_filepath
        self.maps_path = maps_filepath
        self.local_path = local_filepath
        self.config_path = config_filepath

        # assign the config file to self
        self._open_config()

    @classmethod
    def from_name(cls, galaxy_name: str, binning_method: str, verbose = False):
        datapath_dict = file_handler.get_data_paths(galaxy_name, binning_method, verbose=verbose)

        return cls(galaxy_name, binning_method, datapath_dict['LOGCUBE'], datapath_dict['MAPS'], datapath_dict['LOCAL'],
                   datapath_dict['CONFIG'], verbose = verbose)


    @staticmethod
    def _validate_files() -> None:
        """
        Validate that necessary files from the pipeline exist. Only raises a warning for local file
        """
        pass

    def _open_config(self):
        self.default_config = file_handler.parse_ini_file(self.config_path)

    def _open_logcube(self, HDU_name: str) -> np.ndarray:
        with fits.open(self.logcube) as hdul:
            return hdul[HDU_name].data.copy()
    
    def _open_maps(self, HDU_name: str) -> np.ndarray:
        with fits.open(self.maps) as hdul:
            return hdul[HDU_name].data.copy()
    
    def _open_local(self, HDU_name: str) -> np.ndarray:
        with fits.open(self.local) as hdul:
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
    
    def compute_MAPs(self):
        pass

    def sfr(self):
        pass

    def bpt(self):
        pass

    def analyze(self) -> None:
        pass

    def plot_results(self) -> None:
        pass



# class MUSEData:
#     """Short one-line summary.
    
#     Longer description of what the class represents,
#     what main data it stores, and how it’s typically used.
#     """

#     # 1️⃣ Class-level constants or defaults
#     DEFAULT_EXTENSIONS = ("FLUX", "VARIANCE", "MASK")

#     # 2️⃣ __init__ and dunder (special) methods
#     def __init__(self, path: str):
#         self.path = path
#         self._header_cache = None

#     def __repr__(self):
#         return f"<MUSEData: {self.path}>"

#     # 3️⃣ Class methods and static methods (if any)
#     @classmethod
#     def from_directory(cls, directory: str):
#         ...

#     @staticmethod
#     def _validate_header(header):
#         ...

#     # 4️⃣ Private helper methods (_something)
#     def _open_hdul(self):
#         ...

#     def _get_extension(self, key: str):
#         ...

#     # 5️⃣ Public methods
#     def get_flux(self):
#         ...

#     def make_ew_map(self):
#         ...

#     # 6️⃣ Properties (often grouped near the bottom)
#     @property
#     def flux(self):
#         return self._get_extension("FLUX")

#     @property
#     def variance(self):
#         return self._get_extension("VARIANCE")