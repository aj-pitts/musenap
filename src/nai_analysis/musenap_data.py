import numpy as np
from astropy.io import fits
from astropy.io.fits import PrimaryHDU, ImageHDU, BinTableHDU
from astropy.io.fits.header import Header
import os
from typing import Union, TYPE_CHECKING

from src.nai_analysis.utils import defaults, util
from src.nai_analysis.musedap_data import MuseDAPData

if TYPE_CHECKING:
    from src.nai_analysis.maps.musemap import MuseMAP

class MuseNAPData:
    def __init__(self, galaxy_name: str, binning_method: str, analysis_plans: str, verbose: bool = False):

        self._validate(galaxy_name, binning_method, analysis_plans, verbose)

        self.galname = galaxy_name
        self.bin_method = binning_method
        self.analysisplan = analysis_plans
        self.verbose = verbose

        self._initialize()


    @classmethod
    def from_DAP_data(cls, DAP_data: MuseDAPData):
        return cls(DAP_data.galname, DAP_data.bin_method, DAP_data.analysisplan)
    

    @staticmethod
    def _validate(galaxy_name: str, binning_method: str, analysis_plans: str, verbose: bool) -> None:
        for key, arg in {'galaxy_name':galaxy_name, 'binning_method':binning_method, 'analysis_plans':analysis_plans}:
            if not isinstance(arg, str):
                raise ValueError(f"Input for '{key}' must be a string: {arg}")
            
        if not isinstance(verbose, bool):
            raise ValueError(f"Input for 'verbose' must be a bool")
        

    def _initialize(self) -> None:
        util.sys_message(f"Initializing NAP data...", verbose=self.verbose)

        self.galaxy_dir = defaults.get_local_galaxy_dir(self.galname, self.bin_method)
        util.check_filepath(self.galaxy_dir, mkdir=True, verbose=self.verbose)

        self.filename = f"{self.galname}-{self.bin_method}-NAP_MAPS.fits"

        self.filepath = os.path.join(self.galaxy_dir, self.filename)
        if not os.path.isfile(self.filepath):
            util.sys_message(f"NAP MAPS file not found: {self.filepath}", status='WARN', color='yellow', verbose=True)
            self.fileready = False
            self._hdul = None
        else:
            util.sys_message(f"NAP MAPS file: {self.filepath}", status='INFO', color='green', verbose=self.verbose)
            self.fileready = True
            self._hdul = fits.open(self.filepath, memmap=True)

        if self.verbose and self.fileready:
            print(f"{self.galname} {self.bin_method} NAP MAPS:")
            self.print_fileinfo()

    def print_fileinfo(self) -> None:
        """Prints the info of the `filepath` FITS HDU List"""
        with fits.open(self.filepath) as hdul:
            print(hdul.info())

    def get_hdu(self, hdu_name: str) -> Union[PrimaryHDU, ImageHDU, BinTableHDU]:
        """Returns the FITS HDU specified by `hdu_name`"""
        return self._hdul[hdu_name.upper()]
        
    def get_data(self, hdu_name: str) -> np.ndarray:
        """Returns the data of the FITS HDU specified by `hdu_name`"""
        return self._hdul[hdu_name.upper()].data.copy()
        
    def get_header(self, hdu_name: str) -> Header:
        """Returns the header of the FITS HDU specified by `hdu_name`"""
        return self._hdul[hdu_name.upper()].header.copy()
    
    def close_hdul(self) -> None:
        self._hdul.close()
    
    @property
    def available_hdus(self) -> list[str]:
        """Returns the list of HDU names of `filepath`"""
        return [hdu.name for hdu in self._hdul]
    
    def get_map(self, hdu_name: str) -> "MuseMAP":
        """Returns a MuseMAP of the data and the error and mask if available"""
        data = self.get_data(hdu_name)
        error = self._get_hdu_data_or_none(f"{hdu_name}_error")
        mask = self._get_hdu_data_or_none(f"{hdu_name}_mask")

        return MuseMAP(
            name=hdu_name,
            galname=self.galname,
            bin_method=self.bin_method,
            data=data,
            mask=mask,
            error=error,
        )

    def plot_map_grid(self) -> None:
        return

    def _get_hdu_data_or_none(self, hdu_name: str) -> Union[np.ndarray, None]:
        try:
            with fits.open(self.filepath) as hdul:
                return hdul[hdu_name.upper()].data.copy()
        except KeyError:
            return None