import os
import numpy as np
from astropy.io import fits
import astropy.io.fits.header
from typing import Optional
from utils import defaults, util
from bitmask import MuseMapBitMask

class MuseMAP:
    """A simple class to store the MAP measurements for the `MuseNAPData`"""
    def __init__(self, 
                 data: np.ndarray[float], error: np.ndarray[float], mask: np.ndarray[int], name: str,
                 data_header_dict: Optional[dict[str, object | str, tuple[object, str]]] = None, 
                 error_header_dict: Optional[dict[str, object | str, tuple[object, str]]] = None, 
                 mask_header_dict: Optional[dict[str, object | str, tuple[object, str]]] = None,
                 bit_dict: Optional[dict[str, tuple[str, str]]] = None,
                 binmap: Optional[np.ndarray] = None):
        
        self._validate()

        self.data = data
        self.mask = mask
        self.error = error
        self.name = name.upper().replace(' ', '_')
        self.bit_dict = bit_dict if bit_dict is not None else MuseMapBitMask.format_header_dict()
        self.binmap = binmap
        self.header_dict = header_dict if header_dict is not None else {}

    @staticmethod
    def _validate(data, error, mask, binmap, name):
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise ValueError("Data must be a 2D numpy array")
        
        if not isinstance(binmap, np.ndarray) or data.ndim != 2:
            raise ValueError("Data must be a 2D numpy array")
        
        if not isinstance(error, np.ndarray) or error.shape != data.shape:
            raise ValueError("Error must be a numpy array with shape equal to data")
        
        if not isinstance(mask, np.ndarray) or mask.shape != data.shape:
            raise ValueError("Mask must be a numpy array with shape equal to data")
        
        if not isinstance(name, str):
            raise ValueError("Name must be a string")
    
    @staticmethod
    def _validate_header(header):
        pass
        
    @classmethod
    def empty_from_binmap(cls, bin_map: np.ndarray, name: str, default: Optional[float | int] = 0, 
                          additional_bitdefs: Optional[dict[str, int | tuple[int, str]]] = None):
        data = np.zeros_like(bin_map) - default
        uncertainty = np.zeros_like(bin_map)
        mask = np.zeros_like(bin_map, dtype=np.uint32)
        bitinit = MuseMapBitMask(additional_bitdefs=additional_bitdefs)
        bitdict = bitinit.format_header_dict()
        return cls(data, uncertainty, mask, bin_map, name)
    


    def write_to_fits(self, galname: str, bin_method: str, verbose = False) -> None:
        """Write the current map HDU's into the output data fits file"""
        local_path = os.path.join(defaults.get_data_path(), 'local')
        analysis_plans = defaults.analysis_plans()
        corr_key = defaults.corr_key()
        
        fullpath = os.path.join(local_path, f"{galname}-{bin_method}", analysis_plans, corr_key)
        util.check_filepath(fullpath, verbose=verbose)
        
        filename = f"{galname}-{bin_method}-NAP-MAPs.fits"
        filepath = os.path.join(fullpath, filename)
        if not os.path.isfile(filepath):
            hdul = fits.HDUList([fits.PrimaryHDU()])
            
            dataheader_dict = self._default_header(galname, self.name) | self.header_dict
            dataheader = self._header_formatter(fits.Header(), dataheader_dict)
            datahdu = fits.ImageHDU(data=self.data, name=self.name, header=dataheader)
            errorhdu = fits.ImageHDU(data=self.uncertainty, name=f"{self.name}_ERROR")
            maskhdu = fits.ImageHDU(data=self.mask, name=f"{self.name}_MASK")
        else:
            hdul = fits.open(filepath, mode='update')

    @staticmethod
    def _default_header(galname: str, name: str) -> dict:
        return {
            "DESC":(f"{galname} {name.replace("_"," ")} map",""),
            "ERRDATA":(f"{name}_ERROR", "Associated uncertainty values extension"),
            "QUALDATA":(f"{name}_MASK", "Associated quality extension"),
            "EXTNAME":(name, "Extension name"),
            "AUTHOR":("Andrew Pitts","")
        }
    
    def _header_formatter(header: astropy.io.fits.header.Header, header_dict: dict[str, tuple[object, str]]):
        for key, value_pair in header_dict.items():
            header[key.upper()] = value_pair
        return header