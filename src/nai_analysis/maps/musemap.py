import os
import numpy as np
from astropy.io import fits
import astropy.io.fits.header
from typing import Optional

from src.nai_analysis.utils import defaults, util
from src.nai_analysis.maps.bitmask import MuseMapBitMask
from src.nai_analysis.plotting.map_plotter import plot_map, plot_hist
from src.nai_analysis.plotting.plot_config import PLOT_CONFIG

class MuseMAP:
    """A class to store the MAP measurements for the `MuseNAPData`"""
    def __init__(self, 
                name: str, galname: str, bin_method: str,
                spatial_bins: np.ndarray[float], data: np.ndarray[float], mask: Optional[np.ndarray[np.uint32]] = None, error: Optional[np.ndarray[float]] = None,
                header_dict: Optional[dict[str, object | str, tuple[object, str]]] = None, 
                bit_dict: Optional[dict[str, tuple[str, str]]] = None):

        self._validate(name, galname, bin_method, spatial_bins, data, error, mask, header_dict, bit_dict)

        self.galname = galname
        self.bin_method = bin_method
        self.name = name.upper().replace(' ', '_')

        self.directory = defaults.get_local_galaxy_dir(self.galname, self.bin_method)
        self.filename = f"{self.galname}-{self.bin_method}-NAP_MAPS.fits"

        self.spatial_bins = spatial_bins
        self.data = data
        self.mask = mask
        self.error = error
        self.bit_dict = bit_dict if bit_dict is not None else MuseMapBitMask.format_header_dict()
        self.header_dict = header_dict if header_dict is not None else self._default_header(name, bit_dict)


    @staticmethod
    def _validate(name, galname, bin_method, spatial_bins, data, error, mask, header_dict, bit_dict):

        for key, arg in {'name':name, 'galname':galname, 'bin_method':bin_method}.items():
            if not isinstance(arg, str):
                raise ValueError(f"Input for '{key}' must be a string. {type(arg)} given")
        
        for key, arg in {'data':data, 'error':error, 'mask':mask, 'spatial_bins':spatial_bins}.items():
            if arg is not None:
                if not isinstance(arg, np.ndarray):
                    raise ValueError(f"Input for '{key}' must be a NumPy array. {type(arg)} given")
                if arg.ndim != 2:
                    raise ValueError(f"Input for '{key}' must be 2D.")
                if arg.shape != spatial_bins.shape:
                    raise ValueError(f"Shape mismatch between {key} and spatial_bins")
            else:
                if key == 'data' or key == 'spatial_bins':
                    raise ValueError(f"{key} cannot be None")
        
        if header_dict is not None:
            if not isinstance(header_dict, dict):
                raise ValueError(f"header_dict must be a dictionary")
            else:
                for key, value in header_dict.items():
                    if isinstance(value, tuple):
                        if not isinstance(value[0], object) and isinstance(value[1], str):
                            raise ValueError("values of header dict must be either `object` or `tuple[object, str]`")
                    elif not isinstance(value, object):
                        raise ValueError("values of header dict must be either `object` or `tuple[object, str]`")
                    
                    if not isinstance(key, str):
                        raise ValueError("keys of header_dict must be strings")
        
        if bit_dict is not None:
            if not isinstance(bit_dict, dict):
                raise ValueError(f"header_dict must be a dictionary")
            else:
                for key, value in bit_dict.items():
                    if not isinstance(value, tuple):
                        raise ValueError("values of bit dict must be `tuple[str, str]`")
                    else:
                        if not isinstance(value[0], str) and isinstance(value[1], str):
                            raise ValueError("values of bit dict must be `tuple[str, str]`")
                    
                    if not isinstance(key, str):
                        raise ValueError("keys of bit dict must be strings")
    
    @staticmethod
    def _validate_header(header):
        pass
        
    @classmethod
    def empty_from_binmap(cls, name: str, galname: str, bin_method: str, bin_map: np.ndarray, default: Optional[float | int] = 0.0,
                          nomask: Optional[bool] = False, additional_bitdefs: Optional[dict[str, int | tuple[int, str]]] = None):
        data = np.zeros_like(bin_map, dtype=float) - default
        mask = None if nomask else np.zeros_like(bin_map, dtype=np.uint32)
        bitinit = MuseMapBitMask(additional_bitdefs=additional_bitdefs)
        bitdict = bitinit.format_header_dict()
        return cls(name, galname, bin_method, bin_map, data = data, mask = mask, bit_dict = bitdict)
    
    @classmethod
    def from_hdu(cls, hdu_name: str, galname: str, bin_method: str, bin_map: np.ndarray, filepath: Optional[str] = None):
        hdu_name = hdu_name.upper()
        header_dict = {}

        with fits.open(filepath) as hdul:
            data = hdul[hdu_name].data.copy()
            header = hdul[hdu_name].header.copy()

            hdr_dict = {key: (header[key], header.comments[key]) for key in header.keys()}
            header_dict['DATA'] = hdr_dict

            try:
                mask = hdul[f"{hdu_name}_MASK"].data.copy()
                mask_header = hdul[f"{hdu_name}_MASK"].header.copy()
                mhdr_dict = {key: (mask_header[key], mask_header.comments[key]) for key in mask_header.keys()}
                header_dict['MASK'] = mhdr_dict
            except KeyError:
                mask = None
                header_dict['MASK'] = None

            try:
                error = hdul[f"{hdu_name}_ERROR"].data.copy()
                error_header = hdul[f"{hdu_name}_ERROR"].header.copy()
                ehdr_dict = {key: (error_header[key], error_header.comments[key]) for key in error_header.keys()}
                header_dict['ERROR'] = ehdr_dict
            except KeyError:
                error = None
                header_dict['ERROR'] = None


        return cls(hdu_name, galname, bin_method, bin_map, data = data, mask = mask, error = error, header_dict = header_dict)

    @staticmethod
    def _default_header(name: str, bit_dict: dict) -> dict:
        data_dict = {
                "DESC":(f"{name.replace("_"," ")} map",""),
                "ERRDATA":(f"{name}_ERROR", "Associated uncertainty values extension"),
                "QUALDATA":(f"{name}_MASK", "Associated quality extension"),
                "EXTNAME":(name, "Extension name"),
                "AUTHOR":("Andrew Pitts","")
        }
        error_dict = {
                "DESC":(f"{name.replace("_"," ")} uncertainty map",""),
                "DATA":(f"{name}", "Associated values extension"),
                "QUALDATA":(f"{name}_MASK", "Associated quality extension"),
                "EXTNAME":(f"{name}_ERROR", "Extension name"),
                "AUTHOR":("Andrew Pitts","")
        }
        mask_dict = {
                "DESC":(f"{name.replace("_"," ")} mask map",""),
                "DATA":(f"{name}", "Associated values extension"),
                "ERRDATA":(f"{name}_ERROR", "Associated uncertainty values extension"),
                "EXTNAME":(f"{name}_MASK", "Extension name")
        } | bit_dict | {
                "AUTHOR":("Andrew Pitts","")
        }

        return {
            "DATA":data_dict,
            "ERROR":error_dict,
            "MASK":mask_dict
        }

    @staticmethod
    def _header_formatter(header: astropy.io.fits.header.Header, header_dict: dict[str, tuple[object, str]]):
        for key, value_pair in header_dict.items():
            header[key.upper()] = value_pair
        return header

    def plot_data(self, verbose: bool = False) -> None:
        if self.name.upper() not in list(PLOT_CONFIG.keys()):
            util.sys_message(f"Plotting configuration not set for {self.name}", verbose = verbose)
            return
        plot_map(self, verbose=verbose)
        plot_hist(self, verbose=verbose)

    def write_to_fits(self, verbose = False) -> None:
        """Write the current map HDU's into the output data fits file"""

        filepath = os.path.join(self.directory, self.filename)
        
        if not os.path.isfile(filepath):
            newhdul = fits.HDUList([fits.PrimaryHDU()])
            newhdul.writeto(filepath)

        else:
            #newhdul = fits.HDUList([])
            newhdul = []

        if self.error is None:
            self.header_dict['DATA'].pop("ERRDATA")
        if self.mask is None:
            self.header_dict['DATA'].pop("QUALDATA")

        dataheader = self._header_formatter(fits.Header(), self.header_dict['DATA'])
        datahdu = fits.ImageHDU(data=self.data, name=self.name, header=dataheader)
        newhdul.append(datahdu)

        if self.error is not None:
            errorheader = self._header_formatter(fits.Header(), self.header_dict['ERROR'])
            errorhdu = fits.ImageHDU(data=self.error, name=f"{self.name}_ERROR", header=errorheader)
            newhdul.append(errorhdu)

        if self.mask is not None:
            maskheader = self._header_formatter(fits.Header(), self.header_dict['MASK'])
            maskhdu = fits.ImageHDU(data=self.mask, name=f"{self.name}_MASK", header=maskheader)
            newhdul.append(maskhdu)

        with fits.open(filepath, mode="update") as hdul:
            for newhdu in newhdul:
                if newhdu.name in hdul:
                    idx = hdul.index_of(newhdu.name)
                    hdul[idx] = newhdu
                    # hdu = hdul[newhdu.name]
                    # hdu.data = newhdu.data
                    # #hdu.header = newhdu.header
                    # hdu.header.update(newhdu.header)
                    util.sys_message(f"Updating HDU {newhdu.name} in {filepath}", verbose=verbose)

                else:
                    hdul.append(newhdu)
                    util.sys_message(f"Adding HDU {newhdu.name} to {filepath}", verbose=verbose)

            hdul.verify("fix")