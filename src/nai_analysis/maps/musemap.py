import os
import numpy as np
from astropy.io import fits
import astropy.io.fits.header
from typing import Optional
from utils import defaults, util
from bitmask import MuseMapBitMask


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


class MuseMAP:
    """A class to store the MAP measurements for the `MuseNAPData`"""
    def __init__(self, 
                name: str, spatial_bins: np.ndarray[float], data: np.ndarray[float], mask: Optional[np.ndarray[np.uint32]] = None, error: Optional[np.ndarray[float]] = None,
                header_dict: Optional[dict[str, object | str, tuple[object, str]]] = None, 
                bit_dict: Optional[dict[str, tuple[str, str]]] = None,
                binmap: Optional[np.ndarray] = None):

        self._validate()

        self.spatial_bins = spatial_bins
        self.data = data
        self.mask = mask
        self.error = error
        self.name = name.upper().replace(' ', '_')
        self.bit_dict = bit_dict if bit_dict is not None else MuseMapBitMask.format_header_dict()
        self.binmap = binmap
        self.header_dict = header_dict if header_dict is not None else self._default_header(name, bit_dict)

    ##TODO: finish validations of all init inputs
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
    def empty_from_binmap(cls, name: str, bin_map: np.ndarray, default: Optional[float | int] = 0,
                          nomask: Optional[bool] = False, additional_bitdefs: Optional[dict[str, int | tuple[int, str]]] = None):
        data = np.zeros_like(bin_map) - default
        mask = None if nomask else np.zeros_like(bin_map, dtype=np.uint32)
        bitinit = MuseMapBitMask(additional_bitdefs=additional_bitdefs)
        bitdict = bitinit.format_header_dict()
        return cls(name, bin_map, data, mask = mask, bin_map = bin_map, bit_dict = bitdict)

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

    def _header_formatter(header: astropy.io.fits.header.Header, header_dict: dict[str, tuple[object, str]]):
        for key, value_pair in header_dict.items():
            header[key.upper()] = value_pair
        return header

    ##TODO: verbose
    def write_to_fits(self, filepath: str, verbose = False) -> None:
        """Write the current map HDU's into the output data fits file"""

        if not os.path.isfile(filepath):
            raise ValueError(f"File does not exist: {filepath}")
        
        newhdul = fits.HDUList([fits.PrimaryHDU()])

        if self.error is None:
            self.header_dict['DATA'].pop("ERRDATA")
        if self.mask is None:
            self.header_dict['DATA'].pop("QUALDATA")

        dataheader = self._header_formatter(fits.Header(), self.header_dict['DATA'])
        datahdu = fits.ImageHDU(data=self.data, name=self.name, header=dataheader)
        newhdul.append(datahdu)

        if self.error is not None:
            errorheader = self._header_formatter(fits.Header(), self.header_dict['ERROR'])
            errorhdu = fits.ImageHDU(data=self.uncertainty, name=f"{self.name}_ERROR", header=errorheader)
            newhdul.append(errorhdu)

        if self.mask is not None:
            maskheader = self._header_formatter(fits.Header(), self.header_dict['MASK'])
            maskhdu = fits.ImageHDU(data=self.mask, name=f"{self.name}_MASK", header=maskheader)
            newhdul.append(maskhdu)

        with fits.open(filepath, mode="update") as hdul:
            for newhdu in newhdul:
                if newhdu.name in hdul:
                    hdu = hdul[newhdu.name]
                    hdu.data = newhdu.data
                    hdu.header = newhdu.header

                else:
                    hdul.append(newhdu)
            hdul.verify("fix")

    def plot_map(self, directory: str, figname: str, title: str,
                 show = False, save = True, verbose = False,
                 cmap = 'viridis', vmin = None, vmax = None, ax = None,
                 **imshow_kwargs):
        
        plotmap = np.copy(self.data)
        plotmap[self.mask.astype(bool)] = np.nan

        if ax is None:
            fig, ax = plt.subplots()

        im = ax.imshow(plotmap, origin = 'lower', extent=[32.4, -32.6,-32.4, 32.6],
                       cmap=cmap, vmin=vmin, vmax=vmax, **imshow_kwargs)
        ax.set_xlabel(r'$\Delta \alpha$ (arcsec)')
        ax.set_ylabel(r'$\Delta \delta$ (arcsec)')
        ax.set_facecolor('lightgray')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.01)

        cbar = plt.colorbar(im, cax=cax, orientation = 'horizontal')
        cbar.set_label(title, labelpad=-55)
        cax.xaxis.set_ticks_position('top')

        if save:
            util.check_filepath(filepath=directory, mkdir=True, verbose=verbose)
            savepath = os.path.join(directory, figname)
            plt.savefig(savepath, bbox_inches='tight')
            util.sys_message(f'Figure {figname} saved to {directory}')
        if show:
            plt.show()
        else:
            plt.close()

    def plot_hist(self, directory: str, figname: str, title: str,
                  nbins = 40,
                  show = False, save = True, verbose = False, ax = None, **hist_kwargs):
        
        data = util.get_unique_bin_values(np.copy(self.data), self.spatial_bins, self.mask)

        histbins = np.linspace(data.min(), data.max(), nbins)

        if ax is None:
            fig, ax = plt.subplots()

        ax.hist(data, bins=histbins, color='k', **hist_kwargs)
        ax.set_xlabel(title)
        ax.set_ylabel(r"$N_{\mathrm{bins}}$")

        if save:
            util.check_filepath(filepath=directory, mkdir=True, verbose=verbose)
            savepath = os.path.join(directory, figname)
            plt.savefig(savepath, bbox_inches='tight')
            util.sys_message(f'Figure {figname} saved to {directory}')
        if show:
            plt.show()
        else:
            plt.close()