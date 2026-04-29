import os
import numpy as np
from astropy.io import fits
from astropy.io.fits.header import Header
from matplotlib.axes import Axes
from typing import Optional

from src.nai_analysis.utils import defaults, util
from src.nai_analysis.maps.bitmask import MuseMapBitMask
from src.nai_analysis.plotting.plot_config import PLOT_CONFIG

class MuseMAP:
    """A class to store the MAP measurements for the `MuseNAPData`"""
    def __init__(self, 
                name: str, galname: str, bin_method: str,
                spatial_bins: np.ndarray[float], data: np.ndarray[float], 
                mask: Optional[np.ndarray[np.uint32]] = None, 
                error: Optional[np.ndarray[float]] = None,
                header_dict: Optional[dict[str, object | str, tuple[object, str]]] = None, 
                additional_bitdefs: Optional[dict[str, int | tuple[int, str]]] = None):

        # validate inputs
        self._validate(name, galname, bin_method, spatial_bins, data, error, mask, header_dict)

        # initialize names
        self.galname = galname
        self.bin_method = bin_method
        self.name = name.upper().replace(' ', '_')

        # initialize paths
        self.directory = defaults.get_local_galaxy_dir(self.galname, self.bin_method)
        self.filename = defaults.get_default_filename(self.galname, self.bin_method)

        # initialize data
        self.spatial_bins = spatial_bins
        self.data = data
        self.mask = mask
        self.error = error

        # initialize bitmask
        self.bitmask = MuseMapBitMask(additional_bitdefs=additional_bitdefs)
        self.bit_dict = self.bitmask.format_header_dict()
        self.header_dict = header_dict if header_dict is not None else self._default_header(name, self.bit_dict)


    @staticmethod
    def _validate(name, galname, bin_method, spatial_bins, data, error, mask, header_dict):

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
    
    @staticmethod
    def _validate_header(header):
        pass
        
    @classmethod
    def empty_from_binmap(cls, name: str, galname: str, bin_method: str, bin_map: np.ndarray, default: Optional[float | int] = 0.0,
                          nomask: Optional[bool] = False, additional_bitdefs: Optional[dict[str, int | tuple[int, str]]] = None):
        data = np.zeros_like(bin_map, dtype=float) - default
        mask = None if nomask else np.zeros_like(bin_map, dtype=np.uint32)
        # bitinit = MuseMapBitMask(additional_bitdefs=additional_bitdefs)
        # bitdict = bitinit.format_header_dict()
        return cls(name, galname, bin_method, bin_map, data = data, mask = mask, additional_bitdefs = additional_bitdefs)
    
    @classmethod
    def from_hdu(cls, hdu_name: str, galname: str, bin_method: str, bin_map: np.ndarray, filepath: Optional[str] = None):
        hdu_name = hdu_name.upper()
        header_dict = {}

        if filepath is None:
            galdir = defaults.get_local_galaxy_dir(galname, bin_method)
            fname = defaults.get_default_filename(galname, bin_method)
            filepath = os.path.join(galdir, fname)

        if not os.path.exists(filepath):
            raise ValueError(f"File does not exist: {filepath}")

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
        
        if header_dict['MASK'] is not None:
            default_bits = list(range(6)) + [30]
            add_bits = {}
            for key, value in header_dict['MASK'].items():
                if "BIN_" in key:
                    if isinstance(value, tuple):
                        value = value[0]
                    b = int(key[-1])
                    if b not in default_bits:
                        add_bits[value] = b
            if len(add_bits) == 0:
                add_bits = None

        return cls(hdu_name, galname, bin_method, bin_map, data = data, mask = mask, error = error, header_dict = header_dict, additional_bitdefs = add_bits)

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
    def _header_formatter(header: Header, header_dict: dict[str, tuple[object, str]]):
        for key, value_pair in header_dict.items():
            header[key.upper()] = value_pair
        return header

    def plot_map(self, ax: Optional[Axes] = None, verbose: bool = False, show: bool = False, save: bool = False, **kwargs) -> None:
        """
        Plots an imshow of the `MuseMAP` data only if the `.name` of the instance is a key of `PLOT_CONFIG`

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The Axes to plot the map to. If None, a new figure will be created.
        verbose : bool, default: False
            Print verbose statements
        show : bool, default: False
            Display the figure output inline. Only displays the figure if `ax` is `None`.
        save : bool, default: True
            Saves the figure as a .pdf to the default figures directory. Only saves if `ax` is `None`
        **kwargs :
            xlabel_on : bool, default: True
                Label the x-axis
            ylabel_on : bool, default: True
                Label the y-axis
            facecolor : str, default: 'lightgray'
                The facecolor of the imshow
        """
        if self.name.upper() not in list(PLOT_CONFIG.keys()):
            util.sys_message(f"Plotting configuration not set for {self.name}. Skipping plot", verbose=verbose, status='WARN')
            return
        
        import matplotlib.pyplot as plt
        from src.nai_analysis.plotting.map_plotter import plot_map_data


        plotmap = np.ma.array(data=self.data, mask=self.mask.astype(bool))
        unique_data = util.get_unique_bin_values(self.data, self.spatial_bins, self.mask)
        config = PLOT_CONFIG[self.name.upper()]

        symmetric = config.get('symmetric', symmetric)
        imshow_kwargs: dict = config.get('imshow_kwargs', {})

        title =  config['title']
        cmap = imshow_kwargs.get('cmap', None)
        vmin = imshow_kwargs.get('vmin', np.percentile(unique_data, 5))
        vmax = imshow_kwargs.get('vmax', np.percentile(unique_data, 95))

        xlabel_on = kwargs.get('xlabel_on', True)
        ylabel_on = kwargs.get('ylabel_on', True)
        facecolor = kwargs.get('facecolor', 'lightgray')

        if symmetric:
            maxv = max(abs(vmin), vmax)
            vmin = -maxv
            vmax = maxv
        
        if save:
            directory = self.directory
            figdir = os.path.join(directory, 'figures')
            mapdir = os.path.join(figdir, 'maps')
            figname = f"{self.galname}-{self.bin_method}-{self.name}.pdf"
            util.check_filepath(mapdir, mkdir=True, verbose=verbose)
            
            output = os.path.join(mapdir, figname)
        else:
            output = None

        plot_map_data(
                plotmap,
                ax=ax,
                title=title,
                xlabel_on=xlabel_on,
                ylabel_on=ylabel_on,
                facecolor=facecolor,
                save_path=output,
                show=show,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap
            )

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