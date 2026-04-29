import numpy as np
from astropy.io import fits
from astropy.io.fits import PrimaryHDU, ImageHDU, BinTableHDU
from astropy.io.fits.header import Header
import os
from typing import Union, TYPE_CHECKING, Optional

from src.nai_analysis.utils import defaults, util
from src.nai_analysis.musedap_data import MuseDAPData, DAPMap
from src.nai_analysis.plotting import plot_helpers

from matplotlib.patches import Circle

from src.nai_analysis.tools.apertures import Aperture

if TYPE_CHECKING:
    from src.nai_analysis.maps.musemap import MuseMAP

class MuseNAPData:
    def __init__(self, galaxy_name: str, binning_method: str, analysis_plans: Optional[str] = None, verbose: bool = False):
        analysis_plans = defaults.analysis_plans() if analysis_plans is None else analysis_plans

        self._validate(galaxy_name, binning_method, analysis_plans, verbose)

        self.galname = galaxy_name
        self.bin_method = binning_method
        self.analysisplan = analysis_plans
        self.verbose = verbose

        self.dap_data = MuseDAPData.from_name(self.galname, self.bin_method, verbose=False)

        self._initialize()

        self._data_cache: dict[str, np.ndarray] = {}
        self._map_cache: dict[str, "MuseMAP"] = {}


    @classmethod
    def from_DAP_data(cls, DAP_data: MuseDAPData):
        return cls(DAP_data.galname, DAP_data.bin_method, DAP_data.analysisplan, DAP_data.verbose)
    

    @staticmethod
    def _validate(galaxy_name: str, binning_method: str, analysis_plans: str, verbose: bool) -> None:
        for key, arg in {'galaxy_name':galaxy_name, 'binning_method':binning_method, 'analysis_plans':analysis_plans}.items():
            if not isinstance(arg, str):
                raise ValueError(f"Input for '{key}' must be a string: {arg}")
            
        if not isinstance(verbose, bool):
            raise ValueError(f"Input for 'verbose' must be a bool")
        

    def _initialize(self) -> None:
        util.sys_message(f"Initializing NAP data...", verbose=self.verbose)

        self.galaxy_dir = defaults.get_local_galaxy_dir(self.galname, self.bin_method)
        util.check_filepath(self.galaxy_dir, mkdir=True, verbose=self.verbose)

        self.figures_dir = os.path.join(self.galaxy_dir, 'figures')
        util.check_filepath(self.figures_dir, mkdir=True, verbose=self.verbose)

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

        # if self.verbose and self.fileready:
        #     print(f"{self.galname} {self.bin_method} NAP MAPS:")
        #     self.print_fileinfo()


    def _get_hdu_data_or_none(self, hdu_name: str) -> Union[np.ndarray, None]:
        try:
            with fits.open(self.filepath) as hdul:
                return hdul[hdu_name.upper()].data.copy()
        except KeyError:
            return None


    def print_fileinfo(self) -> None:
        """Prints the info of the `filepath` FITS HDU List"""
        with fits.open(self.filepath) as hdul:
            print(hdul.info())

    def get_hdu(self, hdu_name: str) -> Union[PrimaryHDU, ImageHDU, BinTableHDU]:
        """Returns the FITS HDU specified by `hdu_name`"""
        return self._hdul[hdu_name.upper()]
        
    def get_data(self, hdu_name: str) -> np.ndarray:
        """Returns the data of the FITS HDU specified by `hdu_name`"""
        if hdu_name in self._data_cache:
            return self._data_cache[hdu_name]
        
        data = self._hdul[hdu_name.upper()].data.copy()
        self._data_cache[hdu_name] = data
        return data
        
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
        from src.nai_analysis.maps.musemap import MuseMAP

        if hdu_name in self._map_cache:
            return self._map_cache[hdu_name]

        muse_map = MuseMAP.from_hdu(hdu_name=hdu_name, galname=self.galname, bin_method=self.bin_method, bin_map=self.dap_data.spatial_bins)

        self._map_cache[hdu_name] = muse_map
        return muse_map

    def plot_grid(self, hdu_names: Optional[list[str]] = None, grid_dims: tuple[int | None, int | None] = (None,4), show = False, save = True) -> None:
        """
        Creates a grid of MAP plots using the built in map plotter of `MuseMAP`. Optionally specify 
        the maps to be plot by inputting their hdu_names. Optionally set the dimensions of the grid
        plot by inputting `grid_dims`. 
        """
        from src.nai_analysis.plotting.map_plotter import plot_map_grid

        hdu_names = ['SNR_NAI', 'WEQ_NAI', 'SFRSD', 'V_CEN'] if hdu_names is None else hdu_names

        maps_to_plot = {name:self.get_map(name) for name in hdu_names}

        plot_map_grid(maps_to_plot, grid_dims=grid_dims, directory=self.figures_dir, figname=f"{self.galname}-{self.bin_method}-MAPGRID.pdf",
                      show=show, save=save)
        
        
    def plot_map_grid(
            self, 
            hdu_names: Optional[list[str]] = None, 
            grid_dims: tuple[Optional[int], Optional[int]] = (None, 4), 
            show: bool = False, 
            save: bool = True
        ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from src.nai_analysis.plotting.map_plotter import plot_map_data
        from src.nai_analysis.plotting import plot_helpers

        nrows, ncols = grid_dims
        if nrows is None and ncols is None:
            raise ValueError("At least one dimension must be specified")
        
        hdu_names = ['SNR_NAI', 'WEQ_NAI', 'SFRSD', 'BPT', 'V_GAS', 'V_CEN_SYS', 'V_CEN'] if hdu_names is None else hdu_names

        nplots = len(hdu_names)
        ncols = int(np.ceil(nplots / nrows)) if ncols is None else ncols
        nrows = int(np.ceil(nplots / ncols)) if nrows is None else nrows

        base_w, base_h = plt.rcParams["figure.figsize"]
        fig = plt.figure(figsize=(ncols*base_w, nrows*base_h))
        gs = GridSpec(nrows, ncols, figure=fig, hspace=0, wspace=0.)

        axes = []
        for i in range(nrows):
            for j in range(ncols):
                ax = fig.add_subplot(gs[i, j])
                ax.set_aspect('equal')
                axes.append(ax)

        maps_to_plot = {}
        for name in hdu_names:
            if name == 'V_GAS':
                vgas = self.dap_data.get_emline('EMLINE_GVEL', 'Ha-6564')
                gasmap = np.ma.array(vgas.data, mask=vgas.mask.astype(bool))
                maps_to_plot = {'V_GAS':gasmap}
            muse_map = self.get_map(name)
            marray = np.ma.array(muse_map.data, mask=muse_map.mask.astype(bool))
            maps_to_plot[name] = marray

        for idx, (ax, key) in enumerate(zip(axes, maps_to_plot.keys())):
            row, col = divmod(idx, ncols)
            plot_map_data(maps_to_plot[key], ax=ax, xlabel_on=False, ylabel_on=False)
            if row < nrows - 1:
                ax.set_xticklabels([])
            if col > 0:
                ax.set_yticklabels([])

        plot_helpers.gs_group_label(gs=gs, fig=fig, xlabel=r'$\Delta \alpha$ (arcsec)', ylabel=r'$\Delta \delta$ (arcsec)')
        plot_helpers.panel_label(axes=axes)
        
        if save:
            figname = f"{self.galaxy_dir}-NAP-MAPS-GRID.pdf"
            outdir = os.path.join(self.figures_dir, 'maps')
            savepath = os.path.join(outdir, figname)
            fig.savefig(savepath, bbox_inches='tight')
            util.sys_message(f"MAP Grid saved to {savepath}", verbose=self.verbose)

        if show:
            plt.show()
        else:
            plt.close()

    def place_aperture(
            self, 
            aperture: Aperture,
            map_names: list[str] = ["SFRSD", "V_CEN"],
        ) -> dict:
        """Place one or more apertures on Muse MAPs and print statistics of each aperture region. `center` must be a tuple of (x,y) or (column,row) coordinates."""

        spatial_bins = self.dap_data.spatial_bins
        imy, imx = np.indices(spatial_bins.shape)

        if aperture.isolate != 'none':
            vcenmap = self.get_map("v_cen")
            vcen_mask = vcenmap.mask.astype(bool)
            if aperture.isolate == 'outflow':
                w = vcenmap.data < 0
            elif aperture.isolate == 'inflow':
                w = vcenmap.data > 0
            else:
                raise ValueError(f"isolate '{aperture.isolate}' not recognized")
            iso_mask = np.logical_and(~vcen_mask, w)
        else:
            iso_mask = np.ones_like(spatial_bins).astype(bool)

        masks = []
        for name in map_names:
            datamap = self.get_map(name)
            masks.append(datamap.mask.astype(bool))
        mask = np.logical_or(masks[0], masks[1])
            
        aper_info = {}
            
        x, y, a = aperture.x, aperture.y, aperture.a
        b = a if aperture.b is None else aperture.b
        theta = np.deg2rad(aperture.angle)
            
        dx = imx - x
        dy = imy - y

        x_rot = dx * np.cos(theta) + dy * np.sin(theta)
        y_rot = -dx * np.sin(theta) + dy * np.cos(theta)

        aper_mask = (x_rot / a)**2 + (y_rot / b)**2 <= 1
        binids = np.unique(spatial_bins[aper_mask])

        aper_info['bins'] = binids
        for name in map_names:
            datamap = self.get_map(name)
            m = ~mask & aper_mask & iso_mask
            data = datamap.data[m]

            aper_info[name] = data

        return aper_info