import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import cmasher as cmr
import os

from src.nai_analysis.utils import util, defaults
from src.nai_analysis.plotting.plot_config import PLOT_CONFIG

from typing import Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from src.nai_analysis.maps.musemap import MuseMAP

import numpy as np

def plot_map(
        muse_map: "MuseMAP",
        ax: Optional[Axes] = None,
        title: Optional[str] = None,
        facecolor: Union[str, None] = 'lightgray',
        xlabel_on: Optional[bool] = True,
        ylabel_on: Optional[bool] = True,
        show: Optional[bool] = False,
        save: Optional[bool] = True,
        verbose: Optional[bool] = False,
        **imshow_kwargs: Optional[dict] 
    ) -> None:
    """
    Plots an `matplotlib.pyplot.imshow` of the MuseMAP data.
    
    Creates an imshow of the MuseMAP data which is masked by MuseMAP.mask; optionally specify
    the `matplotlib.axes.Axes`, whether to display and/or save the figure, and various args
    including the `imshow` kwargs.
    """
    plt.style.use(defaults.matplotlib_rc())

    data = muse_map.data
    mask = muse_map.mask
    plotmap = np.ma.array(data=data, mask=mask.astype(bool))

    unique_data = util.get_unique_bin_values(data, muse_map.spatial_bins, mask)

    config = PLOT_CONFIG[muse_map.name.upper()]

    title = title if title is not None else config['title']
    cmap = imshow_kwargs.get('cmap', config['cmap'])
    vmin = imshow_kwargs.get('vmin', np.percentile(unique_data, [5]))
    vmax = imshow_kwargs.get('vmax', np.percentile(unique_data, [95]))

    if ax is None:
        fig, ax = plt.subplots()

    im = ax.imshow(plotmap, origin = 'lower', extent=[32.4, -32.6,-32.4, 32.6],
                    cmap = cmap, vmin = vmin, vmax = vmax,
                    **imshow_kwargs)
    if xlabel_on:
        ax.set_xlabel(r'$\Delta \alpha$ (arcsec)')
    if ylabel_on:
        ax.set_ylabel(r'$\Delta \delta$ (arcsec)')
    if facecolor is not None:
        ax.set_facecolor(facecolor)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad=0.01)

    cbar = plt.colorbar(im, cax=cax, orientation = 'horizontal')
    cbar.set_label(title, labelpad=-55)
    cax.xaxis.set_ticks_position('top')
    if save:
        directory = muse_map.directory
        figdir = os.path.join(directory, 'figures')
        mapdir = os.path.join(figdir, 'maps')
        figname = f"{muse_map.galname}-{muse_map.bin_method}-{muse_map.name}.pdf"
        util.check_filepath(mapdir, mkdir=True, verbose=verbose)
        
        output = os.path.join(mapdir, figname)
        plt.savefig(output, bbox_inches = 'tight')
        util.sys_message(f"{muse_map.name} MAP plot saved to {output}", verbose=verbose)
    if show:
        plt.show()
    else:
        plt.close()

def plot_hist(
        muse_map: "MuseMAP",
        ax: Optional[Axes] = None,
        nbins: Optional[int] = 40,
        title: Optional[str] = None,
        xlabel_on: Optional[bool] = True,
        ylabel_on: Optional[bool] = True,
        show: Optional[bool] = False,
        save: Optional[bool] = True,
        verbose: Optional[bool] = False,
        **hist_kwargs: Optional[dict] 
        ) -> None:

    plt.style.use(defaults.matplotlib_rc())
    
    data = muse_map.data
    mask = muse_map.mask

    unique_data = util.get_unique_bin_values(data, muse_map.spatial_bins, mask)
    histbins = np.linspace(unique_data.min(), unique_data.max(), nbins)

    if ax is None:
        fig, ax = plt.subplots()
    
    config = PLOT_CONFIG[muse_map.name.upper()]

    color = hist_kwargs.get('color', 'k')
    title = title if title is not None else config['title']

    ax.hist(unique_data, bins=histbins, color=color, **hist_kwargs)
    if xlabel_on:
        ax.set_xlabel(title)
    if ylabel_on:
        ax.set_ylabel(r"$N_{\mathrm{bins}}$")

    if save:
        directory = muse_map.directory
        figname = f"{muse_map.galname}-{muse_map.bin_method}-{muse_map.name}-HIST.pdf"
        figdir = os.path.join(directory, 'figures')
        histdir = os.path.join(figdir, 'hists')
        util.check_filepath(histdir, mkdir=True, verbose=verbose)

        output = os.path.join(histdir, figname)
        plt.savefig(output, bbox_inches = 'tight')
        util.sys_message(f"{muse_map.name} HIST plot saved to {output}", verbose=verbose)
    if show:
        plt.show()
    else:
        plt.close()