import os
from numpy.ma import MaskedArray
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Circle, Ellipse
from matplotlib.colors import ListedColormap

from src.nai_analysis.utils import util, defaults
from src.nai_analysis.plotting.plot_config import PLOT_CONFIG
from src.nai_analysis.plotting import plot_helpers
from src.nai_analysis.tools.apertures import Aperture
from src.nai_analysis.musedap_data import MuseDAPData

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.nai_analysis.maps.musemap import MuseMAP

import numpy as np

def plot_map_data(
        data: np.ndarray,
        ax: Optional[Axes] = None,
        apertures: Optional[list[Aperture]] = None,
        background_map: Optional[np.ndarray] = None,
        contour_map: Optional[np.ndarray] = None,
        pa: bool = False,
        title: str = '',
        xlabel_on: bool = True,
        ylabel_on: bool = True,
        facecolor: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = False,
        verbose: bool = False,
        **imshow_kwargs
) -> None:
    
    cmap = imshow_kwargs.get('cmap', None)
    vmin = imshow_kwargs.get('vmin', np.percentile(data[np.isfinite(data)], 5))
    vmax = imshow_kwargs.get('vmax', np.percentile(data[np.isfinite(data)], 95))

    extent = plot_helpers.muse_extent()

    if ax is None:
        fig, ax = plt.subplots()
        exit_early = False
    else:
        exit_early = True
    
    if background_map is not None:
        map_background(background_map, ax=ax)

    im = ax.imshow(data, origin = 'lower', extent=extent,
                cmap = cmap, vmin = vmin, vmax = vmax)

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

    if contour_map is not None:
        levels = np.percentile(contour_map, [16, 84])
        colors = ['k'] * len(levels)
        map_contour(contour_map, levels, ax=ax, smooth=True, colors=colors)

    if apertures is not None:
        for aper in apertures:
            draw_aperture(ax=ax, aper=aper, im_shape=data.shape)

    if pa:
        raise ValueError('position angle is broken')

    if exit_early:
        return
    
    if save_path is not None:
        path, fname = os.path.split(save_path)
        util.check_filepath(path, mkdir=False, verbose=True)
        fig.savefig(save_path, bbox_inches = 'tight')
        util.sys_message(f"Figure saved to {save_path}", verbose=verbose)
    if show:
        plt.show()
    else:
        plt.close()

def draw_aperture(ax: Axes, aper: Aperture, im_shape: Optional[tuple] = None, edgecolor: str ='k', zorder: int = 20) -> None:
    x, y = aper.x, aper.y

    if im_shape is not None:
        x0, x1, y0, y1 = plot_helpers.muse_extent()
        ny, nx = im_shape
        x_coords = np.linspace(x0, x1, nx)
        y_coords = np.linspace(y0, y1, ny)
        dx = abs(x1 - x0) / (nx - 1)
        dy = abs(y1 - y0) / (ny - 1)

        x = x_coords[x]
        y = y_coords[y]

    if aper.b is None:
        r = aper.a * dx
        patch = Circle((x, y), r, edgecolor=edgecolor, facecolor='none', zorder=zorder)
    else:
        w = 2 * aper.a * dx
        h = 2 * aper.b * dy
        patch = Ellipse((x, y), width=w, height=h, angle=aper.angle, edgecolor=edgecolor, facecolor='none', zorder=zorder)

    ax.add_patch(patch)
        
    ax.text(x, y, aper.label, ha='center', va='center', color=aper.labelcolor)

def map_background(
        data: np.ndarray[int],
        ax: Axes,
        extent: list = [32.4, -32.6,-32.4, 32.6],
        zorder: int = 0,
        color: str = 'lightgray'
    ) -> None:

    greymap = ListedColormap(['white', color])
    ax.imshow(data, cmap=greymap, vmin=0, vmax=1, origin='lower', extent=extent, zorder=zorder)

def map_contour(
        data: np.ndarray | MaskedArray, 
        levels: list,
        ax: Axes,
        smooth: bool = True,
        colors: Optional[list] = None,
        linewidths: Optional[list] = None,
        linestyles: Optional[list] = None,
        extent: list = [32.4, -32.6,-32.4, 32.6],
        zorder: int = 10
    ) -> None:

    if smooth:
        from scipy.ndimage import gaussian_filter
        data = gaussian_filter(data, sigma=2)

    ax.contour(data, levels = levels, colors = colors, linewidths = linewidths, linestyles = linestyles,
                origin='lower', extent=extent, zorder=zorder)

def map_filled_contour(
        data: np.ndarray | MaskedArray,
        levels: list,
        ax: Axes,
        extent: list = [32.4, -32.6,-32.4, 32.6],
        colors: Optional[list[str]] = None,
        alpha: float = 1.,
        zorder: int = 10
    ) -> None:
    ny, nx = data.shape
    x0, x1, y0, y1 = extent
    ys = np.linspace(y0, y1, ny); xs = np.linspace(x0, x1, nx)
    X, Y = np.meshgrid(xs, ys, indexing='xy')

    ax.contourf(X, Y, data, levels=levels, colors=colors, alpha = alpha, origin='lower', extent=extent, zorder=zorder)

def position_angle(
        dap_data: MuseDAPData, 
        shape: tuple,
        ax: Axes, 
        extent: list = [32.4, -32.6,-32.4, 32.6],
        color: str = 'k',
        linewidth: float = 0.8,
        zorder: int = 10,
    ) -> None:
    x0, x1, y0, y1 = extent
    ny, nx = shape

    r = dap_data.r_eff
    rmap = dap_data.r_coords
    cen = np.unravel_index(np.argmin(rmap), rmap.shape)
    pix_y, pix_x = cen

    x = np.linspace(x0, x1, nx)
    y = np.linspace(y0, y1, ny)
    cenx = x[pix_x]
    ceny = y[pix_y]

    pa = np.deg2rad(dap_data.pa)
    dx = np.sin(pa)
    dy = np.cos(pa)

    ax.plot(
        [cenx - r*dx, cenx + r*dx],
        [ceny - r*dy, ceny + r*dy],
        color=color,
        linewidth=linewidth,
        zorder=zorder
    )

def plot_hist(
        muse_map: "MuseMAP",
        ax: Optional[Axes] = None,
        nbins: Optional[int] = 40,
        title: Optional[str] = None,
        xlabel_on: bool = True,
        ylabel_on: bool = True,
        show: bool = False,
        save: bool = True,
        verbose: bool = False,
        **hist_kwargs
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

def plot_map_grid(maps: dict[str, "MuseMAP"], grid_dims: tuple[int | None, int | None] = (None, None), 
                  directory: Optional[str] = None, figname: Optional[str] = None, show = False, save = True,
                  verbose = False):
    nrows, ncols = grid_dims

    if nrows is None and ncols is None:
        raise ValueError("At least one dimension must be specified")
    
    if save:
        if directory is None or figname is None:
            raise ValueError("Output directory and figure name must be specified if save is True")
    
    plt.style.use(defaults.matplotlib_rc())
    
    nplots = len(maps)

    ncols = int(np.ceil(nplots / nrows)) if ncols is None else ncols
    nrows = int(np.ceil(nplots / ncols)) if nrows is None else nrows

    # Setup figure and GridSpec
    base_w, base_h = plt.rcParams["figure.figsize"]
    fig = plt.figure(figsize=(ncols*base_w, nrows*base_h))
    gs = GridSpec(nrows, ncols, figure=fig, hspace=0, wspace=0.)

    # Create the nrow x ncol axes
    axes = []
    for i in range(nrows):
        for j in range(ncols):
            ax = fig.add_subplot(gs[i, j])
            ax.set_aspect('equal')  # Keep square axes
            axes.append(ax)

    for idx, (ax, key) in enumerate(zip(axes, maps.keys())):
        plot_map(maps[key], ax=ax, xlabel_on=False, ylabel_on=False, show=False, save=False)

    for ax in axes[nplots:]:
        ax.axis('off')

    for i, ax in enumerate(axes):
        row, col = divmod(i, ncols)
        if row < nrows-1:  # Hide x ticks except for bottom row
            ax.set_xticklabels([])
        if col > 0:  # Hide y ticks except for left column
            ax.set_yticklabels([])

    plot_helpers.gs_group_label(gs=gs, fig=fig, xlabel=r'$\Delta \alpha$ (arcsec)', ylabel=r'$\Delta \delta$ (arcsec)')
    plot_helpers.panel_label(axes=axes)

    if save:
        name, ext = os.path.splitext(figname)
        if not ext:
            figname = f"{name}.pdf"
        
        util.check_filepath(directory, mkdir=False)
        output = os.path.join(directory, figname)
        plt.savefig(output, bbox_inches='tight')
        util.sys_message(f"Map grid saved to {output}", verbose=verbose)

    if show:
        plt.show()
    else:
        plt.close()