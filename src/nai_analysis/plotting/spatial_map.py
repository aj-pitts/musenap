from typing import Optional, Union
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, ListedColormap
from matplotlib.patches import Circle, Ellipse
from src.nai_analysis.musedap_data import MuseDAPData
import numpy as np
from numpy.ma import MaskedArray
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import os
from src.nai_analysis.utils import util
from src.nai_analysis.plotting.plot_helpers import muse_extent
from src.nai_analysis.tools.apertures import Aperture

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

#         x = np.linspace(32.4, -32.6, nx)
#         y = np.linspace(-32.4, 32.6, ny)
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

def draw_aperture(ax: Axes, aper: Aperture, im_shape: Optional[tuple] = None, edgecolor: str ='k', zorder: int = 20) -> None:
    x, y = aper.x, aper.y

    if im_shape is not None:
        x0, x1, y0, y1 = muse_extent()
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


# def plot_spatial_map(
#         muse_map: MuseMAP,
#         ax: Axes,
#         rotation_contours: bool = False,
#         position_angle: bool = False,
#         r_eff: Union[bool, float] = False,
#         symmetric: bool = False,
#         title: Optional[str] = None,
#         vmin: Optional[float] = None,
#         vmax: Optional[float] = None,
#         facecolor: Union[str, None] = 'lightgray',
#         cmap: Optional[str | Colormap] = None,
#     ) -> None:
#     """
#     Plots an `matplotlib.pyplot.imshow` of the MuseMAP data.
#     """
#     data = muse_map.data
#     mask = muse_map.mask.astype(bool)

#     ny, nx = data.shape 

#     plotmap = np.ma.array(data=data, mask=mask)
#     dap_data = MuseDAPData.from_name(muse_map.galname, muse_map.bin_method)

#     vmin =  np.percentile(plotmap, 5) if vmin is None else vmin
#     vmax = np.percentile(plotmap, 95) if vmax is None else vmax
    
#     if symmetric:
#         maxv = max(abs(vmin), abs(vmax))
#         vmin = -maxv
#         vmax = maxv
        
#     if r_eff:
#         if isinstance(r_eff, bool):
#             r_eff = 1
#         r_map = dap_data.r_coords
#         #rmap = np.ma.array(data=r_map, mask=r_map>r_eff)
#         rmap = np.copy(r_map)
#         w = r_map>r_eff
#         rmap[w] = 0; rmap[~w] = 1
#         greymap = ListedColormap(['white', 'lightgray'])
#         ax.imshow(rmap, cmap=greymap, vmin=0, vmax=1, origin='lower', extent=[32.4, -32.6,-32.4, 32.6])

#     im = ax.imshow(plotmap, origin = 'lower', extent=[32.4, -32.6,-32.4, 32.6],
#                     cmap = cmap, vmin = vmin, vmax = vmax)
    
#     if position_angle:
#         r = dap_data.r_eff
#         rmap = dap_data.r_coords
#         cen = np.unravel_index(np.argmin(rmap), rmap.shape)
#         cen_y, cen_x = cen

#         x = np.linspace(32.4, -32.6, nx)
#         y = np.linspace(-32.4, 32.6, ny)
#         x0 = x[cen_x]
#         y0 = y[cen_y]

#         pa = np.deg2rad(dap_data.pa)
#         dx = np.sin(pa)
#         dy = np.cos(pa)

#         ax.plot(
#             [x0 - r*dx, x0 + r*dx],
#             [y0 - r*dy, y0 + r*dy],
#             color='k',
#             linewidth=.8
#         )

#     if rotation_contours:
#         emline_vel = dap_data.get_map_data("EMLINE_GVEL")
#         r_map = dap_data.r_coords
#         gv = np.copy(emline_vel.data[23])
#         gv[emline_vel.mask[23].astype(bool)] = np.nan
#         gv[rmap>1] = np.nan

#         from scipy.ndimage import gaussian_filter
#         gv_smooth = gaussian_filter(gv, sigma=2)
#         s = np.nanstd(gv_smooth)
#         ax.contour(gv_smooth, levels = [-2*s, -s, 0, s, 2*s], colors=['k', 'k', 'k', 'k', 'k'], linewidths = [0.6,0.6, 1.2, 0.7,0.7], linestyles=['--', '--', '-', '-', '-'],
#                    origin='lower', extent=[32.4, -32.6,-32.4, 32.6])


#     ax.set_xlabel(r'$\Delta \alpha$ (arcsec)')
#     ax.set_ylabel(r'$\Delta \delta$ (arcsec)')
#     if facecolor is not None:
#         ax.set_facecolor(facecolor)