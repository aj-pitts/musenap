from typing import Optional, Union
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap

import matplotlib.pyplot as plt
import string

import numpy as np
import seaborn as sns

measurement_labels = {
    "V_CEN":{
        'value':r"$v_{\mathrm{cen}}$",
        'error':r"$\sigma_{v_{\mathrm{cen}}}$",
        'units':r"$\mathrm{\left( km\ s^{-1} \right)}$",
        'units2':r"$\mathrm{km\ s^{-1}}$"
    },
    "SNR_NAI":{
        'value':r"$S/N_{\mathrm{Na\ I}}$",
        'error':'',
        'units':'',
        'units2':''
    },
    "SFRSD":{
        'value':r"$\mathrm{log}\ \Sigma_{\mathrm{SFR}}$",
        'error':r"$\mathrm{log}\ \sigma_{\Sigma_{\mathrm{SFR}}}$",
        'units':r"$\mathrm{\left( M_{\odot}\ yr^{-1}\ kpc^{-2}\ spaxel^{-1} \right)}$",
        'units2':r"$\mathrm{M_{\odot}\ yr^{-1}\ kpc^{-2}\ spaxel^{-1})}$"
    },
    "WEQ_NAI":{
        'value':r"$\mathrm{EW_{Na\ I}}$",
        'error':r"$\mathrm{\sigma_{EW_{Na\ I}}}$",
        'units':r"$\mathrm{\left( \AA \right)}$",
        'units2':r"$\mathrm{\AA}$"
    }
}

def figure_filename(galname: str, bin_method: str, figure_name: str):
    return f"{galname}-{bin_method}-{figure_name}.pdf"

def gs_group_label(gs: GridSpec, fig: Figure, 
                   xlabel: Optional[str] = None, ylabel: Optional[str] = None, title: Optional[str] = None,
                   xfontsize: Optional[Union[float, int, str]] = None,
                   yfontsize: Optional[Union[float, int, str]] = None,
                   titlesize: Optional[Union[float, int, str]] = None,
                   xlabel_pad: float = 15,
                   ylabel_pad: float = 35):

    xfontsize = plt.rcParams['font.size'] if xfontsize is None else xfontsize
    yfontsize = plt.rcParams['font.size'] if yfontsize is None else yfontsize

    ax_group = fig.add_subplot(gs[:,:])
    ax_group.set_xlabel(xlabel, fontsize = xfontsize, labelpad=xlabel_pad)
    ax_group.set_ylabel(ylabel, fontsize = yfontsize, labelpad=ylabel_pad)
    ax_group.set_title(title, fontsize=titlesize)
    ax_group.tick_params(which = 'both', labelcolor='none', top=False, bottom=False, 
                        left=False, right=False)
    
    ax_group.set_frame_on(False)
    ax_group.patch.set_alpha(0)
    ax_group.set_zorder(-1)
    ax_group.set_xticks([])
    ax_group.set_yticks([])
    for spine in ax_group.spines.values():
        spine.set_visible(False)


def panel_label(axes: list[Axes], color: str = 'k', loc: tuple = (0.075, 0.9), fontsize: float = 10) -> None:
    """Labels each panel in a set of Axes by an alphabet character"""
    alphabet = list(string.ascii_lowercase)

    for idx, ax in enumerate(axes):
        char = alphabet[idx]
        ax.text(loc[0], loc[1], f'({char})', fontsize=fontsize, transform = ax.transAxes, color=color)


def setup_figure(
        nrows: int, ncols: int, 
        hspace: float = 0.1, wspace: float = 0.1,
        height_ratios: Optional[list] = None,
        width_ratios: Optional[list] = None,
        fig_width_scalar: float = 1,
        fig_height_scalar: float = 1
    ) -> tuple[Figure, GridSpec]:
    """Sets up a set of `nrows` and `ncols` subplots subplots using `GridSpec`"""

    base_w, base_h = plt.rcParams['figure.figsize']
    fig = plt.figure(figsize=(base_w * ncols * fig_width_scalar, base_h * nrows * fig_height_scalar))
    gs = GridSpec(nrows, ncols, figure=fig, wspace=wspace, hspace=hspace, height_ratios=height_ratios, width_ratios=width_ratios)

    return fig, gs

def muse_extent() -> list:
    """Returns a list of the MUSE offset from the central spaxel in arcsec order of RA0, RA1, Dec0, Dec1"""
    return [32.4, -32.6,-32.4, 32.6]


def median_error(
        ax: Axes, xerr: Optional[float] = None, yerr: Optional[float] = None,
        color: str = 'k',
        elinewidth: float = 1,
        capsize: float = 2,
        loc: tuple = (0.9, 0.1)
    ) -> None:
    """Plot a marker showing the median error on the input ax"""

    if xerr is None and yerr is None:
        raise ValueError("Both xerr and yerr cannot be none")
    
    x_data, y_data = ax.transData.inverted().transform(
    ax.transAxes.transform(loc)
    )
    print(x_data, y_data)
    ax.errorbar(x_data, y_data, xerr=xerr, yerr=yerr, capsize=capsize, elinewidth=elinewidth, ecolor=color, fmt='none')

def rescale_8bit(image: np.ndarray, cmin: float = 0, cmax: Optional[float] = None, scale: str = 'linear') -> np.ndarray[np.uint8]:
    cmax = image.max() if cmax is None else cmax

    rescale = (image - cmin) / (cmax - cmin)
    if scale == 'linear':
        scaledim = 255 * rescale
    elif scale == 'sqrt':
        scaledim = 255 * np.sqrt( rescale )

    scaledim[scaledim < 0] = 0
    scaledim[scaledim > 255] = 255
    return scaledim.astype(np.uint8)

def seaborn_palette(name: str, ncolors: int = 256, cmap:bool = True) -> ListedColormap | list[tuple]:
    return sns.color_palette(name, n_colors=ncolors, as_cmap=cmap)