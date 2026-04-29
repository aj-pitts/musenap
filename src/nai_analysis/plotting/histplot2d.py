from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize, LogNorm
import cmasher as cmr
import seaborn as sns

from typing import Optional

import numpy as np



def plot_hist2d(
        x_data: np.ndarray,
        y_data: np.ndarray,
        ax: Axes,
        bins: int = 50,
        cmap: str | Colormap = 'Greys',
        norm: Optional[str | Normalize] = LogNorm(),
        contour_mask: Optional[np.ndarray] = None,
        contour_cmap: Colormap = cmr.nuclear,
        contour_levels: list = [0.125, 0.25, 0.5, 0.8, 0.95, 0.995],
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
    ) -> None:

    ax.hist2d(x_data, y_data, bins=bins, cmap=cmap, norm=norm)
    if contour_mask is not None:
        x_masked = x_data[contour_mask].byteswap().newbyteorder()
        y_masked = y_data[contour_mask].byteswap().newbyteorder()
        sns.kdeplot(x=x_masked, y=y_masked, levels=contour_levels, cmap=contour_cmap, ax=ax)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xticklabels([])
    
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    else:
        ax.set_yticklabels([])
