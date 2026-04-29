import numpy as np
from typing import Union, Optional
import matplotlib.colors as mcolors
from matplotlib.axes import Axes
from src.nai_analysis.plotting.plot_helpers import muse_extent
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

def bpt_map_plot(
        data: Union[np.ndarray, np.ma.MaskedArray], 
        ax: Optional[Axes] = None, 
        colors = Optional[list[str]] = None
    ) -> None:

    colors = ['black', 'skyblue', 'fuchsia', 'gold', 'crimson'] if colors is None else colors
    cmap = mcolors.ListedColormap(colors)

    boundaries = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)

    extent=muse_extent()
    im = ax.imshow(data, cmap=cmap, norm=norm, origin='lower',extent=extent)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad=0.01)
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal', ticks=[0, 1, 2, 3, 4])
    cbar.ax.set_xticklabels(['ambiguous', 'star-forming', 'composite', 'seyfert', 'liner']);
