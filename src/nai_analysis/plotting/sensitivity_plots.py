import os
import matplotlib.pyplot as plt

from src.nai_analysis.utils import util
from src.nai_analysis.plotting.plot_config import PLOT_CONFIG
from src.nai_analysis.plotting import plot_helpers, histplot2d

from typing import TYPE_CHECKING, Optional

from src.nai_analysis.musenap_data import MuseNAPData

import numpy as np


def SNR_plots(muse_data: MuseNAPData, filename: Optional[str] = None, savedir: Optional[str] = None, show: bool = False, save: bool = True) -> None:
    vcenmap = muse_data.get_map("v_cen")
    vfrac = muse_data.get_map("v_frac")
    ewmap = muse_data.get_map("weq_abs_nai")
    snrmap = muse_data.get_map("snr_nai")
    sfrmap = muse_data.get_map("sfrsd")

    dap_data = muse_data.dap_data

    m = (ewmap.data < -0.5) | (ewmap.data > 2.5) | (snrmap.data > 125) | (sfrmap.data < -4.25) | (sfrmap.data > -1)
    mdict = util.get_unique_bin_values({'ew':ewmap.data, 'sfr':sfrmap.data, 'snr':snrmap.data, 'frac':vfrac.data, 'mask':vcenmap.mask}, dap_data.spatial_bins, mask=m)
    
    frac = mdict['frac']
    mask = mdict['mask']
    detect = (np.abs(frac) >= 0.95) & (~mask.astype(bool))

    fig, gs = plot_helpers.setup_figure(1, 2)
    ax_left = fig.add_subplot(gs[0,0])
    ax_right = fig.add_subplot(gs[0,1])


    histplot2d.plot_hist2d(mdict['ew'], mdict['snr'], ax=ax_left, contour_mask=detect,
                           xlabel=r"$\mathrm{EW_{Na\ I}\ \left( \AA \right)}$",
                           ylabel=r"$S/N$")
    histplot2d.plot_hist2d(mdict['sfr'], mdict['snr'], ax=ax_right, contour_mask=detect,
                           xlabel=r"$\mathrm{\Sigma_{SFR}\ \left( M_{\odot}\ yr^{-1}\ kpc^{-2}\ spaxel^{-1} \right)}$")
    
    if save:
        filename = f'{muse_data.galname}-{muse_data.bin_method}-SNR_EW_SFR.pdf' if filename is None else filename
        savedir = os.path.join(muse_data.figures_dir, 'results') if savedir is None else savedir
        fig.savefig(os.path.join(savedir, filename), bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()