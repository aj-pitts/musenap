from typing import Optional, TYPE_CHECKING
from src.nai_analysis.plotting.rgb_im import RGB_image
from src.nai_analysis.plotting.plot_helpers import seaborn_palette, setup_figure, muse_extent, gs_group_label
from src.nai_analysis.utils import util, defaults
import cmasher as cmr
import string
import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

if TYPE_CHECKING:
    from src.nai_analysis.musedap_data import MuseDAPData


def DAP_MAPS(dap_data: "MuseDAPData", show: bool = False, save: bool = True, verbose: bool = False) -> None:
    extent = muse_extent()

    rgb = RGB_image(dap_data.cube_path)
    snr = dap_data.get_map_data("BIN_SNR")
    radius = dap_data.r_coords
    chisq = dap_data.get_map_data("STELLAR_FOM")
    stellar_vel = dap_data.get_map_data("STELLAR_VEL")
    stellar_sigma = dap_data.get_map_data("STELLAR_SIGMA")
    ha = dap_data.get_emline("EMLINE_GFLUX", 'Ha-6564')
    hb = dap_data.get_emline("EMLINE_GFLUX", 'Hb-4862')

    plotdicts = {
        'RGB':dict(image = rgb, mask = None, cmap = None, vmin = None, vmax = None, v_str = r'$B, V, R$'),
        'RADIUS':dict(image = radius, mask = None, cmap = seaborn_palette('autumn_r'), vmin=0, vmax=1, v_str = r'$R / R_e$'),
        'SNR':dict(image = snr.data, mask = None, cmap = seaborn_palette('mako'), vmin=0, vmax=75, v_str = r'$S/N_g$'),
        'CHISQ':dict(image = chisq.data[2], mask = None, cmap = seaborn_palette('binary_r'), vmin = 0, vmax = 2, v_str = r'$\chi^2_{\nu}$'),

        'STELLAR_VEL':dict(image = stellar_vel.data, mask = stellar_vel.mask.astype(bool), cmap = seaborn_palette('seismic'), vmin = -250, vmax = 250, v_str = r'$V_{\star}\ \left( \mathrm{km\ s^{-1}} \right)$'),
        'STELLAR_SIG':dict(image = stellar_sigma.data, mask = stellar_sigma.mask.astype(bool), cmap = cmr.ember, vmin = 25, vmax = 100, v_str = r'$\sigma_{\star}\ \left( \mathrm{km\ s^{-1}} \right)$'),
        'H_alpha':dict(image = ha.data, mask = ha.mask.astype(bool), cmap = seaborn_palette('bone'), vmin = 0, vmax = 10, v_str = r'$F_{\mathrm{H}\alpha}$'), # \left( \mathrm{10^{-17}\ erg\ s^{-1}\ cm^{-2}\ spaxel^{-1}} \right)
        'H_beta':dict(image = hb.data, mask = hb.mask.astype(bool), cmap = seaborn_palette('bone'), vmin = 0, vmax = 2, v_str = r'$F_{\mathrm{H}\beta}$'), # \left( \mathrm{10^{-17}\ erg\ s^{-1}\ cm^{-2}\ spaxel^{-1}} \right)
    }

    alphabet = list(string.ascii_lowercase)

    nrow = 2
    ncol = 4

    fig, gs = setup_figure(nrow, ncol, hspace=.375, wspace=-.35)

    axes: list[Axes] = []
    for i in range(nrow):
        for j in range(ncol):
            ax = fig.add_subplot(gs[i,j])
            #ax.set_aspect('equal')
            axes.append(ax)

    for ax, key, plot_dict, char in zip(axes, plotdicts.keys(), plotdicts.values(), alphabet):
        #print(key)
        plotmap = plot_dict['image']
        plotmask = plot_dict['mask']

        if key != 'RGB':
            plotmap[plotmap == 0] = np.nan
            if plotmask is not None:
                plotmap[plotmask] = np.nan

        im = ax.imshow(plotmap, origin='lower', vmin=plot_dict['vmin'], vmax=plot_dict['vmax'], cmap=plot_dict['cmap'],
                       extent = extent)
        ax.set_facecolor('lightgray')

        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.01)
        value_string = plot_dict['v_str']
        if key == 'RGB':
            dummy_data = np.zeros(plotmap.shape[:2])
            dummy_cmap = mcolors.ListedColormap(['none'])
            dummy_norm = mcolors.Normalize(vmin=0, vmax=1)
            cbar = fig.colorbar(plt.imshow(dummy_data, cmap=dummy_cmap, norm=dummy_norm),
                                cax=cax, orientation='horizontal')
            cbar.ax.set_facecolor('white')
            cbar.set_ticks([])
            cbar.set_label(value_string, labelpad=-45)
            cbar.outline.set_visible(False)
            cbar.ax.patch.set_alpha(0)
        else:
            def smart_int_formatter(x, pos):
                if abs(x - round(x)) < 1e-2:  # very close to integer
                    return f'{int(round(x))}'
                else:
                    return ''  # empty string means no label
                
            cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
            cbar.set_label(value_string, labelpad=-45)
            cax.xaxis.set_ticks_position('top')
            cbar.ax.xaxis.set_major_formatter(FuncFormatter(smart_int_formatter))


        ax.text(0.075, 0.9, f'({char})', fontsize=10, transform=ax.transAxes, color='white')

    for i, ax in enumerate(axes):
        row, col = divmod(i, ncol)
        if row < nrow-1:  # Hide x ticks except for bottom row
            ax.set_xticklabels([])
        if col > 0:  # Hide y ticks except for left column
            ax.set_yticklabels([])

    # Axis labels for the whole figure
    # fig.text(0.5, 0.025, r'$\Delta \alpha$ (arcsec)', ha='center', va='center', fontsize = 18)
    # fig.text(0.12, 0.5, r'$\Delta \delta$ (arcsec)', ha='center', va='center', rotation='vertical', fontsize = 18) 
    gs_group_label(gs, fig, xlabel=r'$\Delta \alpha$ (arcsec)', ylabel=r'$\Delta \delta$ (arcsec)',
                   xlabel_pad=30, ylabel_pad=0)

    if save:
        galaxy_dir = defaults.get_local_galaxy_dir(dap_data.galname, dap_data.bin_method, dap_data.analysisplan)
        figures_dir = os.path.join(galaxy_dir, 'figures')
        output_dir = os.path.join(figures_dir, 'maps')
        os.makedirs(output_dir, exist_ok=True)

        outfile = os.path.join(output_dir, f"{dap_data.galname}-{dap_data.bin_method}_DAPmaps.pdf")
        plt.savefig(outfile)
        util.sys_message(f"DAP map grid saved to: {outfile}", verbose=verbose)
    if show:
        plt.show()
    else:
        plt.close()