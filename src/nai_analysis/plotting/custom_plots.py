from src.nai_analysis.plotting import plot_helpers
from src.nai_analysis.plotting.plot_config import PLOT_CONFIG
from src.nai_analysis.plotting import plot_helpers, histplot2d
from src.nai_analysis.plotting.spatial_map import draw_aperture
from src.nai_analysis.utils import util
from src.nai_analysis.tools.apertures import Aperture
from typing import TYPE_CHECKING, Optional
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import pearsonr
from matplotlib.colors import LinearSegmentedColormap
from src.nai_analysis.plotting.spatial_map import map_background, map_contour, map_filled_contour, position_angle
from matplotlib.gridspec import GridSpec

if TYPE_CHECKING:
    from src.nai_analysis.musenap_data import MuseNAPData

def vcen_spatialmap(muse_data: "MuseNAPData", filename: Optional[str] = None, savedir: Optional[str] = None, show: bool = False, save: bool = True) -> None:

    cmap = LinearSegmentedColormap.from_list('cmap', ["#72C8FD", "#FFFFFF", '#FD7272'])

    vcen_map = muse_data.get_map("v_cen")
    sfrsd_map = muse_data.get_map("sfrsd")
    ewmap = muse_data.get_map("weq_abs_nai")
    snr_map = muse_data.get_map("snr_nai")

    fig, gs = plot_helpers.setup_figure(1,2)
    ax_left = fig.add_subplot(gs[0,0])
    ax_right = fig.add_subplot(gs[0,1])

    #config = PLOT_CONFIG['V_CEN']
    extent = plot_helpers.muse_extent()

    dap_data = muse_data.dap_data
    r_coords = dap_data.r_coords
    r_lim = 1. #R_e
    #bg = (r_coords < r_lim).astype(int)
    bg = (snr_map.data > 30).astype(int)

    emline_vel = dap_data.get_map_data("EMLINE_GVEL")

    w = (r_coords > r_lim) & (emline_vel.mask[23].astype(bool))
    gas_velocity = np.copy(emline_vel.data[23])
    gas_velocity[w] = np.nan
    s = np.nanstd(gas_velocity)

    data = np.ma.array(vcen_map.data, mask=vcen_map.mask.astype(bool))
    ax_left.imshow(data, origin='lower', vmin=-1, vmax=1, cmap=cmap, extent=extent, zorder=5)
    ax_right.imshow(data, origin='lower', vmin=-1, vmax=1, cmap=cmap, extent=extent, zorder=5)


    ax_right.set_yticklabels([])
    plot_helpers.gs_group_label(gs, fig, xlabel=r'$\Delta \alpha$ (arcsec)', ylabel=r'$\Delta \delta$ (arcsec)')


    for ax, Map in zip([ax_left, ax_right], [sfrsd_map, ewmap]):
        map_background(bg, ax=ax)
        position_angle(dap_data, vcen_map.data.shape, ax=ax)
        map_contour(gas_velocity, levels=[-s, 0, s], ax=ax, 
                    colors=['k']*3, linestyles=['--', '-', '-'], linewidths=[0.6, 1., 0.6])
        
        map_data = np.ma.array(Map.data, mask=Map.mask)
        p1, p2 = np.percentile(map_data, [85, 100])

        map_filled_contour(map_data, levels=[p1, p2], ax=ax, colors='white', alpha=0.7, zorder=6)
    
    if save:
        filename = f'{muse_data.galname}-{muse_data.bin_method}-VCEN_SPATIAL.pdf' if filename is None else filename
        savedir = os.path.join(muse_data.figures_dir, 'custom_plots') if savedir is None else savedir
        util.check_filepath(savedir)
        fig.savefig(os.path.join(savedir, filename), bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

def vcen_spatialmap_test(muse_data: "MuseNAPData", filename: Optional[str] = None, savedir: Optional[str] = None, show: bool = False, save: bool = True) -> None:

    cmap = LinearSegmentedColormap.from_list('cmap', ["#72C8FD", "#FFFFFF", '#FD7272'])

    vcen_map = muse_data.get_map("v_cen")
    sfrsd_map = muse_data.get_map("sfrsd")
    ewmap = muse_data.get_map("weq_abs_nai")
    snr_map = muse_data.get_map("snr_nai")

    fig, gs = plot_helpers.setup_figure(1,1)
    ax = fig.add_subplot(gs[0,0])

    #config = PLOT_CONFIG['V_CEN']
    extent = plot_helpers.muse_extent()

    dap_data = muse_data.dap_data
    r_coords = dap_data.r_coords
    r_lim = 1. #R_e
    #bg = (r_coords < r_lim).astype(int)
    bg = (snr_map.data > 30).astype(int)

    emline_vel = dap_data.get_map_data("EMLINE_GVEL")

    w = (r_coords > r_lim) & (emline_vel.mask[23].astype(bool))
    gas_velocity = np.copy(emline_vel.data[23])
    gas_velocity[w] = np.nan
    s = np.nanstd(gas_velocity)

    data = np.ma.array(vcen_map.data, mask=vcen_map.mask.astype(bool))
    ax.imshow(data, origin='lower', vmin=-1, vmax=1, cmap=cmap, extent=extent, zorder=5)

    plot_helpers.gs_group_label(gs, fig, xlabel=r'$\Delta \alpha$ (arcsec)', ylabel=r'$\Delta \delta$ (arcsec)')

    map_background(bg, ax=ax)
    position_angle(dap_data, vcen_map.data.shape, ax=ax)
    map_contour(gas_velocity, levels=[-s, 0, s], ax=ax, 
                colors=['k']*3, linestyles=['--', '-', '-'], linewidths=[0.6, 1.4, 0.6])

        
    sfr_data = np.ma.array(sfrsd_map.data, mask=sfrsd_map.mask)
    p1, p2 = np.percentile(sfr_data, [85, 100])

    map_filled_contour(sfr_data, levels=[p1, p2], ax=ax, colors='white', alpha=0.7, zorder=6)

    ew_data = np.ma.array(ewmap.data, mask=ewmap.mask)
    p1, p2 = np.percentile(ew_data, [85, 100])
    # map_contour(ew_data, levels=[p1], ax=ax,
    #             colors=['dimgray'], linewidths=[0.5])
    map_filled_contour(ew_data, levels=[p1, p2], ax=ax, colors='green', alpha=0.3, zorder=6)

    if save:
        filename = f'{muse_data.galname}-{muse_data.bin_method}-VCEN_SPATIAL.pdf' if filename is None else filename
        savedir = os.path.join(muse_data.figures_dir, 'custom_plots') if savedir is None else savedir
        util.check_filepath(savedir)
        fig.savefig(os.path.join(savedir, filename), bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

def SNR_plots(muse_data: "MuseNAPData", filename: Optional[str] = None, savedir: Optional[str] = None, show: bool = False, save: bool = True) -> None:
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
        savedir = os.path.join(muse_data.figures_dir, 'custom_plots') if savedir is None else savedir
        util.check_filepath(savedir)
        fig.savefig(os.path.join(savedir, filename), bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

def aperture_maps(
        muse_data: "MuseNAPData",
        apertures: Aperture | list[Aperture],
        map_names: list[str] = ["V_CEN", "SFRSD"],
        filename: Optional[str] = None,
        savedir: Optional[str] = None,
        show: bool = False, 
        save: bool = True
    ) -> None:
    if len(map_names) != 2:
        raise ValueError("aperture maps only supports two map inputs")

    if not isinstance(apertures, list):
        apertures = [apertures]

    fig, gs_main = plot_helpers.setup_figure(1, 2, fig_width_scalar=2.5, fig_height_scalar=1.5, wspace=0.15)
    
    nrows = 1
    ncols = len(map_names)

    gs_maps = gs_main[0,0].subgridspec(nrows, ncols, wspace=0.1)

    for i, mapname in enumerate(map_names):
        ax = fig.add_subplot(gs_maps[0,i])
        datamap = muse_data.get_map(mapname)

        datamap.map_plot(ax=ax)
        ax.set_xlabel('')
        ax.set_ylabel('')

        if i>0:
            ax.set_yticklabels([])
        
        for j, aper in enumerate(apertures):
            draw_aperture(ax=ax, aper=aper, im_shape=datamap.data.shape)

    plot_helpers.gs_group_label(gs_maps, fig, xlabel=r'$\Delta \alpha$ (arcsec)', ylabel=r'$\Delta \delta$ (arcsec)')

    ncols = 3 if len(apertures) > 3 else len(apertures)
    nrows = np.ceil(len(apertures) / ncols).astype(int)

    gs_plots = gs_main[0,1].subgridspec(nrows, ncols)
    plot_helpers.gs_group_label(gs_plots, fig, xlabel=PLOT_CONFIG[map_names[0]]['title'], ylabel=PLOT_CONFIG[map_names[1]]['title'])

    for row in range(nrows):
        for col in range(ncols):
            i = ncols * row + col
            if i > len(apertures) - 1:
                continue
            aper = apertures[i]
            aper_data = muse_data.place_aperture(aper, map_names)
            x = aper_data[map_names[0]]
            y = aper_data[map_names[1]]

            ax = fig.add_subplot(gs_plots[row, col])
            ax.scatter(x, y, s=2, c='k', alpha=0.75)
            ax.text(0.05, 0.95, f"({aper.label})", transform=ax.transAxes, ha='left', va='top')

            p = pearsonr(x, y)
            ax.text(0.95, 0.95, fr"$r={p.statistic:.2f}$" + "\n" + fr"$p={p.pvalue:.1e}$", transform=ax.transAxes, ha='right', va='top')

            ax.set_xlabel('')
            ax.set_ylabel('')


    if save:
        subfname = f"APERTURES-{map_names[0]}-{map_names[1]}"
        filename = f'{muse_data.galname}-{muse_data.bin_method}-{subfname}.pdf' if filename is None else filename
        savedir = os.path.join(muse_data.figures_dir, 'custom_plots') if savedir is None else savedir
        util.check_filepath(savedir)
        fig.savefig(os.path.join(savedir, filename), bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def vcen_with_radius(
        muse_data: "MuseNAPData",
        r_lims: tuple = (0., 0.25, 0.5, 1.),
        show: bool = False,
        save: bool = True
    ) -> None:
    fig, gsmain = plot_helpers.setup_figure(1, 2, fig_width_scalar=1.5, fig_height_scalar=1.5, wspace=.3)

    gsmap = gsmain[0,0].subgridspec(1,1)
    axmap = fig.add_subplot(gsmap[0,0])


    ncols = 2
    nrows = len(r_lims) - 1
    gsplots = gsmain[0,1].subgridspec(nrows, ncols, hspace=0.2, wspace=0.2)
    #fig, gs = plot_helpers.setup_figure(nrows, ncols, hspace=0.2,wspace=0.2)
    
    dap_data = muse_data.dap_data
    rcoords = dap_data.r_coords
    spatial_bins = dap_data.spatial_bins

    vcenmap = muse_data.get_map("v_cen")
    sfrmap = muse_data.get_map("sfrsd")

    vcenmap.map_plot(ax=axmap)
    map_contour(rcoords, levels=list(r_lims),ax=axmap, smooth=False, colors=['k']*len(r_lims), linewidths=[1.6]*len(r_lims))

    mask = np.logical_or(vcenmap.mask.astype(bool), sfrmap.mask.astype(bool))
    mdict = util.get_unique_bin_values({'vcen':vcenmap.data, 'sfr':sfrmap.data, 'r':rcoords}, spatial_bins, mask=mask)
    v = mdict['vcen']
    sfr = mdict['sfr']
    r = mdict['r']

    for i in range(len(r_lims) - 1):
        r0, r1 = r_lims[i], r_lims[i+1]
        w = (r >= r0) & (r < r1)

        axout = fig.add_subplot(gsplots[i,0])
        axin = fig.add_subplot(gsplots[i,1])

        wout = w & (v < 0)
        axout.scatter(sfr[wout], np.abs(v[wout]), c='b')
        p = pearsonr(sfr[wout], np.abs(v[wout]))
        axout.text(0.95, 0.95, s=fr"$r=${p.statistic:.2f}" + "\n" + fr"$p=${p.pvalue:.1g}", ha='right',va='top',transform=axout.transAxes)

        win = w & (v > 0)
        axin.scatter(sfr[win], v[win], c='r')
        p = pearsonr(sfr[win], v[win])
        axin.text(0.95, 0.95, s=fr"$r=${p.statistic:.2f}" + "\n" + fr"$p=${p.pvalue:.1g}", ha='right',va='top',transform=axin.transAxes)

    plot_helpers.gs_group_label(gsplots, fig,
                                r'$\Sigma_{\mathrm{SFR}}\ \left( \mathrm{M_{\odot}\ yr^{-1}\ kpc^{-2}\ spaxel^{-1}} \right)$',
                                r'$\left| v_{\mathrm{cen}} \right|\ \left( \mathrm{km\ s^{-1}} \right)$')
    
    if save:
        filename = f'{muse_data.galname}-{muse_data.bin_method}-VCEN_RADIUS.pdf' if filename is None else filename
        savedir = os.path.join(muse_data.figures_dir, 'custom_plots') if savedir is None else savedir
        util.check_filepath(savedir)
        fig.savefig(os.path.join(savedir, filename), bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def vcen_vs_vgas(
        muse_data: "MuseNAPData",
        show: bool = False,
        save: bool = True
    ) -> None:
    vcen_map = muse_data.get_map("v_cen")
    mask = vcen_map.mask.astype(bool)

    dap_data = muse_data.dap_data
    vgas_map = dap_data.get_emline("EMLINE_GVEL", 'Ha-6564')
    spatial_bins = dap_data.spatial_bins

    mdict = util.get_unique_bin_values({'vcen':vcen_map.data, 'vcen_err':vcen_map.error, 'vgas':vgas_map.data, 'vgas_err':1/np.sqrt(vgas_map.ivar)}, spatial_bins, mask=mask)

    fig, gs = plot_helpers.setup_figure(1,1)
    ax = fig.add_subplot(gs[0,0])


    # ax.errorbar(mdict['vgas'], mdict['vcen'], yerr=mdict['vcen_err'], xerr=mdict['vgas_err'], color='k', linestyle='none', markersize=0.1, elinewidth=0.5)
    ax.scatter(mdict['vgas'], mdict['vcen'], s=.5, c='k')
    ax.set_xlabel(r'$V_{\mathrm{gas}}\ \left( \mathrm{km\ s^{-1}} \right)$')
    ax.set_ylabel(PLOT_CONFIG['V_CEN']['title'])
    plot_helpers.median_error(ax=ax, xerr=np.median(mdict['vgas_err']), yerr=np.median(mdict['vcen_err']))

    if save:
        filename = f'{muse_data.galname}-{muse_data.bin_method}-VCEN_vs_VGAS.pdf' if filename is None else filename
        savedir = os.path.join(muse_data.figures_dir, 'custom_plots') if savedir is None else savedir
        util.check_filepath(savedir)
        fig.savefig(os.path.join(savedir, filename), bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()