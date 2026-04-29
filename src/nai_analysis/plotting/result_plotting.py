from matplotlib.axes import Axes

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from src.nai_analysis.utils import util, defaults
from src.nai_analysis.plotting.plot_config import PLOT_CONFIG
from src.nai_analysis.plotting import plot_helpers
from src.nai_analysis.plotting.plot_helpers import measurement_labels
from src.nai_analysis.musedap_data import MuseDAPData

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.nai_analysis.musenap_data import MuseNAPData

import numpy as np


def incidence(nap_data: "MuseNAPData",
              bin_width: float = 0.2,
              min_spabins: int = 50,
              velocity_mask: bool = True,
              show: bool = False,
              save: bool = True,
              verbose: bool = False
              ) -> None:
    
    def wilson_interval(k, n, z=1.0):
        if n == 0:
            return 0., 0.
        p = k / n
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        margin = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
        return center - margin, center + margin
    
    plt.style.use(defaults.matplotlib_rc())

    spatial_bins = nap_data.dap_data.spatial_bins
    vfrac_map = nap_data.get_map("V_FRAC")
    vcen_map = nap_data.get_map("V_CEN")
    sfrsd_map = nap_data.get_map("SFRSD")
    
    sfr_distr = util.get_unique_bin_values(sfrsd_map.data, spatial_bins, sfrsd_map.mask)
    sfr_gmin = sfr_distr.min()
    sfr_gmax = sfr_distr.max()

    #mask = (vcen_map.data==0) | sfrsd_map.mask
    mask = vcen_map.mask | sfrsd_map.mask if velocity_mask else sfrsd_map.mask
    unique_measurements = util.get_unique_bin_values({"vfrac":vfrac_map.data, "sfrsd":sfrsd_map.data}, spatial_bins, mask)

    sfr = unique_measurements['sfrsd']
    vfrac = unique_measurements['vfrac']

    sfmin, sfmax = sfr.min(), sfr.max()

    base_w, base_h = plt.rcParams['figure.figsize']
    nrow = 1; ncol = 2
    fig = plt.figure(figsize=(base_w * ncol, base_h * nrow))
    gs = gridspec.GridSpec(nrow, ncol, figure=fig, wspace=0.1)
    ax_out = fig.add_subplot(gs[0,0])
    ax_in = fig.add_subplot(gs[0,1])

    plot_helpers.gs_group_label(gs=gs, fig=fig, xlabel = f"{measurement_labels["SFRSD"]['value']} {measurement_labels["SFRSD"]['units']}")

    sfrbins = np.arange(sfmin, sfmax + bin_width, bin_width)
    binned_centers = 0.5 * (sfrbins[:-1] + sfrbins[1:])
    widths = np.diff(sfrbins)
    inds = np.digitize(sfr, sfrbins)

    spabins_per_bin = np.array([np.sum(inds == i) for i in range(1, len(sfrbins))])
    good_bins = spabins_per_bin >= min_spabins

    frac_in, frac_out = [], []
    err_in_lo, err_in_hi = [], []
    err_out_lo, err_out_hi = [], []

    for i in range(1, len(sfrbins)):
        mask_i = i == inds
        n = np.sum(mask_i)
        
        k_in = np.sum(vfrac[mask_i] >= .95)
        k_out = np.sum(vfrac[mask_i] <= -.95)
        
        p_in = k_in / n if n > 0 else 0.
        p_out = k_out / n if n > 0 else 0.
        
        lo_in, hi_in = wilson_interval(k_in, n)
        lo_out, hi_out = wilson_interval(k_out, n)
        
        frac_in.append(p_in)
        frac_out.append(p_out)
        err_in_lo.append(max(0., p_in - lo_in))
        err_in_hi.append(max(0., hi_in - p_in))
        err_out_lo.append(max(0., p_out - lo_out))
        err_out_hi.append(max(0., hi_out - p_out))

    frac_in = np.array(frac_in)
    frac_out = np.array(frac_out)
    err_in = [np.array(err_in_lo), np.array(err_in_hi)]
    err_out = [np.array(err_out_lo), np.array(err_out_hi)]

    frac_in[np.isnan(frac_in)] = 0
    frac_out[np.isnan(frac_out)] = 0

    binned_centers = binned_centers[good_bins]
    widths = widths[good_bins]

    frac_in = frac_in[good_bins]
    frac_out = frac_out[good_bins]

    err_in = [e[good_bins] for e in err_in]
    err_out = [e[good_bins] for e in err_out]
    print(spabins_per_bin[good_bins])
    print(binned_centers)

    maxfrac = max(np.max(frac_in), np.max(frac_out))
    maxerr = max(np.max(err_in), np.max(err_out))
    ymax = np.ceil((maxfrac+maxerr) * 10) / 10

    ax_out.bar(binned_centers, frac_out, width=widths, align='center', edgecolor='k', color = '#0063ff',
            linewidth = 0.9)
    ax_out.errorbar(binned_centers, frac_out, yerr=err_out,
                fmt='none', color='k', capsize=3, linewidth=0.9)
    #ax_out.set_ylabel(r'$f_{v_{\mathrm{cen}} \in \left\{ P(\Delta v) \geq 0.95 \right\} }$')
    ax_out.set_ylabel(r'Incidence')
    ax_out.set_ylim(0, ymax)
    #ax_out.set_xlim(sfmin, sfmax)
    ax_out.set_xlim(min(binned_centers)- 2*bin_width, max(binned_centers) + 2*bin_width)

    ax_in.bar(binned_centers, frac_in, width=widths, align='center', edgecolor='k', color = '#ce0014',
            linewidth = 0.9)
    ax_in.errorbar(binned_centers, frac_in, yerr=err_in,
               fmt='none', color='k', capsize=3, linewidth=0.9)
    ax_in.set_ylim(0, ymax)
    ax_in.set_yticklabels([])
    ax_in.set_xlim(min(binned_centers)- 2*bin_width, max(binned_centers) + 2*bin_width)

    if save:
        fname = plot_helpers.figure_filename(nap_data.galname, nap_data.bin_method, "Incidence")
        out = os.path.join(nap_data.figures_dir, 'results', fname)
        plt.savefig(out, bbox_inches='tight')
        util.sys_message(f"Incidence plot saved to {out}", verbose=verbose)

    if show:
        plt.show()
    else:
        plt.close()

def vcen_hists(
        nap_data: "MuseNAPData",
        vcen_bin_width: float,
        vcen_error_bin_width: float,
        snr_min: float = 30,
        snr_max: float = 90,
        snr_width: float = 5,
        snr_lims: Optional[list[tuple]] = None,
        sfr_min: float = -3.5,
        sfr_max: float = -1,
        sfr_width: float = 0.25,
        sfr_lims: Optional[list[tuple]] = [(-3.5, -2.5), (-2.5, -2.25), (-2.25, -2.0), (-2.0, -1.0)],
        max_cols: int = 4,
        max_rows: Optional[int] = None,
        show = False,
        save = True,
        verbose = False
) -> None:
    plt.style.use(defaults.matplotlib_rc())

    spatial_bins = nap_data.dap_data.spatial_bins
    vfrac_map = nap_data.get_map("V_FRAC")
    vcen_map = nap_data.get_map("V_CEN")
    snr_map = nap_data.get_map("SNR_NAI")
    sfr_map = nap_data.get_map("SFRSD")

    unique_measurements = util.get_unique_bin_values(
        {
            "vfrac":vfrac_map.data, 
            "sfr":sfr_map.data,
            "snr":snr_map.data, 
            "vcen":vcen_map.data, 
            "vcen_error":vcen_map.error
        }, 
        spatial_bins,
        mask=vcen_map.mask)

    vcen = unique_measurements['vcen']
    vcen_err = unique_measurements['vcen_error']
    vfrac = unique_measurements['vfrac']
    snr = unique_measurements['snr']
    sfr = unique_measurements['sfr']

    snr_lims = [(x, x + snr_width) for x in np.arange(snr_min, snr_max, snr_width)] if snr_lims is None else snr_lims # + [(snr_max, np.inf)] if snr_lims is None else snr_lims
    sfr_lims = [(x, x + sfr_width) for x in np.arange(sfr_min, sfr_max, sfr_width)] if sfr_lims is None else sfr_lims

    lims = [snr_lims, sfr_lims]
    arrs = [snr, sfr]
    labels = [measurement_labels['SNR_NAI']['value'], measurement_labels['SFRSD']['value']]
    fignames = ['vcen_snr_hists','vcen_sfr_hists']

    for lim, arr, label, figname in zip(lims, arrs, labels, fignames):
        nplots = len(lim)
        ncol = max_cols
        nrow = int(np.ceil(nplots / ncol))
        max_rows = nrow if max_rows is None else max_rows
        if nrow > max_rows:
            raise ValueError(f"Can not fit {nplots} plots into {ncol} cols and {nrow} rows.")

        base_w, base_h = plt.rcParams['figure.figsize']
        fig_vcen = plt.figure(figsize=(base_w * ncol, base_h * nrow))
        gs_vcen = gridspec.GridSpec(nrow, ncol, figure=fig_vcen, wspace=0.1)

        fig_sigma = plt.figure(figsize=(base_w * ncol, base_h * nrow))
        gs_sigma = gridspec.GridSpec(nrow, ncol, figure=fig_sigma, wspace=0.1)

        axes_vcen: list[Axes] = []
        axes_sigma: list[Axes] = []
        i = 1
        for row in range(nrow):
            for col in range(ncol):
                if i>nplots:
                    continue
                ax_vcen = fig_vcen.add_subplot(gs_vcen[row, col])
                axes_vcen.append(ax_vcen)
                ax_sigma = fig_sigma.add_subplot(gs_sigma[row, col])
                axes_sigma.append(ax_sigma)
                i+=1

        row_max_vcen = np.zeros(nrow)
        row_max_sigma = np.zeros(nrow)
        for idx, (ax_vcen, ax_sig, l) in enumerate(zip(axes_vcen, axes_sigma, lim)):
            row = idx // ncol
            col = idx % ncol

            mmin, mmax = l

            title = rf"${mmin:.1f} < ${label}$ \leq {mmax:.1f}$"
            if not np.isfinite(mmax):
                title = rf"{label}$ > {mmin:.1f}$"
            
            w = (arr > mmin) & (arr <= mmax)
            vcen_subset = vcen[w]
            err_subset = vcen_err[w]
            frac_subset = vfrac[w]

            vin = vcen_subset[frac_subset >= 0.95]
            vout = vcen_subset[frac_subset <= -0.95]
            vinsig = vcen_subset[np.abs(frac_subset) < 0.95]

            ein = err_subset[frac_subset >= 0.95]
            eout = err_subset[frac_subset <= -0.95]
            einsig = err_subset[np.abs(frac_subset) < 0.95]

            vmin, vmax = vcen_subset.min(), vcen_subset.max()
            vbins = np.arange(vmin, vmax + vcen_bin_width, vcen_bin_width)

            emin, emax = err_subset.min(), err_subset.max()
            ebins = np.arange(emin, emax + vcen_error_bin_width, vcen_error_bin_width)

            counts, _, _ = ax_vcen.hist(vinsig, bins=vbins, color='dimgray', histtype='stepfilled', linewidth=1.5, alpha=0.4)
            row_max_vcen[row] = max(row_max_vcen[row], counts.max())
            counts, _, _ = ax_vcen.hist(vout, bins=vbins, color='blue', histtype='step', linewidth=1.5)
            row_max_vcen[row] = max(row_max_vcen[row], counts.max())
            counts, _, _ = ax_vcen.hist(vin, bins=vbins, color='red', histtype='step', linewidth=1.5)
            row_max_vcen[row] = max(row_max_vcen[row], counts.max())
            ax_vcen.set_title(title)

            counts, _, _ = ax_sig.hist(einsig, bins=ebins, color='dimgray', histtype='stepfilled', linewidth=1.5, alpha=0.4)
            row_max_sigma[row] = max(row_max_sigma[row], counts.max())
            counts, _, _ = ax_sig.hist(eout, bins=ebins, color='blue', histtype='step', linewidth=1.5)
            row_max_sigma[row] = max(row_max_sigma[row], counts.max())
            counts, _, _ = ax_sig.hist(ein, bins=ebins, color='red', histtype='step', linewidth=1.5)
            row_max_sigma[row] = max(row_max_sigma[row], counts.max())
            ax_sig.set_title(title)

            percentiles = np.percentile(err_subset, [50, 75, 95])
            ax_sig.vlines(percentiles, 0, 1000, color='dimgray', linestyles='dashed', linewidths=0.9)
            # ax_sig.vlines(np.percentile(ein, [95]), 0, 1000, colors='tab:red', linestyles='dashed')
            # ax_sig.vlines(np.percentile(eout, [95]), 0, 1000, colors='tab:blue', linestyles='dashed')
            # ax_sig.vlines(np.percentile(einsig, [95]), 0, 1000, colors='k', linestyles='dashed')

            s = '\n'.join(rf"$P_{{{q}}} = {np.median(a):.0f}$ {measurement_labels['V_CEN']['units2']}" for a, q in zip(percentiles, [50, 75, 95]))
            #s = rf"med {measurement_labels['V_CEN']['error']} = {np.median(err_subset):.0f} {measurement_labels['V_CEN']['units']}"
            ax_sig.text(0.95, 0.95, s, transform=ax_sig.transAxes, ha='right', va='top', fontsize=12)

            ax_vcen.set_xlim(-250, 250)
            ax_sig.set_xlim(0,100)

            if col > 0:
                ax_vcen.set_yticklabels([])
                ax_sig.set_yticklabels([])
            
            if row != nrow - 1:
                ax_vcen.set_xticklabels([])
                ax_sig.set_xticklabels([])

            for idx, ax in enumerate(axes_vcen):
                row = idx // ncol
                ax.set_ylim(0, int(np.ceil((row_max_vcen[row] + 5)/10)*10))

            for idx, ax in enumerate(axes_sigma):
                row = idx // ncol
                ax.set_ylim(0, int(np.ceil((row_max_sigma[row] + 5)/10)*10))

        plot_helpers.gs_group_label(gs=gs_vcen, fig=fig_vcen, 
                                    xlabel=f"{measurement_labels["V_CEN"]['value']} {measurement_labels["V_CEN"]['units']}", 
                                    ylabel=r"$N_{\mathrm{bin}}$",
                                    xfontsize=20, yfontsize=20)
        plot_helpers.gs_group_label(gs=gs_sigma, fig=fig_sigma, 
                                    xlabel=f"{measurement_labels["V_CEN"]['error']} {measurement_labels["V_CEN"]['units']}",
                                    ylabel=r"$N_{\mathrm{bin}}$",
                                    xfontsize=20, yfontsize=20)

        if save:
            fname = plot_helpers.figure_filename(nap_data.galname, nap_data.bin_method, figname)
            out = os.path.join(nap_data.figures_dir, 'results', fname)
            fig_vcen.savefig(out, bbox_inches='tight')
            util.sys_message(f"{figname} plot saved to {out}", verbose=verbose)

            fname = plot_helpers.figure_filename(nap_data.galname, nap_data.bin_method, f"sigma_{figname}")
            out = os.path.join(nap_data.figures_dir, 'results', fname)
            fig_sigma.savefig(out, bbox_inches='tight')
            util.sys_message(f"Sigma {figname} hists plot saved to {out}", verbose=verbose)
        if show:
            plt.show()
        else:
            plt.close()

def sfr_snr_scatter(nap_data: "MuseNAPData",
                show = False,
                save = True,
                verbose = False) -> None:
    plt.style.use(defaults.matplotlib_rc())

    spatial_bins = nap_data.dap_data.spatial_bins
    vfrac_map = nap_data.get_map("V_FRAC")
    vcen_map = nap_data.get_map("V_CEN")
    sfrsd_map = nap_data.get_map("SFRSD")
    snr_map = nap_data.get_map("SNR_NAI")

    mask = snr_map.mask.astype(bool) | sfrsd_map.mask.astype(bool) | vcen_map.mask.astype(bool)
    unique_measurements = util.get_unique_bin_values({"vfrac":vfrac_map.data, "sfrsd":sfrsd_map.data, "snr":snr_map.data, "vcen":vcen_map.data}, spatial_bins, mask=mask)

    sfr = unique_measurements['sfrsd']
    vcen = unique_measurements['vcen']
    vfrac = unique_measurements['vfrac']
    snr = unique_measurements['snr']

    inflows = vfrac >= 0.95
    outflows = vfrac <= -0.95

    base_w, base_h = plt.rcParams['figure.figsize']
    nrow = 1; ncol = 2
    fig = plt.figure(figsize=(base_w * ncol, base_h * nrow))
    gs = gridspec.GridSpec(nrow, ncol, figure=fig, wspace=0.1)
    ax_in = fig.add_subplot(gs[0,0])
    ax_out = fig.add_subplot(gs[0,1])

    ax_in.hexbin(snr, sfr, gridsize=40, cmap='Greys', mincnt=1)
    ax_out.hexbin(snr, sfr, gridsize=40, cmap='Greys', mincnt=1)
    
    ax_in.scatter(snr[inflows], sfr[inflows], color='red', s=1, alpha=0.8)
    ax_out.scatter(snr[outflows], sfr[outflows], color='blue', s=1, alpha=0.8)

    ax_out.set_yticklabels([])

    ax_in.set_ylabel(f"{measurement_labels["SFRSD"]['value']} {measurement_labels["SFRSD"]['units']}")

    plot_helpers.gs_group_label(gs=gs, fig=fig, xlabel=measurement_labels["SNR_NAI"]['value'])

    if save:
        fname = plot_helpers.figure_filename(nap_data.galname, nap_data.bin_method, "snr_sfr_scatter")
        out = os.path.join(nap_data.figures_dir, 'results', fname)
        fig.savefig(out, bbox_inches='tight')
        util.sys_message(f"SFR vs SNR scatter plot saved to {out}", verbose=verbose)
    if show:
        plt.show()
    else:
        plt.close()


def vcen_scatters(nap_data: "MuseNAPData",
                show = False,
                save = True,
                verbose = False) -> None:
    plt.style.use(defaults.matplotlib_rc())

    spatial_bins = nap_data.dap_data.spatial_bins
    vfrac_map = nap_data.get_map("V_FRAC")
    vcen_map = nap_data.get_map("V_CEN")
    sfrsd_map = nap_data.get_map("SFRSD")
    snr_map = nap_data.get_map("SNR_NAI")
    weq_map = nap_data.get_map("WEQ_NAI")

    mask = snr_map.mask.astype(bool) | sfrsd_map.mask.astype(bool) | vcen_map.mask.astype(bool)
    unique_measurements = util.get_unique_bin_values(
        {
            "vfrac":vfrac_map.data, 
            "sfrsd":sfrsd_map.data, 
            "snr":snr_map.data, 
            "vcen":vcen_map.data,
            "weq":weq_map.data
        }, spatial_bins, mask=mask)

    sfr = unique_measurements['sfrsd']
    vcen = unique_measurements['vcen']
    vfrac = unique_measurements['vfrac']
    snr = unique_measurements['snr']
    weq = unique_measurements['weq']

    inflows = vfrac >= 0.95
    outflows = vfrac <= -0.95
    insig = np.abs(vfrac) < 0.95

    base_w, base_h = plt.rcParams['figure.figsize']
    nrow = 1; ncol = 3
    fig = plt.figure(figsize=(base_w * ncol, base_h * nrow))
    gs = gridspec.GridSpec(nrow, ncol, figure=fig, wspace=0.1)
    ax_snr = fig.add_subplot(gs[0,0])
    ax_sfr = fig.add_subplot(gs[0,1])
    ax_ew = fig.add_subplot(gs[0,2])

    #ax_snr.hexbin(snr, vcen, gridsize=40, cmap='Greys', mincnt=1)
    ax_snr.scatter(snr[insig], vcen[insig], color='dimgray', s=1, alpha=0.5)
    ax_snr.scatter(snr[outflows], vcen[outflows], color='blue', s=1, alpha=0.8)
    ax_snr.scatter(snr[inflows], vcen[inflows], color='red', s=1, alpha=0.8)
    
    #ax_sfr.hexbin(sfr, vcen, gridsize=40, cmap='Greys', mincnt=1)
    ax_sfr.scatter(sfr[insig], vcen[insig], color='dimgray', s=1, alpha=0.5)
    ax_sfr.scatter(sfr[outflows], vcen[outflows], color='blue', s=1, alpha=0.8)
    ax_sfr.scatter(sfr[inflows], vcen[inflows], color='red', s=1, alpha=0.8)

    ax_ew.scatter(weq[insig], vcen[insig], color='dimgray', s=1, alpha=0.5)
    ax_ew.scatter(weq[outflows], vcen[outflows], color='blue', s=1, alpha=0.8)
    ax_ew.scatter(weq[inflows], vcen[inflows], color='red', s=1, alpha=0.8)

    ax_snr.set_ylabel(f"{measurement_labels["V_CEN"]['value']} {measurement_labels["V_CEN"]['units']}")
    ax_snr.set_xlabel(measurement_labels['SNR_NAI']['value'])

    ax_sfr.set_xlabel(f"{measurement_labels['SFRSD']['value']} {measurement_labels['SFRSD']['units']}")

    ax_ew.set_xlabel(f"{measurement_labels['WEQ_NAI']['value']} {measurement_labels['WEQ_NAI']['units']}")

    ax_sfr.set_yticklabels([])
    ax_ew.set_yticklabels([])

    if save:
        fname = plot_helpers.figure_filename(nap_data.galname, nap_data.bin_method, "vcen_scatter")
        out = os.path.join(nap_data.figures_dir, 'results', fname)
        fig.savefig(out, bbox_inches='tight')
        util.sys_message(f"VCEN scatter plot saved to {out}", verbose=verbose)
    if show:
        plt.show()
    else:
        plt.close()