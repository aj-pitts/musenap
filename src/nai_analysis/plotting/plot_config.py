import cmasher as cmr

PLOT_CONFIG: dict[str, dict] = {
    "SNR_NAI":dict(facecolor='k', title = r"$S/N_{\mathrm{Na\ I}}$", imshow_kwargs = dict(cmap = cmr.sapphire, vmin = 0)),
    #"REDSHIFT":dict(symmetric = True, title = r"$z$", imshow_kwargs = dict(cmap = 'coolwarm')),
    "SFRSD":dict(title = r"$\mathrm{log \Sigma_{SFR}}\ \left( \mathrm{M_{\odot}\ yr^{-1}\ kpc^{-2}\ spx^{-1}} \right)$", imshow_kwargs = dict(cmap = 'rainbow')),
    "WEQ_NAI":dict(title = r"$\mathrm{EW_{Na\ I}}$", imshow_kwargs = dict(cmap = cmr.gem)),
    "WEQ_ABS_NAI":dict(title = r"$\mathrm{EW_{Na\ I,\ abs}}$", imshow_kwargs = dict(cmap = cmr.amber)),
    "WEQ_EM_NAI":dict(title = r"$\mathrm{EW_{Na\ I,\ em}}$", imshow_kwargs = dict(cmap = cmr.freeze_r)),
    "V_CEN":dict(symmetric = True, title = r"$v_{\mathrm{cen}}\ \left( \mathrm{km\ s^{-1}} \right)$", imshow_kwargs = dict(cmap = 'seismic')),
    "V_CEN_SYS":dict(symmetric = True, title = r"$v_{\mathrm{cen,\ sys}}\ \left( \mathrm{km\ s^{-1}} \right)$", imshow_kwargs = dict(cmap = 'seismic')),
    "V_GAS":dict(symmetric = True, title = r"$v_{\mathrm{H\alpha}}\ \left( \mathrm{km\ s^{-1}} \right)$", imshow_kwargs = dict(cmap = 'seismic'))
}