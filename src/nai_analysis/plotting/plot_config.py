import cmasher as cmr

PLOT_CONFIG = {
    "SNR_NAI":dict(cmap = cmr.sapphire, title = r"$S/N_{\mathrm{Na\ I}}$", vmin = 0),
    "REDSHIFT":dict(cmap = 'coolwarm', title = r"$z$"),
    "SFRSD":dict(cmap = 'rainbow', title = r"$\mathrm{log \Sigma_{SFR}}\ \left( \mathrm{M_{\odot}\ yr^{-1}\ kpc^{-2}\ spx^{-1}} \right)$"),
    "WEQ_NAI":dict(cmap = cmr.gem, title = r"$\mathrm{EW_{Na\ I}}$"),
    "WEQ_ABS_NAI":dict(cmap = cmr.amber, title = r"$\mathrm{EW_{Na\ I,\ abs}}$"),
    "WEQ_EM_NAI":dict(cmap = cmr.freeze_r, title = r"$\mathrm{EW_{Na\ I,\ em}}$"),
    "V_CEN":dict(cmap = 'seismic', title = r"$v_{\mathrm{cen}}\ \left( \mathrm{km\ s^{-1}} \right)$"),
}