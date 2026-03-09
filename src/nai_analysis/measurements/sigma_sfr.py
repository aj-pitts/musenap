import numpy as np
import astropy.units as u
import astropy.constants as cst
from astropy.coordinates import Angle
import warnings
import time

from src.nai_analysis.measurements.measurement_map import MeasurementMAP
from src.nai_analysis.maps.musemap import MuseMAP
from src.nai_analysis.maps.bitmask import MuseMapBitMask
from src.nai_analysis.utils import util, progress

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.nai_analysis.engine.measurement_engine import MeasurementEngine


class SigmaSfrMAP(MeasurementMAP):

    name = "sfrsd"
    dependencies = ["redshift"]

    def compute(self, engine: "MeasurementEngine") -> MuseMAP:
        with progress.ProgressWheel(f"Computing {self.name} MAP"):
            start = time.time()
            warnings.simplefilter("ignore", RuntimeWarning)

            DAP = self.dap_data
            spatial_bins = DAP.spatial_bins

            redshift_musemap = engine.get('redshift')
            redshifts = redshift_musemap.data.astype(np.float64) ## necessary for large D_cm values

            sfr_map = MuseMAP.empty_from_binmap(self.name, DAP.galname, DAP.bin_method, spatial_bins)
            bm = MuseMapBitMask()

            valid = spatial_bins>=0
            counts = np.bincount(spatial_bins[valid])
            bin_sizes = np.zeros_like(spatial_bins)
            bin_sizes[valid] = counts[spatial_bins[valid]]

            c = cst.c
            H0 = 70 * u.km / u.s / u.Mpc
            D_kpc = (c / H0).to(u.kpc).value * redshifts
            D_cm = (c / H0).to(u.cm).value * redshifts

            angle_per_bin = Angle(0.2, unit='arcsec')
            areas_per_bin = (D_kpc * angle_per_bin.radian)**2
            areas = bin_sizes * areas_per_bin


            ha = DAP.emline_gflux[23]
            #ha_ivar = DAP.emline_gflux_ivar[23]
            hb = DAP.emline_gflux[14]
            #hb_ivar = DAP.emline_gflux_ivar[14]

            flux = self.correct_dust(ha, hb)

            luminosity = 4 * np.pi * D_cm**2 * flux * 1e-17

            sfr = np.log10(luminosity) - 41.27

            sigma_sfr = np.log10( (10 ** sfr) / areas )

            bm.set_flag(sfr_map.mask, ~valid, ["NO_VALUE", "DO_NOT_USE"])
            sigma_sfr[~valid] = 0

            bm.set_flag(sfr_map.mask, ~np.isfinite(sigma_sfr), ["DO_NOT_USE", "MATH_ERROR"])
            sigma_sfr[~np.isfinite(sigma_sfr)] = 0

            #bm.set_flag(sfr_map.mask, error == - 999, ["UNREALIABLE", "UNCERTAINTY_OOB"])

            sfr_map.data = sigma_sfr
            sfr_map.error = np.zeros_like(sigma_sfr)

            end = time.time()
        util.sys_message(f"Constructed {self.name} MAP: time to complete {end-start:.3g} s", color='green', verbose=self.verbose)
        return sfr_map

    
    def correct_dust(self, Ha, Hb, HaHb_ratio = 2.87, Rv = 3.1, k_Ha = 2.45, k_Hb = 3.65):
        """
        Corrects H-alpha flux for dust attenuation using the Balmer decrement method.

        Parameters
        ----------
        Ha : array_like
            Observed H-alpha flux values.
        Hb : array_like
            Observed H-beta flux values.
        HaHb_ratio : float, optional
            Theoretical H-alpha/H-beta flux ratio for case B recombination in an 
            ionized gas. The default is 2.87, which assumes a typical electron density 
            and temperature for HII regions.
        Rv : float, optional
            Total-to-selective extinction ratio, typically 3.1 for the Milky Way. 
            (Note: This parameter is currently not used in the calculation.)
        k_Ha : float, optional
            Extinction coefficient at the wavelength of H-alpha (6563 Å).
        k_Hb : float, optional
            Extinction coefficient at the wavelength of H-beta (4861 Å).
        """

        E_BV = (2.5 / (k_Hb - k_Ha)) * np.log10( (Ha / Hb) / HaHb_ratio )
        E_BV[~np.isfinite(E_BV)] = 0
        A_gas = E_BV * k_Ha

        power = 0.4 * A_gas
        F_corr = Ha * (10 ** power)
        return F_corr