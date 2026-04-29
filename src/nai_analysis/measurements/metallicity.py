import numpy as np
from numpy.polynomial.polynomial import polyroots, polyval, polyder
from scipy.ndimage import median as ndmedian

from src.nai_analysis.measurements.measurement_map import MeasurementMAP
from src.nai_analysis.maps.musemap import MuseMAP
from src.nai_analysis.maps.bitmask import MuseMapBitMask
from src.nai_analysis.utils import util, progress
from typing import TYPE_CHECKING

import time

if TYPE_CHECKING:
    from src.nai_analysis.engine.measurement_engine import MeasurementEngine


class MetallicityMAP(MeasurementMAP):

    name = "metallicity"
    dependencies = []

    def compute(self, engine: "MeasurementEngine") -> MuseMAP:
        with progress.ProgressWheel(f"Computing {self.name} MAP"):
            start = time.time()

            dap_data = self.dap_data
            spatial_bins = dap_data.spatial_bins

            h_alpha = dap_data.get_emline("EMLINE_GFLUX", 'Ha-6564')
            h_beta = dap_data.get_emline("EMLINE_GFLUX", 'Hb-4862')

            n_ii = dap_data.get_emline("EMLINE_GFLUX", 'NII-6585')
            o_iii = dap_data.get_emline("EMLINE_GFLUX", 'OIII-5008')

            n2 = n_ii.data / h_alpha.data
            o3n2 = (o_iii.data / h_beta.data) / (n2)

            # https://academic.oup.com/mnras/article/491/1/944/5638748
            coef_o3n2 = [0.281, -4.765, -2.268]
            coef_n2 = [-0.489, 1.513, -2.554, -5.293, -2.867]
            coefficients = {
                'N2':[-0.489, 1.513, -2.554, -5.293, -2.867],
                'O3N2':[0.281, -4.765, -2.268],
                'O3S2':[0.191, -4.292, -2.538, 0.053, 0.332],
                'RS32':[-0.054, -2.546, -1.970, 0.082, 0.222 ],
                'S2':[-0.442, -0.360, -6.271, -8.339, -3.559 ],
                'R3':[-0.277, -3.549, -3.593, -0.981]
            }

            Z_map = MuseMAP.empty_from_binmap("metallicity", dap_data.galname, dap_data.bin_method, spatial_bins)
            bm = MuseMapBitMask()
            bm.set_flag(Z_map.mask, spatial_bins==-1, ['NO VALUE'])

            for binid in np.unique(spatial_bins)[1:]:
                w = binid == spatial_bins
                log_o3n2 = np.log10(np.median(o3n2[w]))
                log_n2 = np.log10(np.median(n2[w]))

                if not np.isfinite(log_o3n2) and not np.isfinite(log_n2):
                    bm.set_flag(Z_map.mask, w, ["DO NOT USE", "NO VALUE"])
                    continue

                Z_n2 = self.solve_metallicity(log_n2, coef_n2)
                Z_o3n2 = self.solve_metallicity(log_o3n2, coef_o3n2)

                if not np.isfinite(Z_n2) or not np.isfinite(Z_o3n2):
                    bm.set_flag(Z_map.mask, w, ['UNRELIABLE'])
                
                Z = np.nanmean([Z_n2, Z_o3n2])

                if not np.isfinite(Z):
                    bm.set_flag(Z_map.mask, w, ['DO NOT USE', 'MATH ERROR'])
                    continue

                Z_map.data[w] = Z

            end = time.time()
        util.sys_message(f"Constructed {self.name} MAP: time to complete {end-start:.3g} s", color='green', verbose=self.verbose)
        return Z_map


    @staticmethod
    def solve_metallicity(log_R: np.ndarray, coefficients: list) -> float:
        if not np.isfinite(log_R):
            return np.nan

        z_ref = 8.69
        z_lims = np.array([7.6, 8.9]) - z_ref

        coeffs = coefficients.copy()
        coeffs[0] -= log_R

        roots = polyroots(coeffs)
        real_roots = roots[np.abs(roots.imag) < 1e-6].real

        if len(real_roots) == 0:
            return np.nan
        
        elif len(real_roots) == 1:
            x = real_roots[0]

        else:
            valid = real_roots[(real_roots > z_lims[0]) & (real_roots < z_lims[1])]
            if len(valid) == 0:
                return np.nan
            
            x = valid[np.argmin(valid)]
        
        return x + z_ref
