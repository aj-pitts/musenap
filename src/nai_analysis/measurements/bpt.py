import numpy as np
import warnings
import time
from typing import TYPE_CHECKING

from src.nai_analysis.measurements.measurement_map import MeasurementMAP
from src.nai_analysis.maps.musemap import MuseMAP
from src.nai_analysis.maps.bitmask import MuseMapBitMask
from src.nai_analysis.utils import progress, util
if TYPE_CHECKING:
    from src.nai_analysis.engine.measurement_engine import MeasurementEngine

class BPTMap(MeasurementMAP):
    
    name = "BPT"
    dependencies = []
    
    def compute(self, engine: "MeasurementEngine"):
        with progress.ProgressWheel(f"Computing {self.name} MAP"):
            start = time.time()
            ha = self.dap_data.get_emline('EMLINE_GFLUX', 'Ha-6564').data
            hb = self.dap_data.get_emline('EMLINE_GFLUX', 'Hb-4862').data
            oiii = self.dap_data.get_emline('EMLINE_GFLUX', 'OIII-5008').data
            nii = self.dap_data.get_emline('EMLINE_GFLUX', 'NII-6585').data
            sii = self.dap_data.get_emline('EMLINE_GFLUX', 'SII-6718').data + self.dap_data.get_emline('EMLINE_GFLUX', 'SII-6732').data
            oi = self.dap_data.get_emline('EMLINE_GFLUX', 'OI-6302').data

            log_oiii_hb = np.log10(oiii / hb)
            log_nii_ha = np.log10(nii / ha)
            log_sii_ha = np.log10(sii / ha)
            log_oi_ha = np.log10(oi / ha)

            Ka03_nii = lambda x: 0.61 / (x - 0.05) + 1.3
            Ke01_nii = lambda x: 0.61 / (x - 0.47) + 1.19
            Ke01_sii = lambda x: 0.72 / (x - 0.32) + 1.3
            Ke01_oi = lambda x: 0.73 / (x - 0.59) + 1.33
            SL_sii = lambda x: 1.89 * x + 0.76
            SL_oi = lambda x: 1.18 * x + 1.3

            star_forming = (Ka03_nii(log_nii_ha) > log_oiii_hb) & (Ke01_sii(log_sii_ha) > log_oiii_hb) & (Ke01_oi(log_oi_ha) > log_oiii_hb)
            composite = ((Ka03_nii(log_nii_ha)) < log_oiii_hb) & (Ke01_nii(log_nii_ha) > log_oiii_hb)
            
            Ke01 = (Ke01_nii(log_nii_ha) < log_oiii_hb) & (Ke01_sii(log_sii_ha) < log_oiii_hb) & (Ke01_oi(log_oi_ha) < log_oiii_hb)
            SL = (SL_sii(log_sii_ha) < log_oiii_hb) & (SL_oi(log_oi_ha) < log_oiii_hb)

            seyfert = Ke01 & SL
            liner = Ke01 & ~ SL

            ambiguous = ~star_forming & ~composite & ~seyfert & ~liner

            bpt_map = MuseMAP.empty_from_binmap(self.name, self.dap_data.galname, self.dap_data.bin_method, self.dap_data.spatial_bins, default=-1)
            bm = MuseMapBitMask()
            bm.set_flag(bpt_map.mask, self.dap_data.spatial_bins == -1, ['NO VALUE'])

            bpt_map.data[ambiguous] = 0
            bpt_map.data[liner] = 4
            bpt_map.data[seyfert] = 3
            bpt_map.data[composite] = 2
            bpt_map.data[star_forming] = 1

            end = time.time()
        util.sys_message(f"Constructed {self.name} MAP: time to complete {end-start:.3g} s", color='green', verbose=self.verbose)

        return bpt_map
