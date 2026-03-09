from src.nai_analysis.maps.musemap import MuseMAP
from src.nai_analysis.measurements.registry import measurement_registry
from src.nai_analysis.musedap_data import MuseDAPData
from src.nai_analysis.measurements.mcmc_table import MCMCTable

import numpy as np
from typing import Union

class MeasurementEngine:
    def __init__(self, dap_data: MuseDAPData, verbose = False):
        self.dap_data = dap_data
        self.verbose = verbose

        self._cache: dict[str, MuseMAP] = {}
        self._array_cache: dict[str, np.ndarray] = {}
        self._registry = measurement_registry


    def get(self, name: str) -> Union[MuseMAP, np.ndarray]:
        if name in self._cache:
            return self._cache[name]
        if name in self._array_cache:
            return self._array_cache[name]
        
        if name == 'mcmc_table':
            mcmc_arr = self._compute_mcmc_table()
            self._array_cache[name] = mcmc_arr
            return mcmc_arr
        
        if name not in self._registry:
            raise ValueError(f"Measurement '{name}' is not registered.\nCurrent measurements: {self._registry.keys()}")

        measurement_cls = self._registry[name]

        for dep in getattr(measurement_cls, "dependencies", []):
            self.get(dep)

        measurement = measurement_cls(self.dap_data, self.verbose)
        measurement_map = measurement.compute(self)

        self._cache[name] = measurement_map
        return measurement_map
    
    def _compute_mcmc_table(self) -> np.ndarray:
        mcmc_obj = MCMCTable('mcmc_table', self.dap_data, self.verbose)
        return mcmc_obj.compute()