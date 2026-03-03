from maps.musemap import MuseMAP
from measurements.measurement_map import MeasurementMAP
from measurements.registry import measurement_registry
from musedap_data import MuseDAPData

class MeasurementEngine:
    def __init__(self, dap_data: MuseDAPData, verbose = False):
        self.dap_data = dap_data
        self.verbose = verbose

        self._cache: dict[str, MuseMAP] = {}
        self._registry = measurement_registry


    def get(self, name: str) -> MuseMAP:
        if name in self._cache:
            return self._cache[name]
        
        if name not in self._registry:
            raise ValueError(f"Measurement '{name}' is not registered.\nCurrent measurements: {self._registry.keys()}")

        measurement_cls = self._registry[name]

        for dep in getattr(measurement_cls, "dependencies", []):
            self.get(dep)

        measurement = measurement_cls(self.dap_data, self.verbose)
        measurement_map = measurement.compute()

        self._cache[name] = measurement_map
        return measurement_map
