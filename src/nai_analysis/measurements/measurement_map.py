from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from src.nai_analysis.musedap_data import MuseDAPData
from src.nai_analysis.maps.musemap import MuseMAP

if TYPE_CHECKING:
    from src.nai_analysis.engine.measurement_engine import MeasurementEngine

class MeasurementMAP(ABC):

    name: str
    dependencies: list[str] = []

    """Base class of a measurement map that is to be computed. All measurement maps computed for the MUSE NAP inherit this class"""
    def __init__(self, dap_data: MuseDAPData, verbose: bool = False):
        self.dap_data = dap_data
        self.verbose = verbose

    @abstractmethod
    def compute(self, engine: "MeasurementEngine") -> MuseMAP:
        """Compute the measurement and return a MuseMAP instance."""
        pass
