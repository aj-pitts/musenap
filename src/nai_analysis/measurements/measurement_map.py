from abc import ABC, abstractmethod
from musedap_data import MuseDAPData
from musenap_data import MuseNAPData
from typing import Optional, TYPE_CHECKING
from maps.musemap import MuseMAP

if TYPE_CHECKING:
    from engine.measurement_engine import MeasurementEngine

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
    
