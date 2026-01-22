from musedap_data import MuseDAPData
from musenap_data import MuseNAPData
from typing import Optional
from maps.musemap import MuseMAP

class measurementMAP:
    """Base class of a measurement map that is to be computed. All measurement maps computed for the MUSE NAP inherit this class"""
    def __init__(self, dap_data: MuseDAPData, nap_data: Optional[MuseNAPData] = None, verbose = False):
        self.dap_data = dap_data
        self.nap_data = nap_data
        self.verbose = verbose

    def compute(self) -> MuseMAP:
        return NotImplementedError("measurement MAPs must implement compute()")
    
