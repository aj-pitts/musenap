import numpy as np
from astropy.io import fits
from utils import defaults, file_handler, util
from functools import cached_property

from musedap_data import MuseDAPData
from musemap import MuseMAP

class MuseNAPData:
    def __init__(self, galaxy_name: str, binning_method: str, analysis_plans: str):
        pass

    @staticmethod
    def _validate():
        pass

    @classmethod
    def from_DAP_data(cls, DAP_data: MuseDAPData):
        return cls()
    