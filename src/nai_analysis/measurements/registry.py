from typing import Type, Dict
from src.nai_analysis.measurements.measurement_map import MeasurementMAP
from src.nai_analysis.measurements.sigma_sfr import SigmaSfrMAP
from src.nai_analysis.measurements.redshift import RedshiftMAP
from src.nai_analysis.measurements.nai_velocity import VcenMAP, VmaxMAP, VfracMAP, VcensysMAP
from src.nai_analysis.measurements.nai_snr import SnrMAP
from src.nai_analysis.measurements.nai_ew import WeqMAP, WeqAbsMAP, WeqEmMAP
from src.nai_analysis.measurements.mcmc import LambdaMAP, LogNMAP, bDMAP, CfMAP
from src.nai_analysis.measurements.metallicity import MetallicityMAP
from src.nai_analysis.measurements.bpt import BPTMap

measurement_registry: Dict[str, Type[MeasurementMAP]] = {
    "redshift":RedshiftMAP,
    "snr_nai":SnrMAP,
    "sfrsd":SigmaSfrMAP,
    "weq_nai":WeqMAP,
    "weq_abs_nai":WeqAbsMAP,
    "weq_em_nai":WeqEmMAP,
    "lambda":LambdaMAP,
    "logn":LogNMAP,
    "bd":bDMAP,
    "cf":CfMAP,
    "v_cen":VcenMAP,
    "v_max":VmaxMAP,
    "v_frac":VfracMAP,
    "v_sys":VcensysMAP,
    "metallicity":MetallicityMAP,
    "bpt":BPTMap
}