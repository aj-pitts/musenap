from .sigma_sfr import SigmaSfrMAP
from .redshift import RedshiftMAP
from .nai_velocity import VcenMAP, VmaxMAP, VfracMAP
from .nai_snr import SnrMAP
from .nai_ew import WeqMAP, WeqAbsMAP, WeqEmMAP
from .mcmc import LambdaMAP, LogNMAP, bDMAP, CfMAP

measurement_registry = {
    "redshift":RedshiftMAP,
    "sfrsd":SigmaSfrMAP,
    "vcen":VcenMAP,
    "vmax":VmaxMAP,
    "vfrac":VfracMAP,
    "snr_nai":SnrMAP,
    "weq_nai":WeqMAP,
    "weq_abs_nai":WeqAbsMAP,
    "weq_em_nai":WeqEmMAP,
    "lambda":LambdaMAP,
    "logn":LogNMAP,
    "bd":bDMAP,
    "cf":CfMAP
}