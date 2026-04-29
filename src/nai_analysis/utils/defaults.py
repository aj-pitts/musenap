import os
from pathlib import Path
import json
from typing import Optional

def get_default_path(subdirectory: Optional[str] = None) -> str:
    """returns the root path of nai_analysis, or a subdirectory of the root"""
    root_path = Path(__file__).parent.parent
    requested_path = os.path.join(root_path, subdirectory) if subdirectory is not None else root_path
    if not os.path.exists(requested_path):
        raise ValueError(f"Requested path does not exist: {requested_path}")
    
    return requested_path

def get_data_path() -> str:
    """Returns the root path to the data directory specified by 'data_path' in config/paths.json"""
    config_dir = get_default_path('config')
    json_file = os.path.join(config_dir, 'paths.json')

    with open(json_file) as js:
        path_config = json.load(js)
    
    data_path = path_config.get('data_path', None)
    if data_path is None:
        raise KeyError(f"No 'data_path' found in confgi/paths.json")

    if not os.path.exists(data_path):
        raise ValueError(f"data_path does not exist! Update config/paths.json")
    
    return data_path

def get_local_data_path() -> str:
    """Returns the root path to the local data directory, a subdirectory of 'data_path' in config/paths.json"""
    datapath = get_data_path()
    local = os.path.join(datapath, 'local')

    if not os.path.exists(local):
        raise ValueError(f"Local Path does not exist! {local}")
    
    return local

def get_nap_outputs_path() -> str:
    local = get_local_data_path()
    nap_outputs = os.path.join(local, 'nap_outputs')

    if not os.path.exists(nap_outputs):
        raise ValueError(f"NAP outputs path does not exist {nap_outputs}")
    
    return nap_outputs

def get_local_galaxy_dir(galname: str, bin_method: str, analysis_plan: str = None) -> str:
    """
    Returns the root path to the local data directory of a specific galaxy
    Assumes the local directory is a subdirectory of 'data_path' in config/paths.json
    """
    nap_out = get_nap_outputs_path()
    analysis = analysis_plans() if analysis_plan is None else analysis_plan
    subdir = os.path.join(nap_out, f"{galname}-{bin_method}", "BETA-CORR", analysis)
    if not os.path.exists(subdir):
        raise ValueError(f"Galaxy directory not found: {subdir}")
    return subdir

def get_default_filename(galname: str, bin_method: str) -> str:
    """Returns the NAP output FITS filename for the input galaxy"""
    return f"{galname}-{bin_method}-NAP_MAPS.fits"

def get_pipeline_data_path() -> str:
    """Returns the root path to the pipeline data directory, a subdirectory of 'data_path' in config/paths.json"""
    datapath = get_data_path()
    pipeline = os.path.join(datapath, 'pipeline')

    if not os.path.exists(pipeline):
        raise ValueError(f"Pipeline Path does not exist! {pipeline}")
    
    return pipeline

def matplotlib_rc() -> str:
    """Returns the path to the Matplotlib style file in config/"""
    style_file = os.path.join(get_default_path('plotting'), 'figures.mplstyle')
    if not os.path.exists(style_file):
        raise ValueError(f"'figures.mplstyle' not found in {style_file}")
    return style_file

def analysis_plans() -> str:
    """Returns a string of the default DAP analysis plan methods: `MILESHC-MASTARSSP-NOISM`"""
    return 'MILESHC-MASTARSSP-NOISM'

def corr_key() -> str:
    """Returns a string of the default DAP correlation corrected status: `BETA-CORR`"""
    return 'BETA-CORR'

def default_quality_flag(bit: int) -> str:
    """Returns a string of the data quality flag based identified by the input bit"""
    return 