import os
from pathlib import Path
import time
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

def matplotlib_rc() -> str:
    """Returns the path to the Matplotlib style file in config/"""
    style_file = os.path.join(get_default_path('config'), 'figures.mplstyle')
    if not os.path.exists(style_file):
        raise ValueError(f"'figures.mplstyle' not found in config/")
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