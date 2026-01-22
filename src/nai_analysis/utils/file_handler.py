from astropy.io import fits
import os
import re
from datetime import datetime
from glob import glob
import numpy as np
import warnings
import configparser
from datetime import datetime
from tqdm import tqdm
import yaml

from musedap_data import MuseDAPData
import util
import defaults

def get_data_paths(galname: str, bin_method: str, require_local = False, verbose=False) -> dict:
    """
    Acquires the path(s) to the primary file(s) of a given galaxy by request.

    Parameters
    ----------
    galname : str
        The name of the galaxy of which the data will be grabbed.

    bin_method : str
        Spatial binning keyword of the binning method of the desired data.

    verbose : bool, optional
        ...
    """
    util.sys_message(f"Acquiring relevant files for {galname}-{bin_method}", verbose=verbose)

    # initialize empty output dictionary
    outdict = {}
    
    # initialize relevant info for path structures
    galsubdir = f"{galname}-{bin_method}"
    analysisplan = defaults.analysis_plans()
    corr_key = 'BETA-CORR'

    # initialize paths to the pipeline data directory and relevant subdirectories
    pipeline_directory = defaults.get_data_path(subdir='pipeline')
    musecubes_parent_dir = os.path.join(pipeline_directory, "muse_cubes")
    dap_parent_dir = os.path.join(pipeline_directory, "dap_outputs")
    mcmc_parent_dir = os.path.join(pipeline_directory, "mcmc_outputs")

    # attach specific galaxy info to paths and check if they exist
    muse_cube_directory = os.path.join(musecubes_parent_dir, galname)
    dap_directory = os.path.join(dap_parent_dir, galsubdir)
    mcmc_directory = os.path.join(mcmc_parent_dir, galsubdir)
    util.check_filepath([muse_cube_directory, dap_directory, mcmc_directory], mkdir=False)

    # get config_file
    configfils = glob(os.path.join(muse_cube_directory, "*.ini"))
    if len(configfils) == 0:
        raise ValueError(f"No config file found for {galname} {bin_method}")
    if len(configfils) > 1:
        raise ValueError(f"More than one config file found for {galname} {bin_method}")

    config_path = configfils[0]
    outdict['CONFIG'] = config_path

    # parse the config file and assign values for filepaths
    default_config = parse_ini_file(config_path)
    plate = default_config['plate']
    ifu = default_config['ifu']

    # construct the dap path tree and attach it to the dap_directory
    dap_outputs_subtree = f"{corr_key}/{bin_method}-{analysisplan}/{plate}/{ifu}"
    dap_outputs_directory = os.path.join(dap_directory, dap_outputs_subtree)

    # get the logcube
    logcube_file = f"manga-{plate}-{ifu}-LOGCUBE-{bin_method}-{analysisplan}.fits"
    logcube_path = os.path.join(dap_outputs_directory, logcube_file)
    util.check_filepath(logcube_path, mkdir=False)
    outdict['LOGCUBE'] = logcube_path

    # get the maps file
    maps_file = f"manga-{plate}-{ifu}-MAPS-{bin_method}-{analysisplan}.fits"
    maps_path = os.path.join(dap_outputs_directory, maps_file)
    util.check_filepath(maps_path, mkdir=False)
    outdict['MAPS'] = maps_path

    # get the mcmc files
    mcmc_outputs_subtree = f"{corr_key}/{analysisplan}"
    mcmc_runs_path = os.path.join(mcmc_directory, mcmc_outputs_subtree)
    mcmc_runs = os.listdir(mcmc_runs_path)

    if len(mcmc_runs) == 0:
        raise ValueError(f"No MCMC runs found in {mcmc_runs_path}")
    if len(mcmc_runs) == 1:
        mcmc_path = os.path.join(mcmc_runs_path, mcmc_runs[0])
    else: # if there are more than one Run subdir, find the one with the most recent date
        dated_dirs = []
        pattern = re.compile(r"^Run_(\d{4}-\d{2}-\d{2})$")
        for run in mcmc_runs:
            match = pattern.match(run)
            if match:
                date_str = match.group(1)
                try:
                    date = datetime.strptime(date_str, "%Y-%m-%d")
                    dated_dirs.append((subdir, date))
                except ValueError:
                    pass # ignore invalid date formats
        if dated_dirs:
            most_recent_run = max(dated_dirs, key=lambda x: x[1])
            mcmc_path = os.path.join(mcmc_runs_path, most_recent_run[0])
        else:
            raise ValueError(f"No valid 'Run_YYYY-MM-DD' subdirectories found in {mcmc_runs_path}")
        

    mcmc_files = glob(os.path.join(mcmc_path), "*.fits")
    if len(mcmc_files) == 0:
        raise ValueError(f"mcmc path {mcmc_path} is an empty directory")
    outdict['MCMC'] = mcmc_files


    # get the file path to the local data
    local_parent_dir = defaults.get_data_path(subdir='local/local_outputs')
    local_data_directory = os.path.join(local_parent_dir, galsubdir, corr_key, analysisplan)
    local_file = f"{galname}-{bin_method}-local_maps.fits"
    local_path = os.path.join(local_data_directory, local_file)
    outdict['LOCAL'] = local_path
    
    # warn if the file does not exist as it is not required
    if not os.path.exists(local_data_directory) or not os.path.isfile(local_path):
        if require_local:
            raise ValueError(f"local file does not exist")
        else:
            raise UserWarning(f"local file does not exist")
        
    return outdict


def get_config_contents(galname: str, bin_method: str) -> dict:
    pass

def parse_ini_file(config_file, solve_parser = True, verbose = False):
    """Parses the .ini configuration files to retrieve data from 'default' section."""
    config = configparser.ConfigParser(allow_no_value=True)

    parsing = True
    while parsing:
        try:
            config.read(config_file)
        except configparser.Error as e:
            if not solve_parser:
                raise configparser.Error(e)
            util.sys_message("Error parsing config file", color='yellow', verbose=verbose)
            util.sys_message(f"Cleaning {config_file}", color='yellow', verbose=verbose)
            clean_ini_file(config_file)

    return config['default']


def clean_ini_file(input_file, overwrite=True):
    """
    Function to clean an .ini configuration file line-by-line if `configparser` returns an error
    while parsing the file.
    """
    if overwrite:
        output_file = input_file
    else:
        fname, fext = os.path.splitext(input_file)
        output_file = f"{fname}_cleaned{fext}"

    print(f"Reading {input_file}")
    with open(input_file, 'r') as file:
        lines = file.readlines()

    print(f"Writing configuration file to {output_file}...")
    with open(output_file, 'w') as file:
        section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith('[') and line.endswith(']'):
                section = line
                file.write(line + '\n')
            elif '=' in line:
                file.write(line + '\n')
            else:
                if section:
                    file.write(line + '\n')

    print("Done.")

def write_to_fits(musedata: MuseDAPData, verbose = False) -> None:
    return