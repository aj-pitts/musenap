import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table, join, Column
import pandas as pd
from glob import glob
import os
import re
from datetime import datetime
import argparse
from tqdm import tqdm
from modules import defaults, file_handler, util, plotter, inspect

from genesis_metallicity.genesis_metallicity import genesis_metallicity

def get_args():
    parser = argparse.ArgumentParser(description="A script to create an Na I velocity map from the MCMC results of a beta-corrected galaxy.")

    parser.add_argument('galname', type=str, help="Input galaxy name.")
    parser.add_argument('bin_method', type=str, help="Input DAP spatial binning method.")
    parser.add_argument('-v','--verbose', help = "Print verbose outputs (default: False)", action='store_true', default = False)

    return parser.parse_args()

def make_metallicity_map(galname, bin_method, verbose = False):
    datapaths = file_handler.init_datapaths(galname, bin_method)

    with fits.open(datapaths['MAPS']) as hdul:
        spatial_bins = hdul['binid'].data[0]
        emlines = hdul['emline_gflux'].data
        emlines_ivar = hdul['emline_gflux_ivar'].data
        emlines_mask = hdul['emline_gflux_mask'].data

        emlines_ew = hdul['emline_gew'].data
        emlines_ew_ivar = hdul['emline_gew_ivar'].data
        emlines_ew_mask = hdul['emline_gew_mask'].data

    with fits.open(datapaths['LOCAL']) as hdul:
        redshift = hdul['redshift'].data
    
    metallicity = np.zeros_like(spatial_bins)
    metallicity_error = np.zeros_like(spatial_bins)
    metallicity_mask = np.zeros_like(spatial_bins)

    ## TODO update masking!!!
    for binID in np.unique(spatial_bins):
        if binID == -1:
            continue
        w = binID == spatial_bins
        ny, nx = np.where(w)
        y, x = ny[0], nx[0]

        emlines_bin = emlines[:, y, x]
        emlines_ivar_bin = emlines_ivar[:, y, x]
        emlines_mask_bin = emlines_mask[:, y, x]

        ew_bin = emlines_ew[:, y, x]
        ew_ivar_bin = emlines_ew_ivar[:, y, x]
        ew_mask_bin = emlines_ew_mask[:, y, x]

        input_dict = {}
        input_dict['redshift'] = redshift[y, x]

        input_dict['OII']      = [emlines_bin[0], 1 / np.sqrt(emlines_ivar_bin[0])] # 'OII-3727', 'OII-3729' in channel 2
        input_dict['Hdelta']   = [emlines_bin[11], 1 / np.sqrt(emlines_ivar_bin[11])]
        input_dict['Hgamma']   = [emlines_bin[12], 1 / np.sqrt(emlines_ivar_bin[12])]
        #input_dict['O4363']    = 
        input_dict['Hbeta']    = [emlines_bin[14], 1 / np.sqrt(emlines_ivar_bin[14])]
        input_dict['O4959']    = [emlines_bin[15], 1 / np.sqrt(emlines_ivar_bin[15])]
        input_dict['O5007']    = [emlines_bin[16], 1 / np.sqrt(emlines_ivar_bin[16])]
        input_dict['Hbeta_EW'] = [ew_bin[0], 1 / np.sqrt(ew_ivar_bin[0])]

        galaxy = genesis_metallicity(input_dict=input_dict, object=galname)

        metallicity[w] = galaxy.metallicity.n
        metallicity_error[w] = galaxy.metallicity.s



    