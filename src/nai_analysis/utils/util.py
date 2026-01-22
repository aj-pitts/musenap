import os
import sys
from typing import Optional, Union, List
import numpy as np

#### INFO OUTPUT AND FILEPATHS
def sys_message(message, status = 'INFO', color = 'reset', verbose = True) -> None:
    """Prints a verbose message to the terminal"""

    colors = {
        'yellow':'\033[93m',
        'green':'\033[32m',
        'red':'\033[31m', 
        'blue':'\033[34m',
        'magenta':'\033[35m',
        'cyan':'\033[36m',
        'reset':'\033[0m'
    }
    if verbose:
        print(f"{colors[color]}{status[:4]}{colors['reset']}: {message}", file=sys.stderr)
    return

def check_filepath(filepath: str | list[str], mkdir = True, verbose = False) -> None:
    """Check if a filepath or list of filepaths exists. If not, the filepath is make if mkdir=True,
    otherwise a ValueError is raised."""
    if isinstance(filepath, str):
        filepath = [filepath]

    for path in filepath:
        if not os.path.exists(path):
            if mkdir:
                os.makedirs(path)
                sys_message(f"Creating filepath: {path}", verbose=verbose)
            else:
                raise ValueError(f"{path} does not exist")

#### DAP MASKS 
def DAP_spec_mask(DAPSPECMASK):
    return

def DAP_pix_mask(DAPPIXMASK: np.ndarray | int, bit_list: Optional[list[int]] = None) -> np.ndarray | bool:
    """Handles the `DAPPIXMASK` bitmask values of the MaNGA DAP data products. Only pixels flagged
    by bit 30 ("DO NOT USE") are masked unless a list of bits to mask is provided."""

    if isinstance(DAPPIXMASK, int):
        return bool(DAPPIXMASK & (1 << 30))
    
    elif isinstance(DAPPIXMASK, np.ndarray):
        return (DAPPIXMASK & 1 << 30) != 0

    else:
        raise ValueError("Invalid input for DAPPIXMASK")

#### DATA HELPERS
def maks_arrays(truth_array: Union[List[bool], np.ndarray], *arrays: np.ndarray) -> tuple:
    """Filter array(s) using a boolean truth array."""
    
    if len(arrays) == 0:
        raise ValueError("At least one array must be provided")
    
    for i, arr in enumerate(arrays):
        if len(arr) != len(truth_array):
            raise ValueError(f"Truth array does not correspond element-wise to array {i}")
        
    return tuple(arr[truth_array] for arr in arrays)


def get_unique_bin_values(measurement_map: np.ndarray | dict, binmap: np.ndarray, mask: Optional[np.ndarray] = None,
                         return_bins = False, return_masked = False) -> np.ndarray | dict:
    """Extracts the unique, and optionally unmasked, values of spatially-binned 2D measurement maps."""

    map_mask = mask.astype(bool) if mask is not None else np.zeros_like(binmap).astype(bool)
    map_mask = np.logical_or(map_mask, binmap == -1)

    good_bins = binmap[~map_mask]
    unique_bins, bin_inds = np.unique(good_bins, return_index=True)

    if return_masked:
        bad_bins = binmap[map_mask]
        unique_bins_bad, bin_inds_bad = np.unique(bad_bins, return_index=True)

    if isinstance(measurement_map, np.ndarray):
        select = measurement_map[~map_mask]
        unique_select = select[bin_inds]

        if return_masked:
            rejected = measurement_map[map_mask]
            unique_reject = rejected[bin_inds_bad]

            if return_bins:
                return unique_select, unique_bins, unique_reject, unique_bins_bad
            return unique_select, unique_reject
        if return_bins:
            return unique_select, unique_bins
        return unique_select

    if isinstance(measurement_map, dict):
        outdict = {}
        for key, m_map in measurement_map.items():
            select = m_map[~map_mask]
            outdict[key]
            if return_masked:
                rejected = m_map[map_mask]
                outdict[f'{key}_rejected'] = rejected

        if return_bins:
            outdict['bins'] = unique_bins
            if return_masked:
                outdict['bins_masked'] = unique_bins_bad

        return outdict

    else:
        raise ValueError("measurement_map must be a 2D array or dictionary of 2D arrays.")

def get_bin_coords(binID: int, binmap: np.ndarray, return_truth = False) -> tuple:
    """
    Returns a tuple of `(y, x)` pixel coordinates of the first occurence of where binID == binmap
    Optionally return `(w, y, x)` where `w` is the 2D boolean array.
    """

    w = binID == binmap
    ys, xs = np.where(w)
    if return_truth:
        return (w, ys[0], xs[0])
    
    return (ys[0], xs[0])