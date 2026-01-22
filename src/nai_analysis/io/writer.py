from astropy.io import fits
from maps.musemap import MuseMAP
from utils import defaults, util
from typing import Optional
import os
import astropy.io.fits.header
import datetime

def map_to_fits(muse_map: MuseMAP, galname: str, bin_method: str, outpath: Optional[str] = None, verbose = False) -> None:
    """Write MuseMAP data to HDU's into a fits file. Optionally specify a separate full path to write the maps to."""
    if outpath is None:
        local_path = os.path.join(defaults.get_data_path(), 'local')
        analysis_plans = defaults.analysis_plans()
        corr_key = defaults.corr_key()

        fullpath = os.path.join(local_path, f"{galname}-{bin_method}", analysis_plans, corr_key)
        util.check_filepath(fullpath, verbose=verbose)
        
        filename = f"{galname}-{bin_method}-NAP-MAPs.fits"
        outpath = os.path.join(fullpath, filename)

    if not os.path.isfile(outpath):
        hdul = fits.HDUList([fits.PrimaryHDU(header=None)])
        hdul.writeto(outpath, overwrite=True)
    
    with fits.open(outpath, mode="update") as hdul:
        data_header = dict_to_header(muse_map.header_dict)
        hdul.append(fits.ImageHDU(data=muse_map.data, header=data_header, name=muse_map.name))

        mask_header = dict_to_header(muse_map.mask_header)
        mask_header['UPDATED'] = datetime.datetime.now(datetime.timezone.utc)
        hdul.append(fits.ImageHDU(data=muse_map.mask, header=mask_header, name=f"{muse_map.name}_MASK"))

        error_header = dict_to_header(muse_map.error_header)
        hdul.append(fits.ImageHDU(data=muse_map.error, header=error_header, name=f"{muse_map.name}_ERROR"))

def dict_to_header(header_dict: dict[str, tuple[object, str]], fits_header: Optional[astropy.io.fits.header.Header] = None) -> astropy.io.fits.header.Header:
    header = fits.Header() if fits_header is None else fits_header
    for key, value_pair in header_dict.items():
        header[key.upper()] = value_pair
    return header
    
def update_hdu(filepath: str, data: object, name: str, header: Optional[astropy.io.fits.header.Header] = None, overwrite = True, verbose = False) -> None:
    if os.path.exists(filepath):
        with fits.open(filepath, mode="update") as hdul:
            if name in hdul:
                del hdul[name]

            hdu = fits.ImageHDU(data=data, name=name, header=header)
            hdul.append(hdu)
            util.sys_message(f"Updated existing HDU: {name} in {filepath}")
    
    else:
        primary = fits.PrimaryHDU()
        hdu = fits.ImageHDU(data=data, name=name, header=header)
        hdul = fits.HDUList([primary, hdu])
        hdul.writeto(filepath, overwrite=overwrite)


