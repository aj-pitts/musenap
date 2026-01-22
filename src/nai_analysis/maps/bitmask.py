from typing import Optional
import numpy as np
from utils.util import sys_message

class MuseMapBitMask:
    def __init__(self, additional_bitdefs: Optional[dict[str, int | tuple[int, str]]] = None):
        """
        Bit mask handler for 2D map data.
        
        Parameters
        ----------
        additional_bitdefs : dict[str, int] or dict[str, tuple[int, str]], Optional
            A dictionary mapping flag names to bit indices to be added onto the default bits, optional.
            If `additional_bitdefs` is provided, each bit index must be unique and cannot be one of 
            ... . Any non-unique bit index will be ignored. The values of `additonal_bitdefs` may be
            tuples containing the bit index and a description of the flag.
        """
        self._validate_additional(additional_bitdefs)

        self.additional_bitdefs = additional_bitdefs if additional_bitdefs is not None else {}
        self.add_bits = {
            flag_name : bit 
            if isinstance(bit, int) else bit[0]
            for flag_name, bit in self.additional_bitdefs.items()
            }
        self.add_bit_descriptions = {
            flag_name : value[1]
            if isinstance(value, tuple) else ''
            for flag_name, value in self.additional_bitdefs.items()
        }

        self.bitdefs = self.default_bitdefs() | self.add_bits
        self.bitdescriptions = self.default_bitdescriptions() | self.add_bit_descriptions

    def _validate_additional(self, additional_bitdefs):
        default_bits = list(self.bitdefs.values())
        if additional_bitdefs is not None:
            add_bit_vals = []
            if not isinstance(additional_bitdefs, dict):
                raise ValueError(f"additional_bits must be a dictionary")
            for key, value in additional_bitdefs.items():
                if not isinstance(key, str):
                    raise ValueError("The keys of additional_bits must be strings")
                if isinstance(value, int):
                    if value in default_bits:
                        sys_message(f"Additional Bit {value} is already used by default MuseMapBitMask", status='WARNING', color='red')
                    if value in add_bit_vals:
                        sys_message(f"Additional Bit {value} is repeated", status='WARNING', color='red')
                    add_bit_vals.append(value)
                elif isinstance(value, tuple, int):
                    if not isinstance(value[0], int) or not isinstance(value[1], str):
                        raise ValueError("tuple values of additional_bits must be of type (int, str)")
                    if value[0] in default_bits:
                        sys_message(f"Bit {value[0]} is already used by default MuseMapBitMask", status='WARNING', color='red')
                    if value[0] in add_bit_vals:
                        sys_message(f"Additional Bit {value[0]} is repeated", status='WARNING', color='red')
                    add_bit_vals.append(value[0])
                else:
                    raise ValueError("The values of additional_bits must be integers or tuples")
                

    def flag(self, bitname: str) -> int:
        """Return the integer value of a single flag"""
        return 1 << self.bitdefs[bitname]

    def set_flag(self, bitmask: np.ndarray[np.uint32], condition: np.ndarray[bool], bitname: str | list[str]) -> None:
        """
        Set a bit or bits wherever `condition` is True.

        Parameters
        ----------
        bitmask : np.ndarray
            A 2D numpy array of type uint32 to store bitmask integers
        condition : np.ndarray
            A 2D numpy array of bool used to set bit or bits where `True`
        bitname : str or list of str
            A string of the bit name used for assignment, or a list of strings of bit names. `bitname`
            strings will be formatted for captialization and underscore spacing automatically. 
            Default bit names are:
                "NO_VALUE"
                "DAPMASK"
                "UNRELIABLE"
                "MATH_ERROR"
                "LOW_NAI_SNR"
                "UNCERTAINTY_OOB"
                "DO_NOT_USE"
        """
        if isinstance(bitname, str):
            bitname = [bitname]

        for bitn in bitname:
            bitmask[condition] |= self.flag(bitn.upper().replace(' ', '_'))

    def flagged(self, bitmask: np.ndarray, bitname: str) -> np.ndarray:
        """Return boolean mask where the bit is set"""
        return (bitmask & self.flag(bitname)) != 0

    @staticmethod
    def default_bitdefs() -> dict:
        return {
            "NO_VALUE":0,
            "DAPMASK":1,
            "UNRELIABLE":2,
            "MATH_ERROR":3,
            "LOW_NAI_SNR":4,
            "UNCERTAINTY_OOB":5,
            "DO_NOT_USE":30
        }
    
    
    @staticmethod
    def default_bitdescriptions() -> dict:
        return {
            "NO_VALUE":'No coverage in this spaxel',
            "DAPMASK":'Spaxel covariate DAP MAP was flagged by DAPPIXMASK',
            "UNRELIABLE":'Uncertainty is large or could not be computed',
            "MATH_ERROR":'Mathematical error in computing value',
            "LOW_NAI_SNR":'S/N of Na I is below user-set threshold',
            "UNCERTAINTY_OOB":'Uncertainty is beyond user-set threshold',
            "DO_NOT_USE":'Do not use this spaxel for science'
        }

    def format_header_dict(self) -> dict:
        mask_header = {}
        for flag_name, bit in self.bitdefs.items():
            mask_header[f"BIT_{bit}"] = (flag_name, f'Descrption of mask Bit {bit}')
        return mask_header
            