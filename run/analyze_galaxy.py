import argparse
from src.nai_analysis.musedap_data import MuseDAPData
from src.nai_analysis.musenap_data import MuseNAPData
from src.nai_analysis.engine.measurement_engine import MeasurementEngine

def analyze(galname, bin_method, verbose = False):
    muse_data = MuseDAPData.from_name(galaxy_name=galname, binning_method=bin_method, verbose=verbose)
    engine = MeasurementEngine(muse_data, verbose=verbose)
    return

def get_args():
    parser = argparse.ArgumentParser(description="A script to handle the HII region collaboration.")

    parser.add_argument('galname', type=str, help="Input galaxy name.")
    parser.add_argument('bin_method', type=str, help="Input DAP spatial binning method.")
    parser.add_argument('-v','--verbose', help = "Print verbose outputs (default: False)", action='store_true', default = False)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    analyze(args.galname, args.bin_method)
