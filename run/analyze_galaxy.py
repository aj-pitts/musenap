import argparse
import os

from src.nai_analysis.utils import defaults
from src.nai_analysis.musedap_data import MuseDAPData
from src.nai_analysis.musenap_data import MuseNAPData
from src.nai_analysis.engine.measurement_engine import MeasurementEngine
from src.nai_analysis.measurements.registry import measurement_registry

from src.nai_analysis.utils import util

def validate_paths(galname, bin_method, verbose = True) -> None:
    util.sys_message(f"Checking output paths...", verbose=verbose)
    data_directory = defaults.get_data_path()
    local = os.path.join(data_directory, 'local/nap_outputs')
    util.check_filepath(local, verbose=verbose)
    analysis = defaults.analysis_plans()
    output = os.path.join(local, f"{galname}-{bin_method}", "BETA-CORR", analysis)
    util.check_filepath(output, verbose=verbose)
    figures = os.path.join(output, 'figures')
    util.check_filepath(figures, verbose=verbose)
    map_figs = os.path.join(figures, 'maps')
    hist_figs = os.path.join(figures, 'hists')
    util.check_filepath(map_figs, verbose=verbose)
    util.check_filepath(hist_figs, verbose=verbose)

def analyze(galname, bin_method, verbose = False):
    validate_paths(galname, bin_method, verbose=verbose)

    muse_data = MuseDAPData.from_name(galaxy_name=galname, binning_method=bin_method, verbose=verbose)
    engine = MeasurementEngine(muse_data, verbose=verbose)
    registry = measurement_registry

    util.sys_message(f"Beginning pipeline for {galname} {bin_method}", verbose=verbose)
    for name in registry.keys():
        result = engine.get(name)
        result.plot_data(verbose=verbose)
        result.write_to_fits(verbose=verbose)

def get_args():
    parser = argparse.ArgumentParser(description="Run the pipeline for a galaxy")

    parser.add_argument('galname', type=str, help="Input galaxy name.")
    parser.add_argument('bin_method', type=str, help="Input DAP spatial binning method.")
    parser.add_argument('-v','--verbose', help = "Print verbose outputs (default: False)", action='store_true', default = False)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    analyze(args.galname, args.bin_method, args.verbose)
