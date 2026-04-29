import argparse
from src.nai_analysis.musenap_data import MuseNAPData
from src.nai_analysis.plotting import custom_plots


def get_args():
    parser = argparse.ArgumentParser(description="Run the pipeline for a galaxy")

    parser.add_argument('galname', type=str, help="Input galaxy name.")
    parser.add_argument('bin_method', type=str, help="Input DAP spatial binning method.")
    parser.add_argument('-v','--verbose', help = "Print verbose outputs (default: False)", action='store_true', default = False)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    muse_data = MuseNAPData(args.galname, args.bin_method, verbose=args.verbose)
    dap_data = muse_data.dap_data

    dap_data.plot_maps()
    muse_data.plot_grid()

    

