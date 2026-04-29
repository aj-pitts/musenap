import argparse
import os
from src.nai_analysis.utils import util, defaults
from run.analyze_galaxy import validate_paths, analyze

def get_args():
    parser = argparse.ArgumentParser(description="Run the NAP on the available sample of galaxies if their files exist")

    parser.add_argument('bin_method', type=str, help="Input DAP spatial binning method.")
    parser.add_argument('-v','--verbose', help = "Print verbose outputs (default: False)", action='store_true', default = False)
    parser.add_argument('--dryrun', help = "Dry run of the sample: computes data but does not save data or save plots", action='store_true', default=False)

    return parser.parse_args()

def run_nap(bin_method, dry_run = False, verbose = False):
    util.sys_message(f"Acquiring sample", verbose=verbose)
    data = defaults.get_data_path()
    pipeline_data = os.path.join(data, 'pipeline')
    muse_cubes = os.path.join(pipeline_data, 'muse_cubes')
    
    cube_names = os.listdir(muse_cubes)

    for name in cube_names:
        cube_path = os.path.join(muse_cubes, name)
        if not os.path.isdir(cube_path):
            continue
        
        try:
            stage = "VALIDATE_PATHS"
            validate_paths(name, bin_method, verbose=verbose)

            stage = "ANALYZE"
            analyze(name, bin_method, dry_run=dry_run, verbose=verbose)
        except:
            util.sys_message(f"Cube {name} failed at {stage}", status='WARN', color='yellow')
            continue

if __name__ == "__main__":

    args = get_args()
    run_nap(args.bin_method, args.dryrun, args.verbose)