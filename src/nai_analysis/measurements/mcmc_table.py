from measurement_map import MeasurementMAP
from typing import TYPE_CHECKING
import numpy as np
from glob import glob
import re
from astropy.io import fits
from astropy.table import Table
from concurrent.futures import ThreadPoolExecutor, as_completed


class MCMCTable(MeasurementMAP):

    name = "mcmc_table"
    dependencies = []

    def compute(self) -> np.ndarray:
        DAP = self.dap_data

        mcmc_dir = DAP.mcmc_dir
        mcmc_paths = glob(mcmc_dir)

        sorted_paths = self.sort_paths

        row_counts = []
        for path in sorted_paths:
            with fits.open(path, memmap=True) as hdul:
                row_counts.append(hdul[1].header['NAXIS2'])

        total_rows = sum(row_counts)
        start_ids = np.concatenate([[0], np.cumsum(row_counts[:-1])])

        dtype = [
            ('id', int), ('bin', int), 
            ('velocities', float), ('lambda samples', object), ('percentiles', object)
        ]

        result = np.empty(total_rows, dtype=dtype)

        def load_file(args):
            path, start_id, count = args
            with fits.open(path, memmap=True) as hdul:
                tbl = Table(hdul[1].data)
                sl = slice(start_id, start_id + count)

                result['id'][sl] = np.arange(start_id, start_id + count)
                result['bin'][sl] = tbl['bin']
                result['velocities'][sl] = tbl['velocities']
                result['percentiles'][sl] = list(tbl['percentiles'])

                samples = np.array(tbl['samples'])
                flat = samples[:, :, 1000:, 0].reshape(len(tbl), -1)
                result['lambda samples'][sl] = list(flat)     

        args = list(zip(sorted_paths, start_ids, row_counts))

        nworkers = 8
        with ThreadPoolExecutor(max_workers=nworkers) as executor:
            futures = [executor.submit(load_file, a) for a in args]

            for f in futures:
                f.result()

        return result
    
    @staticmethod
    def sort_paths(mcmc_paths):
        def extract_run_number(filepath):
            match = re.search(r"-run-(\d+)\.fits$", filepath)
            return int(match.group(1)) if match else float('inf')

        sorted_paths = sorted(mcmc_paths, key=extract_run_number)
        return sorted_paths    