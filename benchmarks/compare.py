"""Matched Python/R benchmark, with serial repetitions and an identical input CSV.

Run: OPENBLAS_NUM_THREADS=1 python benchmarks/compare.py --output /tmp/spats-compare
Requires R SpATS 1.0-20. Timings exclude imports and CSV I/O, include the fit's
basis construction, covariance and diagnostics. Outputs raw timings and
agreement measures. Fit repetitions run sequentially to avoid contention.
"""

import argparse
import gc
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter

from benchmark import data
import numpy as np
import pandas as pd
import scipy
from pyspats import fit_trial, PSANOVA, SpATSControl, __version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.output / "field.csv"
    data(100, 100, 2000).to_csv(source, index=False)
    frame = pd.read_csv(source)
    timings = []
    for random in (False, True):
        for run in range(1, args.repeats + 1):
            gc.collect()
            start = perf_counter()
            model = fit_trial(
                data=frame,
                response="yield",
                genotype="geno",
                genotype_as_random=random,
                spatial=PSANOVA("col", "row", nseg=(10, 10), nest_div=2),
                control=SpATSControl(max_iter=500, tolerance=1e-6),
            )
            elapsed = perf_counter() - start
            record = dict(
                language="Python",
                random=random,
                run=run,
                seconds=elapsed,
                iterations=model.n_iterations,
            )
            if not model.converged:
                raise RuntimeError("Python benchmark did not converge")
            timings.append(record)
            print(json.dumps(record), flush=True)
            if run == args.repeats:
                pd.DataFrame({"fitted": model.fitted_values}).to_csv(
                    args.output / f"python_fitted_{str(random).upper()}.csv",
                    index=False,
                )
            del model
    subprocess.run(
        [
            "Rscript",
            str(Path(__file__).with_name("compare_r.R")),
            str(source.resolve()),
            str(args.output.resolve()),
            str(args.repeats),
        ],
        check=True,
    )
    rtimings = pd.read_csv(args.output / "r_timings.csv")
    results = pd.concat([pd.DataFrame(timings), rtimings], ignore_index=True)
    results.to_csv(args.output / "timings.csv", index=False)
    agreement = {}
    for random in (False, True):
        tag = str(random).upper()
        py = pd.read_csv(args.output / f"python_fitted_{tag}.csv").fitted
        r = pd.read_csv(args.output / f"r_fitted_{tag}.csv").fitted
        agreement[tag] = float(np.max(np.abs(py - r)))
    report = dict(
        pyspats=__version__,
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        pandas=pd.__version__,
        platform=platform.platform(),
        thread_environment={
            key: os.environ.get(key)
            for key in [
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            ]
        },
        max_absolute_fitted_difference=agreement,
    )
    (args.output / "metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    print(results.groupby(["language", "random"]).seconds.agg(["median", "min", "max"]))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
