"""Reproducible Gaussian trial timings; run from repository root.

OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python benchmarks/benchmark.py
Reports fitting (including basis, covariance and diagnostics), no CSV I/O.
"""

import json
import platform
from time import perf_counter
import numpy as np
import pandas as pd
from pyspats import fit_trial, PSANOVA, SpATSControl


def data(rows, cols, genotypes):
    rng = np.random.default_rng(421)
    row, col = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    n = rows * cols
    geno = np.resize(np.arange(genotypes), n)
    rng.shuffle(geno)
    effects = rng.normal(0, 2, genotypes)
    return pd.DataFrame(
        {
            "row": row.ravel(),
            "col": col.ravel(),
            "geno": geno,
            "yield": 20
            + effects[geno]
            + np.sin(row.ravel() / 5)
            + np.cos(col.ravel() / 6)
            + rng.normal(0, 1, n),
        }
    )


if __name__ == "__main__":
    records = []
    for rows, cols, g in [(16, 12, 24), (50, 40, 400), (100, 100, 2000)]:
        d = data(rows, cols, g)
        for random in (False, True):
            t = perf_counter()
            model = fit_trial(
                data=d,
                response="yield",
                genotype="geno",
                genotype_as_random=random,
                spatial=PSANOVA("col", "row", nseg=(10, 10), nest_div=2),
                control=SpATSControl(max_iter=500, tolerance=1e-6),
            )
            records.append(
                dict(
                    n=len(d),
                    genotypes=g,
                    random=random,
                    seconds=perf_counter() - t,
                    iterations=model.n_iterations,
                    converged=model.converged,
                    numpy=np.__version__,
                    python=platform.python_version(),
                    machine=platform.machine(),
                )
            )
            print(json.dumps(records[-1]), flush=True)
