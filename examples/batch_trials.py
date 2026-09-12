"""Run: python examples/batch_trials.py input.csv output_directory

Input columns: trial, yield, height, genotype, col, row.
Set OPENBLAS_NUM_THREADS=1 before Python when using multiple processes.
"""

import argparse
from pathlib import Path
import pandas as pd
from pyspats import fit_trials, PSANOVA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, job in enumerate(
        fit_trials(
            data=data,
            by="trial",
            responses=["yield", "height"],
            genotype="genotype",
            genotype_as_random=True,
            spatial=PSANOVA("col", "row", nseg=10, nest_div=2),
            workers=args.workers,
            on_error="record",
        )
    ):
        prefix = f"job_{i:06d}"
        record = {
            "prefix": prefix,
            "trial": job.key[0],
            "trait": job.response,
            "ok": job.ok,
            "error": job.error,
        }
        if job.model is not None:
            job.model.to_frame().to_csv(output / f"{prefix}_plots.csv", index=False)
            job.model.genotype_predictions().to_csv(
                output / f"{prefix}_genotypes.csv", index=False
            )
            job.model.summary().to_csv(output / f"{prefix}_components.csv")
            record.update(
                converged=job.model.converged,
                iterations=job.model.n_iterations,
                observations=job.model.n_obs,
                dispersion=job.model.psi,
            )
        manifest.append(record)
    pd.DataFrame(manifest).to_csv(output / "manifest.csv", index=False)
    if any(not r["ok"] for r in manifest):
        raise SystemExit("Some fits failed or did not converge; see manifest.csv")


if __name__ == "__main__":
    main()
