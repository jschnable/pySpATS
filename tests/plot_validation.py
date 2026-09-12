"""Render diagnostics from the real fitted wheat model into a requested directory."""

import argparse
from pathlib import Path
from pyspats import fit_trial, PSANOVA, variogram, plot_variogram
from pyspats.datasets import load_wheatdata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = fit_trial(
        data=load_wheatdata(),
        response="yield",
        genotype="geno",
        genotype_as_random=True,
        spatial=PSANOVA("col", "row"),
        random=["R", "C"],
    )
    model.plot(show=False).savefig(args.output / "wheat.png", dpi=150)
    plot_variogram(variogram(model), show=False).savefig(
        args.output / "variogram.png", dpi=150
    )
