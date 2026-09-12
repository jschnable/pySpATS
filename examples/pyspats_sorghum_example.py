"""Fit the bundled sorghum example; export current results to an explicit directory."""

import argparse
from pathlib import Path
import pandas as pd
from pyspats import fit_trial, PSANOVA, SpATSControl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trait", default="EstimatedPlotYield")
    args = parser.parse_args()
    data = pd.read_csv(Path(__file__).with_name("sorghum_data.csv"))
    result = fit_trial(
        data=data,
        response=args.trait,
        genotype="PINumber",
        genotype_as_random=True,
        spatial=PSANOVA("Column", "Row", nseg=(10, 10), nest_div=2),
        fixed=["Treatment"],
        random=["Block"],
        control=SpATSControl(max_iter=1000, tolerance=1e-7),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    result.to_frame().to_csv(args.output / "plots.csv", index=False)
    result.genotype_predictions().to_csv(args.output / "genotypes.csv", index=False)
    result.summary().to_csv(args.output / "components.csv")
    print(result)
    print(result.summary())
    print(f"Generalized heritability: {result.heritability:.4f}")
    if not result.converged:
        raise SystemExit("Fit did not converge; inspect results before use")


if __name__ == "__main__":
    main()
