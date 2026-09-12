"""A complete, reproducible analysis of the real R SpATS wheat trial."""

from pyspats import fit_trial, PSANOVA
from pyspats.datasets import load_wheatdata


def main():
    result = fit_trial(
        data=load_wheatdata(),
        response="yield",
        genotype="geno",
        genotype_as_random=True,
        spatial=PSANOVA("col", "row", nseg=10),
        random=["R", "C"],
    )
    print(result)
    print(result.summary())
    print(f"Generalized heritability: {result.heritability:.4f}")
    print(result.genotype_predictions().head())


if __name__ == "__main__":
    main()
