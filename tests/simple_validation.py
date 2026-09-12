"""Run the real wheat example and check fitted output integrity."""

import numpy as np
from pyspats import fit_trial, PSANOVA
from pyspats.datasets import load_wheatdata

if __name__ == "__main__":
    model = fit_trial(
        data=load_wheatdata(),
        response="yield",
        genotype="geno",
        genotype_as_random=True,
        spatial=PSANOVA("col", "row"),
        random=["R", "C"],
    )
    assert model.converged and np.isfinite(model.fitted_values).all()
    assert 0 <= model.heritability <= 1
    print(model)
    print(model.summary())
