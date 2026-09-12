import pandas as pd
import numpy as np
import pytest
from pyspats import fit_trials, PSANOVA
from .test_r_parity import ROOT


@pytest.mark.parametrize("workers", [1, 2])
def test_batch_preserves_identity_and_errors(workers):
    d = pd.read_csv(ROOT / "field.csv")
    d = pd.concat([d.assign(field="one"), d.assign(field="two")])
    d["bad"] = np.nan
    records = list(
        fit_trials(
            data=d,
            by="field",
            responses=["yield", "bad"],
            workers=workers,
            on_error="record",
            genotype="geno",
            spatial=PSANOVA("col", "row", nseg=4),
        )
    )
    assert [(r.key, r.response) for r in records] == [
        (("one",), "yield"),
        (("one",), "bad"),
        (("two",), "yield"),
        (("two",), "bad"),
    ]
    assert records[0].ok and records[2].ok
    assert records[1].model is None and "No observations" in records[1].error
    np.testing.assert_allclose(
        records[0].model.fitted_values, records[2].model.fitted_values
    )
