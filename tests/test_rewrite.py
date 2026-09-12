"""Regression, invariance and independent linear-algebra checks for the rewrite."""

import pickle
import numpy as np
import pandas as pd
import pytest
from pyspats import *
from pyspats.engine import fit_mixed
from .test_r_parity import ROOT, fit_case


def test_covariance_and_schur_against_full_mixed_equations():
    m = fit_case("PSANOVA_random")
    A = np.column_stack((m._X, m._Z))
    p = m._X.shape[1]
    prec = (m._penalties / np.array(list(m.var_comp.values()))[:, None]).sum(0)
    H = A.T @ A / m.psi + np.diag(np.r_[np.zeros(p), prec])
    np.testing.assert_allclose(
        H @ m.coefficients, A.T @ m.data["yield"] / m.psi, rtol=1e-9, atol=1e-8
    )
    np.testing.assert_allclose(H @ m.covariance, np.eye(len(H)), atol=2e-6)
    expected = np.diag(A @ np.linalg.solve(H, A.T))
    np.testing.assert_allclose(
        m.predict(return_se=True).se_link ** 2, expected, rtol=1e-8, atol=1e-8
    )
    # Observation-space marginal GLS is independent of the coefficient solve.
    X, Z = m._X, m._Z
    V = m.psi * np.eye(len(A)) + (Z / prec) @ Z.T
    ViX = np.linalg.solve(V, X)
    beta = np.linalg.solve(X.T @ ViX, ViX.T @ m.data["yield"])
    np.testing.assert_allclose(m.coefficients[:p], beta, rtol=1e-8, atol=1e-8)


def test_missing_alignment_and_input_unchanged():
    d = pd.read_csv(ROOT / "field.csv")
    d.index = (
        np.arange(len(d)) // 2
    )  # Duplicate labels must not break positional masks.
    d.iloc[0, d.columns.get_loc("yield")] = np.nan
    d.iloc[1, d.columns.get_loc("row")] = np.nan
    before = d.copy(deep=True)
    m = fit_trial(
        data=d, response="yield", genotype="geno", spatial=PSANOVA("col", "row", nseg=4)
    )
    pd.testing.assert_frame_equal(d, before)
    assert np.isfinite(m.fitted_values[0]) and np.isnan(m.fitted_values[1])
    assert np.isnan(m.residuals[0]) and m.n_obs == len(d) - 2
    assert m.to_frame().index.equals(d.index)


def test_prediction_persistence_permutation_and_offset():
    m = fit_case("weighted")
    order = np.random.default_rng(1).permutation(len(m.data))
    new = m.data.iloc[order]
    np.testing.assert_allclose(
        m.predict(new, offset="off"), m.fitted_values[order], atol=1e-12
    )
    saved = pickle.loads(pickle.dumps(m))
    np.testing.assert_allclose(
        saved.predict(new, offset="off"), m.fitted_values[order], atol=1e-12
    )
    with pytest.raises(ValueError, match="unseen"):
        m.predict(new.assign(geno="never seen"))
    with pytest.raises(ValueError, match="outside"):
        m.predict(new.assign(col=1000))


def test_offset_equivalence_and_response_units():
    m = fit_case("weighted")
    d = m.data.copy()
    d["yield"] -= d.off
    kw = dict(
        genotype="geno",
        spatial=m.spec,
        genotype_as_random=True,
        fixed=["treatment"],
        random=["block"],
        weights="weight",
        control=m.control,
    )
    other = fit_trial(data=d, response="yield", **kw)
    np.testing.assert_allclose(other.fitted_values + d.off, m.fitted_values, atol=1e-9)
    d["yield"] *= 10
    scaled = fit_trial(data=d, response="yield", **kw)
    np.testing.assert_allclose(
        scaled.fitted_values / 10, other.fitted_values, atol=2e-5
    )
    assert scaled.psi / 100 == pytest.approx(other.psi, rel=1e-5)


def test_nonconvergence_is_visible_and_consistent():
    d = pd.read_csv(ROOT / "field.csv")
    with pytest.warns(ConvergenceWarning):
        m = fit_trial(
            data=d,
            response="yield",
            genotype="geno",
            spatial=PSANOVA("col", "row", nseg=4),
            control=SpATSControl(max_iter=1),
        )
    assert not m.converged
    assert m.psi == 1 and all(v == 1 for v in m.var_comp.values())
    A = np.column_stack((m._X, m._Z))
    np.testing.assert_allclose(A @ m.coefficients, m.fitted_values)


@pytest.mark.parametrize(
    "kw,match",
    [
        ({"weights": -1}, "nonnegative"),
        ({"weights": [1, 2]}, "one value"),
        ({"offset": np.inf}, "finite"),
        ({"fixed": "~block"}, "list"),
        ({"fixed": ["col"]}, "rank deficient"),
    ],
)
def test_invalid_inputs(kw, match):
    d = pd.read_csv(ROOT / "field.csv")
    with pytest.raises((ValueError, TypeError), match=match):
        fit_trial(
            data=d,
            response="yield",
            genotype="geno",
            spatial=PSANOVA("col", "row", nseg=4),
            **kw,
        )


def test_adjustment_is_centered_and_not_genotype_prediction():
    m = fit_case("PSANOVA_random")
    assert abs(m.spatial_trend.mean()) < 1e-12
    np.testing.assert_allclose(m.adjusted_values + m.spatial_trend, m.data["yield"])
    assert len(m.genotype_predictions()) == 24
    assert m.genotype_predictions().se.min() > 0
    assert "adjusted" not in fit_case("poisson").to_frame()


def test_genotype_heritability_accounts_for_fixed_population():
    d = pd.read_csv(ROOT / "field.csv")
    m = fit_trial(
        data=d,
        response="yield",
        genotype="geno",
        genotype_as_random=True,
        spatial=PSANOVA("col", "row", nseg=4),
        fixed=["population"],
    )
    assert m._nominal["geno"] == 22
    assert m.heritability == pytest.approx(m.effective_dim["geno"] / 22)


def test_plot_uses_actual_fitted_surface():
    import matplotlib

    matplotlib.use("Agg")
    m = fit_case("PSANOVA_random")
    fig = m.plot(show=False)
    np.testing.assert_allclose(fig.axes[2].collections[0].get_array(), m.spatial_trend)


def test_variogram_order_invariance_and_exact_pairs():
    from pyspats.variogram import variogram, directional_variogram
    from scipy.spatial.distance import pdist

    m = fit_case("PSANOVA_random")
    v = variogram(m, max_dist=5, n_bins=10)
    ds = pdist(m.data[["col", "row"]])
    gs = pdist(m.residuals[:, None], metric="sqeuclidean") / 2
    edges = np.linspace(0, 5, 11)
    counts = np.histogram(ds, edges)[0]
    sums = np.histogram(ds, edges, weights=gs)[0]
    np.testing.assert_allclose(v.gamma, sums[counts > 0] / counts[counts > 0])
    a = directional_variogram(m, 0, max_dist=5)
    b = directional_variogram(m, 180, max_dist=5)
    np.testing.assert_array_equal(a.n_pairs, b.n_pairs)
    np.testing.assert_allclose(a.gamma, b.gamma)
    order = np.arange(len(m.data))[::-1]
    m.data = m.data.iloc[order]
    m.residuals = m.residuals[order]
    m.observed = m.observed[order]
    c = directional_variogram(m, 0, max_dist=5)
    np.testing.assert_allclose(a.gamma, c.gamma)
