"""Variance partitions agree with random-intercept variances and full model covariance."""

import copy

import numpy as np
import pandas as pd
import pytest

from pyspats import PSANOVA, ConvergenceWarning, SpATSControl, fit_trial, fit_trials

from .test_r_parity import ROOT, fit_case


def included(table):
    return table.loc[table.role != "fixed"]


def test_random_intercepts_reduce_to_lme4_style_variances():
    model = fit_case("PSANOVA_random")
    table = model.variance_partition().set_index("component")
    assert table.loc["geno", "variance"] == pytest.approx(model.var_comp["geno"])
    assert table.loc["block", "variance"] == pytest.approx(model.var_comp["block"])
    assert table.loc["residual", "variance"] == pytest.approx(model.psi)
    assert table.percent.sum() == pytest.approx(100)
    assert table.attrs["total_variance"] == pytest.approx(table.variance.sum())
    assert (table.environment == "environment_1").all()
    assert (table.trait == "yield").all()
    assert table.attrs["converged"]


@pytest.mark.parametrize(
    "case", ["PSANOVA_random", "SAP_random", "SAP.ANOVA_random", "weighted", "missing"]
)
def test_total_matches_independent_observation_covariance(case):
    model = fit_case(case)
    keep = model.observed[model.valid_obs]
    Z = model._Z[keep]
    tau = np.array(list(model.var_comp.values()))
    G = np.diag(1 / (model._penalties / tau[:, None]).sum(0))
    R = np.diag(model.psi / model.weights[model.observed])
    V = Z @ G @ Z.T + R
    expected = np.trace(V) / len(V)
    report = model.variance_partition()
    assert report.attrs["total_variance"] == pytest.approx(expected)
    assert report.attrs["n_obs"] == len(V)
    assert report.loc[
        report.component == "residual", "variance"
    ].item() == pytest.approx(np.trace(R) / len(R))
    assert included(report).variance.ge(0).all()


@pytest.mark.parametrize("kind,count", [("PSANOVA", 5), ("SAP", 3), ("SAP.ANOVA", 5)])
def test_spatial_components_are_additive_covariance_blocks(kind, count):
    model = fit_case(f"{kind}_random")
    total = model.variance_partition()
    details = model.variance_partition(spatial="components")
    spatial = details[
        details.component.str.startswith("spatial:") & (details.role != "fixed")
    ]
    assert len(spatial) == count  # SAP has two penalties, but three covariance blocks.
    assert spatial.variance.sum() == pytest.approx(
        total.loc[total.component == "spatial", "variance"].item()
    )
    assert spatial.percent.sum() == pytest.approx(
        total.loc[total.component == "spatial", "percent"].item()
    )
    assert included(details).percent.sum() == pytest.approx(100)


def test_unbalanced_population_variances_are_averaged_over_fitted_plots():
    model = fit_case("populations")
    report = model.variance_partition().set_index("component")
    for population in ("P1", "P2"):
        fraction = (model.data.loc[model.observed, "population"] == population).mean()
        name = f"geno:{population}"
        assert report.loc[name, "variance"] == pytest.approx(
            fraction * model.var_comp[name]
        )


def test_fixed_effects_are_explicitly_outside_variance_budget():
    model = fit_case("PSANOVA_fixed")
    report = model.variance_partition()
    fixed = report[report.role == "fixed"]
    assert set(fixed.component) == {"geno", "treatment", "spatial:polynomial"}
    assert fixed[["variance", "percent"]].isna().all().all()
    assert included(report).percent.sum() == pytest.approx(100)
    assert report.attrs["variance_budget"] == "random effects + residual"


def test_reporting_is_invariant_to_coefficient_units_and_does_not_mutate_fit():
    model = fit_case("PSANOVA_random")
    before = model.variance_partition(spatial="components")
    scaled = copy.deepcopy(model)
    # An equivalent random-coefficient parameterization Z*=Z*s, G*=G/s².
    scales = np.linspace(0.2, 5, model._Z.shape[1])
    scaled._Z *= scales
    scaled._penalties *= scales**2
    after = scaled.variance_partition(spatial="components")
    np.testing.assert_allclose(before.variance, after.variance)
    np.testing.assert_allclose(before.percent, after.percent)
    pd.testing.assert_frame_equal(
        before, model.variance_partition(spatial="components")
    )


def test_environment_keys_work_for_batch_output():
    data = pd.read_csv(ROOT / "field.csv")
    data = pd.concat(
        [data.assign(site="A", year=2025), data.assign(site="B", year=2026)]
    )
    jobs = list(
        fit_trials(
            data=data,
            by=["site", "year"],
            responses="yield",
            genotype="geno",
            genotype_as_random=True,
            spatial=PSANOVA("col", "row", nseg=4),
        )
    )
    frames = [job.model.variance_partition() for job in jobs]
    assert jobs[0].model.environment_id == ("A", 2025)
    assert jobs[1].model.environment_id == ("B", 2026)
    combined = pd.concat(frames, ignore_index=True)
    assert set(combined.environment) == {("A", 2025), ("B", 2026)}
    np.testing.assert_allclose(combined.groupby("environment").percent.sum(), 100)


def test_explicit_environment_label_is_metadata_only():
    data = pd.read_csv(ROOT / "field.csv")
    args = {
        "data": data,
        "response": "yield",
        "genotype": "geno",
        "spatial": PSANOVA("col", "row", nseg=4),
    }
    a = fit_trial(**args, environment_id="Lincoln_2025")
    b = fit_trial(**args, environment_id="Lincoln_2026")
    np.testing.assert_array_equal(a.fitted_values, b.fitted_values)
    assert (a.variance_partition().environment == "Lincoln_2025").all()
    with pytest.raises(TypeError, match="hashable"):
        fit_trial(**args, environment_id=[])


def test_nonconverged_fit_is_identified_in_report():
    data = pd.read_csv(ROOT / "field.csv")
    with pytest.warns(ConvergenceWarning):
        model = fit_trial(
            data=data,
            response="yield",
            genotype="geno",
            spatial=PSANOVA("col", "row", nseg=4),
            control=SpATSControl(max_iter=1),
        )
    assert model.variance_partition().attrs["converged"] is False


@pytest.mark.parametrize("family", ["poisson", "binomial"])
def test_non_gaussian_variance_budget_is_not_silently_mislabeled(family):
    with pytest.raises(NotImplementedError, match="Gaussian"):
        fit_case(family).variance_partition()


def test_unknown_partition_mode_is_rejected():
    with pytest.raises(ValueError, match="spatial"):
        fit_case("PSANOVA_random").variance_partition(spatial="penalty_parameters")


@pytest.mark.parametrize(
    "case", ["PSANOVA_random", "PSANOVA_fixed", "populations", "weighted", "missing"]
)
def test_adjusted_outputs_share_reference_and_preserve_residuals(case):
    model = fit_case(case)
    plots = model.adjusted_plots()
    means = model.genotype_means()
    np.testing.assert_allclose(
        plots.adjusted - plots.predicted_at_reference,
        plots.observed - plots.fitted,
        atol=1e-12,
    )
    expected = plots.genotype.map(means.set_index("genotype").predicted)
    np.testing.assert_allclose(
        plots.loc[model.valid_obs, "predicted_at_reference"], expected[model.valid_obs]
    )
    assert plots.index.equals(model.data.index)
    assert means.n_obs.sum() == model.n_obs
    assert plots.attrs["reference"] == means.attrs["reference"]
    assert plots.loc[plots.observed.isna(), "adjusted"].isna().all()


def test_balanced_means_match_explicit_design_averages_and_covariance():
    model = fit_case("PSANOVA_random")
    means = model.genotype_means()
    base = model.data.loc[model.observed].copy()
    for row in means.itertuples():
        contrasts = []
        for treatment in model._encoding["treatment"]:
            data = base.copy()
            data[model.genotype] = row.genotype
            data["treatment"] = treatment
            x, z = model._design(data)
            for name in model.random:
                z[:, model._random_slices[name]] = 0
            contrasts.append(np.concatenate([x.mean(0), z.mean(0)]))
        contrast = np.mean(contrasts, axis=0)
        assert row.predicted == pytest.approx(contrast @ model.coefficients)
        assert row.se**2 == pytest.approx(contrast @ model.covariance @ contrast)


def test_reference_overrides_and_offsets_move_both_outputs_together():
    model = fit_case("PSANOVA_random")
    levels = model._encoding["treatment"]
    a = model.genotype_means(reference={"treatment": levels[0]}, reference_offset=3)
    b = model.genotype_means(reference={"treatment": levels[1]}, reference_offset=3)
    balanced = model.genotype_means(reference_offset=3)
    np.testing.assert_allclose((a.predicted + b.predicted) / 2, balanced.predicted)
    np.testing.assert_allclose(balanced.predicted - model.genotype_means().predicted, 3)
    plots = model.adjusted_plots(reference={"treatment": levels[0]}, reference_offset=3)
    np.testing.assert_allclose(
        plots.predicted_at_reference,
        plots.genotype.map(a.set_index("genotype").predicted),
    )
    for reference in [{"unknown": 1}, {"treatment": "unseen"}]:
        with pytest.raises(ValueError):
            model.genotype_means(reference=reference)
    with pytest.raises(ValueError):
        model.adjusted_plots(reference_offset=np.nan)


def test_numeric_reference_override_and_duplicate_plot_indices():
    data = pd.read_csv(ROOT / "field.csv")
    data["covariate"] = np.random.default_rng(19).normal(size=len(data))
    data.index = np.zeros(len(data), dtype=int)
    model = fit_trial(
        data=data,
        response="yield",
        genotype="geno",
        spatial=PSANOVA("col", "row", nseg=4),
        fixed=["covariate"],
    )
    base = model.genotype_means()
    shifted = model.genotype_means(reference={"covariate": data.covariate.mean() + 2})
    beta = model.coefficients[model._fixed_names.index("covariate")]
    np.testing.assert_allclose(shifted.predicted - base.predicted, 2 * beta)
    plots = model.adjusted_plots()
    assert plots.index.equals(data.index)
    assert plots.plot_id.is_unique
    with pytest.raises(ValueError):
        model.genotype_means(reference={"covariate": np.inf})


@pytest.mark.parametrize("family", ["poisson", "binomial"])
def test_adjustments_reject_non_gaussian_models(family):
    model = fit_case(family)
    with pytest.raises(NotImplementedError):
        model.genotype_means()
    with pytest.raises(NotImplementedError):
        model.adjusted_plots()
