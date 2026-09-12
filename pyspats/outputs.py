"""Standard output tables, with environment and trait keys on every row.

Reporting is separate from model estimation. A variance partition describes
random-effect and residual variance assigned by the fitted Gaussian model.
"""

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .core import SpATS


def variance_partition(
    model: "SpATS", *, spatial: Literal["total", "components"] = "total"
) -> pd.DataFrame:
    """Return model-assigned variance on the phenotype scale.

    For a random coefficient block with design Z and fitted prior covariance G,
    contribution = trace(Z G Z') / n. Ordinary random intercept factors reduce
    to their fitted variance parameter. Population-specific genotype components
    are averaged over the entire fitted environment, including zero incidence
    outside their populations. This is not the variance of fitted BLUPs or the
    coefficient prediction-error covariance.

    The reference population is the observed, positive-weight fitting plots,
    each counted equally. Precision weights affect residual variance (psi/w),
    not the relative representation of plots in this report. Spatial reporting
    uses the fitted basis, including the explicit fit-time centering setting.

    Fixed terms are listed with role='fixed' and missing variance/percent;
    they have no estimated variance parameter in this model. Included rows sum
    to 100 percent. There is no duplicated total row.

    Parameters
    ----------
    spatial : {'total', 'components'}
        Combine all penalized spatial effects by default, or show the disjoint
        spline coefficient blocks. SAP penalty parameters overlap and are not
        separate variance contributions; components reports covariance blocks.

    Returns
    -------
    pandas.DataFrame
        environment, trait, component, role, variance and percent. Attributes
        record the denominator, number of plots, reporting method, spatial
        parameterization and fit convergence. Variance has squared trait units.
        For fixed effects the missing entries are deliberate, not zero variance.
    """
    if spatial not in ("total", "components"):
        raise ValueError("spatial must be 'total' or 'components'")
    if model.family.family != "gaussian":
        raise NotImplementedError(
            "Phenotype-scale variance partitioning currently requires a Gaussian model"
        )

    tau = np.asarray([model.var_comp[name] for name in model._variance_names])
    precision = (model._penalties / tau[:, None]).sum(axis=0)
    # The fitted coefficient prior is diagonal, including overlapping SAP
    # penalties. Avoid constructing G or an n-by-n observation covariance.
    coefficient_variance = 1.0 / precision
    observed = model.observed[model.valid_obs]
    mean_square = np.zeros(model._Z.shape[1])
    for start in range(0, len(model._Z), 512):
        stop = start + 512
        block = model._Z[start:stop][observed[start:stop]]
        mean_square += np.einsum("ij,ij->j", block, block)
    mean_square /= model.n_obs
    contribution = coefficient_variance * mean_square

    records = []

    def add(component, role, value):
        records.append(
            {
                "environment": model.environment_id,
                "trait": model.response,
                "component": component,
                "role": role,
                "variance": float(value),
            }
        )

    for name, block in model._random_slices.items():
        add(name, "random", contribution[block].sum())

    if spatial == "total":
        add("spatial", "random", contribution[: model._spatial_q].sum())
    else:
        if model.spec.kind == "SAP":
            names = ["x_marginal_block", "y_marginal_block", "interaction_block"]
        else:
            names = [
                "x_smooth",
                "y_smooth",
                "x_smooth:y_polynomial",
                "x_polynomial:y_smooth",
                "x_smooth:y_smooth",
            ]
            if model.spec.penalty_order == (2, 2):
                names[2:4] = ["x_smooth:y_linear", "x_linear:y_smooth"]
        for name, block in zip(names, model._basis.blocks):
            if block.stop > block.start:
                add(f"spatial:{name}", "random", contribution[block].sum())

    add("residual", "residual", np.mean(model.psi / model.weights[model.observed]))
    total = sum(row["variance"] for row in records)
    if not np.isfinite(total) or total <= 0:
        raise ValueError(
            "Total random-effect and residual variance must be finite and positive"
        )

    if not model.genotype_as_random:
        add(model.genotype, "fixed", np.nan)
    for name in model.fixed:
        add(name, "fixed", np.nan)
    if model._spatial_fixed.stop > model._spatial_fixed.start:
        add("spatial:polynomial", "fixed", np.nan)

    table = pd.DataFrame(records)
    table["percent"] = 100.0 * table["variance"] / total
    table.attrs.update(
        method="mean_marginal_variance",
        variance_budget="random effects + residual",
        total_variance=total,
        n_obs=model.n_obs,
        converged=model.converged,
        population="observed positive-weight plots, equally represented",
        spatial_center=model.spec.center,
        spatial_model=model.spec.kind,
    )
    return table


def _reference(model, reference, reference_offset):
    """One common coefficient contrast; never expand genotype x plot grids."""
    if model.family.family != "gaussian":
        raise NotImplementedError("Adjusted outputs currently require a Gaussian model")
    reference = {} if reference is None else dict(reference)
    unknown = set(reference) - set(model.fixed)
    if unknown:
        raise ValueError(
            f"Reference overrides must name fitted fixed covariates: {unknown}"
        )
    p = model._X.shape[1]
    contrast = np.zeros(len(model.coefficients))
    cursor = 1 if model.genotype_as_random else len(model._genotype_levels)
    if model.genotype_as_random:
        contrast[0] = 1
    conditions = {}
    for name in model.fixed:
        if name in model._encoding:
            levels = model._encoding[name]
            weights = np.full(len(levels), 1 / len(levels))
            if name in reference:
                matches = [value == reference[name] for value in levels]
                if not any(matches):
                    raise ValueError(f"Reference for {name!r} must be a fitted level")
                weights = np.asarray(matches, dtype=float)
            contrast[cursor : cursor + len(levels) - 1] = weights[1:]
            cursor += len(levels) - 1
            conditions[name] = dict(zip(levels, weights.tolist()))
        else:
            value = float(
                reference.get(name, model.data.loc[model.observed, name].mean())
            )
            if not np.isfinite(value):
                raise ValueError(f"Reference for {name!r} must be finite")
            contrast[cursor] = value
            cursor += 1
            conditions[name] = value
    observed = model.observed[model.valid_obs]
    contrast[model._spatial_fixed] = model._X[observed, model._spatial_fixed].mean(
        axis=0
    )
    contrast[p : p + model._spatial_q] = model._Z[observed, : model._spatial_q].mean(
        axis=0
    )
    offset = float(
        np.mean(model.offset[model.observed])
        if reference_offset is None
        else reference_offset
    )
    if not np.isfinite(offset):
        raise ValueError("reference_offset must be finite")
    metadata = {
        "fixed": conditions,
        "offset": offset,
        "spatial": "equal average over observed positive-weight plots",
        "other_random_effects": 0.0,
    }
    return contrast, offset, metadata


def _genotype_indices(model):
    if not model.genotype_as_random:
        return np.arange(len(model._genotype_levels))
    p = model._X.shape[1]
    if not model._populations:
        sl = model._random_slices[model.genotype]
        return np.arange(p + sl.start, p + sl.stop)
    indices = {}
    for pop in dict.fromkeys(model._populations.values()):
        sl = model._random_slices[f"{model.genotype}:{pop}"]
        levels = [g for g in model._genotype_levels if model._populations[g] == pop]
        indices.update(zip(levels, range(p + sl.start, p + sl.stop)))
    return np.array([indices[g] for g in model._genotype_levels])


def genotype_means(model, *, reference=None, reference_offset=None):
    """Genotype phenotype predictions at a shared balanced reference.

    Fixed categorical covariates receive equal level weights; numeric covariates
    use means over observed positive-weight plots. ``reference`` maps fixed
    covariate names to a specific level or numeric value. Spatial effects are
    averaged equally over those plots and other random effects set to zero.
    Known offsets default to their mean; ``reference_offset`` overrides it.

    SEs include covariance with the reference, conditional on fitted variance
    parameters; for random genotypes these are prediction-error SEs. They are
    not future-observation intervals. Reference definitions and convergence are
    stored in DataFrame.attrs. Gaussian models only.
    """
    contrast, offset, metadata = _reference(model, reference, reference_offset)
    indices = _genotype_indices(model)
    covariance_reference = model.covariance @ contrast
    variance = (
        contrast @ covariance_reference
        + 2 * covariance_reference[indices]
        + np.diag(model.covariance)[indices]
    )
    counts = model.data.loc[model.observed, model.genotype].value_counts()
    table = pd.DataFrame(
        {
            "environment": [model.environment_id] * len(indices),
            "trait": model.response,
            "genotype": model._genotype_levels,
            "predicted": contrast @ model.coefficients
            + offset
            + model.coefficients[indices],
            "se": np.sqrt(np.maximum(variance, 0)),
            "n_obs": [int(counts[g]) for g in model._genotype_levels],
            "type": "BLUP" if model.genotype_as_random else "BLUE",
        }
    )
    table.attrs.update(
        reference=metadata, converged=model.converged, genotype_column=model.genotype
    )
    return table


def adjusted_plots(model, *, reference=None, reference_offset=None):
    """Plot phenotypes at the same reference used by genotype_means().

    ``adjusted = predicted_at_reference + observed - fitted`` preserves plot
    residuals. Missing phenotypes or invalid predictors give missing adjustments.
    Predictions are separate and may exist without an observed phenotype.
    All input rows and their index are preserved; plot_id is original row position.
    Identifiers use standardized column names; original data remain in model.data.
    No SE is assigned to noisy adjusted observations. Gaussian models only.
    """
    means = genotype_means(
        model, reference=reference, reference_offset=reference_offset
    )
    lookup = means.set_index("genotype").predicted
    predicted = model.data[model.genotype].map(lookup).to_numpy(dtype=float, copy=True)
    predicted[~model.valid_obs] = np.nan
    observed = model.data[model.response].to_numpy(dtype=float, na_value=np.nan)
    residual = observed - model.fitted_values
    table = pd.DataFrame(
        {
            "environment": [model.environment_id] * len(model.data),
            "trait": model.response,
            "plot_id": np.arange(len(model.data)),
            "genotype": model.data[model.genotype].to_numpy(),
            "observed": observed,
            "fitted": model.fitted_values,
            "residual": residual,
            "predicted_at_reference": predicted,
            "adjusted": predicted + residual,
            "used_for_fit": model.observed,
        },
        index=model.data.index,
    )
    table.attrs.update(means.attrs)
    return table
