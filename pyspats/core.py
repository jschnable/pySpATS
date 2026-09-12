"""Field-trial spatial mixed models with the R SpATS statistical specification."""

import warnings
import numpy as np
import pandas as pd
from .control import SpATSControl
from .families import Family, gaussian
from .model_basis import SpatialBasis, SpatialSpec, spatial_spec
from .engine import fit_mixed, ConvergenceWarning


class SpATS:
    """Fit a spatial mixed model to one field and one response.

    Parameters
    ----------
    response, genotype : str
        Data columns. Genotype IDs are categorical even when stored as numbers.
    spatial : tuple[str, str] or SpatialSpec
        ``('col', 'row')`` uses PSANOVA with 10 segments per axis. Use
        ``PSANOVA('col', 'row', nseg=(20, 30), nest_div=2)`` to configure it.
    data : pandas.DataFrame
        One row per plot. Input is copied. Missing responses are excluded from
        estimation but predicted where predictors are complete. Missing
        predictors yield NaN predictions. Original row order/index are retained.
    genotype_as_random : bool
        False estimates genotype effects (BLUEs); True predicts effects with
        shrinkage (BLUPs) and permits generalized heritability estimation.
    fixed, random : sequence[str], optional
        Additive columns. Numeric fixed columns are covariates; other fixed
        columns and all random columns are categorical. Formula strings are
        intentionally unsupported. Cast numeric treatment codes to category.
    geno_decomp : str, optional
        Genotype population column: estimates separate genotype variances.
        Each genotype must belong to exactly one population.
    weights : array-like or column name, optional
        Nonnegative precision weights. Gaussian residual variance is psi/weight.
        Zero weights exclude a plot from fitting; they do not mean a zero trait.
    offset : scalar, array-like or column name, optional
        Known contribution on the link scale, subtracted before fitting.
    family : Family, optional
        Gaussian by default. Poisson/log and binomial/logit use R-style working
        mixed models (approximate inference, not exact GLMM likelihood).
    control : SpATSControl, optional
        Iteration tolerance, limit and monitoring.

    Notes
    -----
    Construction fits immediately for compatibility. ``fit_trial`` is a
    keyword-only function with the same result. Covariance and prediction SEs
    condition on fitted variance parameters; they omit smoothing uncertainty.
    """

    def __init__(
        self,
        response: str,
        genotype: str,
        spatial: SpatialSpec | tuple[str, str] | dict,
        data: pd.DataFrame,
        genotype_as_random: bool = False,
        geno_decomp: str | None = None,
        fixed: list[str] | None = None,
        random: list[str] | None = None,
        family: Family | None = None,
        offset=None,
        weights=None,
        control: SpATSControl | None = None,
    ):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame")
        self.response, self.genotype, self.spatial = response, genotype, spatial
        self.spec = spatial_spec(spatial)
        self.genotype_as_random = bool(genotype_as_random)
        self.geno_decomp = geno_decomp
        self.fixed, self.random = self._terms(fixed), self._terms(random)
        self.family = gaussian() if family is None else family
        if self.family.family not in ("gaussian", "poisson", "binomial"):
            raise NotImplementedError(
                "Supported families are Gaussian, Poisson and binomial"
            )
        if geno_decomp and not genotype_as_random:
            raise ValueError("geno_decomp requires genotype_as_random=True")
        self.control = control or SpATSControl()
        self.data = data.copy(deep=True)
        required = list(
            dict.fromkeys(
                [response, genotype, self.spec.x, self.spec.y]
                + self.fixed
                + self.random
                + ([geno_decomp] if geno_decomp else [])
            )
        )
        missing = [c for c in required if c not in data]
        if missing:
            raise ValueError(f"Missing columns in data: {missing}")
        if not data.columns.is_unique:
            raise ValueError("Data column names must be unique")
        self.weights = self._vector(weights, 1, "weights", data)
        self.offset = self._vector(offset, 0, "offset", data)
        if np.any(self.weights < 0):
            raise ValueError("weights must be nonnegative")
        predictors = [c for c in required if c != response]
        valid = ~data[predictors].isna().any(axis=1).to_numpy()
        for c in [self.spec.x, self.spec.y] + [
            c for c in self.fixed if pd.api.types.is_numeric_dtype(data[c])
        ]:
            vals = pd.to_numeric(data[c], errors="raise").to_numpy(
                dtype=float, na_value=np.nan
            )
            if np.isinf(vals).any():
                raise ValueError(f"Column {c!r} contains infinite values")
        self.valid_obs = valid
        y_all = pd.to_numeric(data[response], errors="raise").to_numpy(
            dtype=float, na_value=np.nan
        )
        if np.isinf(y_all).any():
            raise ValueError("Response contains infinite values")
        observed = valid & np.isfinite(y_all) & (self.weights > 0)
        self.observed = observed
        self.n_obs = int(observed.sum())
        if not self.n_obs:
            raise ValueError(
                "No observations have complete predictors, response and positive weight"
            )
        y = y_all[observed]
        if self.family.family == "poisson" and (
            np.any(y < 0) or np.any(y != np.floor(y))
        ):
            raise ValueError("Poisson responses must be nonnegative integer counts")
        if self.family.family == "binomial" and np.any((y < 0) | (y > 1)):
            raise ValueError(
                "Binomial responses must be in [0, 1]; use weights for trial counts"
            )
        complete = data.loc[valid]
        self._basis = SpatialBasis(
            complete[self.spec.x].to_numpy(float),
            complete[self.spec.y].to_numpy(float),
            self.spec,
            observed[valid],
        )
        self._encoding = {}
        for c in list(dict.fromkeys([genotype] + self.fixed + self.random)):
            categorical = (
                c == genotype
                or c in self.random
                or not pd.api.types.is_numeric_dtype(data[c])
            )
            if categorical:
                levels = list(pd.unique(data.loc[observed, c]))
                try:
                    levels.sort()
                except TypeError:
                    pass
                unseen = set(complete[c]) - set(levels)
                if unseen:
                    warnings.warn(
                        f"{c!r} has levels without observed responses; their plots receive NaN predictions: {unseen}",
                        UserWarning,
                        stacklevel=2,
                    )
                    valid &= data[c].isin(levels).to_numpy()
                self._encoding[c] = levels
        self.valid_obs = valid
        complete = data.loc[valid]
        self._genotype_levels = self._encoding[genotype]
        self._populations = None
        if geno_decomp:
            pairs = complete[[genotype, geno_decomp]].drop_duplicates()
            if pairs[genotype].duplicated().any():
                raise ValueError(
                    "Each genotype must belong to one geno_decomp population"
                )
            self._populations = dict(zip(pairs[genotype], pairs[geno_decomp]))
        X, Z = self._design(complete, training=True)
        local_obs = observed[valid]
        Xfit, Zfit = X[local_obs], Z[local_obs]
        w = self.weights[observed]
        off = self.offset[observed]
        self._nominal = {}
        p = Xfit.shape[1]
        for name, sl in self._random_slices.items():
            if not self.genotype_as_random or not (
                name == self.genotype
                or (self.geno_decomp and name.startswith(self.genotype + ":"))
            ):
                continue
            block = Zfit[:, sl]
            counts = block.sum(0)
            residualized = Xfit - block @ ((block.T @ Xfit) / counts[:, None])
            tol = (
                max(Xfit.shape)
                * np.finfo(float).eps
                * max(np.linalg.norm(Xfit, 2), 1)
                * 10
            )
            self._nominal[name] = (
                block.shape[1] + np.linalg.matrix_rank(residualized, tol=tol) - p
            )
        self._X, self._Z = X, Z
        initial = None
        self.outer_history = []
        if self.family.family == "gaussian":
            result = fit_mixed(
                Xfit,
                Zfit,
                y - off,
                w,
                self._penalties,
                self.control,
                update_dispersion=self.control.update_psi_gauss
                or self.control.update_psi,
                genotype_indices=self._genotype_indices,
            )
            self.outer_converged = True
        else:
            if self.family.family == "poisson":
                mu = y + 0.1
                eta = np.log(mu)
            else:
                mu = (w * y + 0.5) / (w + 1)
                eta = np.log(mu / (1 - mu))
            self.outer_converged = False
            for outer in range(self.control.max_iter):
                derivative = np.maximum(self.family.d_inverse_link(eta), 1e-12)
                variance = np.maximum(self.family.variance(mu), 1e-12)
                working = eta - off + (y - mu) / derivative
                working_w = w * derivative**2 / variance
                result = fit_mixed(
                    Xfit,
                    Zfit,
                    working,
                    working_w,
                    self._penalties,
                    self.control,
                    initial=initial,
                    update_dispersion=self.control.update_psi,
                    genotype_indices=self._genotype_indices,
                )
                neweta = (
                    Xfit @ result.coefficients[:p]
                    + Zfit @ result.coefficients[p:]
                    + off
                )
                change = np.sum((neweta - eta) ** 2) / max(np.sum(neweta**2), 1e-20)
                self.outer_history.append(float(change))
                eta = neweta
                mu = self.family.inverse_link(eta)
                initial = result.dispersion, result.variances
                if change < self.control.tolerance:
                    self.outer_converged = True
                    break
        self._result = result
        self.coefficients = result.coefficients
        self.covariance = result.covariance
        self.psi = result.dispersion
        self.var_comp = dict(zip(self._variance_names, result.variances))
        self.effective_dim = {
            "fixed": float(p),
            **dict(zip(self._variance_names, result.effective_dimensions)),
        }
        self.deviance = result.objective
        self.n_iterations = len(result.history)
        self.converged = result.converged and self.outer_converged
        self.history = pd.DataFrame(result.history)
        if not self.converged:
            warnings.warn(
                "SpATS did not converge; inspect history or increase max_iter",
                ConvergenceWarning,
                stacklevel=2,
            )
        eta = X @ self.coefficients[:p] + Z @ self.coefficients[p:] + self.offset[valid]
        self.fitted_values = np.full(len(data), np.nan)
        self.fitted_values[valid] = self.family.inverse_link(eta)
        self.residuals = y_all - self.fitted_values
        self.spatial_trend = np.full(len(data), np.nan)
        self.spatial_trend[valid] = (
            X[:, self._spatial_fixed] @ self.coefficients[self._spatial_fixed]
            + Z[:, : self._spatial_q] @ self.coefficients[p : p + self._spatial_q]
        )
        # Center for an interpretable correction, leaving an overall trait level.
        self.spatial_trend -= np.average(self.spatial_trend[observed], weights=w)
        self.adjusted_values = (
            y_all - self.spatial_trend if self.family.family == "gaussian" else None
        )
        self.residual_df = self.n_obs - sum(self.effective_dim.values())

    @staticmethod
    def _terms(terms):
        if terms is None:
            return []
        if isinstance(terms, str):
            raise TypeError(
                "fixed/random must be a list of column names, not a formula string"
            )
        terms = list(terms)
        if len(set(terms)) != len(terms):
            raise ValueError("Duplicate model terms")
        return terms

    @staticmethod
    def _vector(value, default, name, data):
        if isinstance(value, str):
            value = data[value]
        a = np.asarray(default if value is None else value, dtype=float)
        if a.ndim == 0:
            a = np.full(len(data), a)
        if a.shape != (len(data),) or not np.isfinite(a).all():
            raise ValueError(
                f"{name} must be finite and have one value per input row (or be scalar)"
            )
        return a.copy()

    def _encode(self, data, name, drop=False):
        if name not in self._encoding:
            return data[name].to_numpy(dtype=float).reshape(-1, 1)
        levels = self._encoding[name]
        if not data[name].isin(levels).all():
            raise ValueError(
                f"Missing or unseen levels in {name!r}; predictions require fitted levels"
            )
        codes = pd.Index(levels).get_indexer(data[name].astype(object))
        if np.any(codes < 0):
            raise ValueError(
                f"Missing or unseen levels in {name!r}; predictions require fitted levels"
            )
        out = np.zeros((len(data), len(levels)))
        out[np.arange(len(data)), codes] = 1
        return out[:, 1:] if drop else out

    def _design(self, data, training=False):
        sx, sz = self._basis.evaluate(
            data[self.spec.x].to_numpy(float), data[self.spec.y].to_numpy(float)
        )
        # Full genotype indicators when fixed match R's identifiable parameterization.
        if self.genotype_as_random:
            parts = [np.ones((len(data), 1))]
            names = ["Intercept"]
        else:
            parts = [self._encode(data, self.genotype)]
            names = [f"{self.genotype}[{g}]" for g in self._genotype_levels]
        for c in self.fixed:
            part = self._encode(data, c, drop=True)
            parts.append(part)
            names += (
                [f"{c}[{v}]" for v in self._encoding[c][1:]]
                if c in self._encoding
                else [c]
            )
        start = sum(v.shape[1] for v in parts)
        parts.append(sx)
        names += [f"spatial_polynomial_{i + 1}" for i in range(sx.shape[1])]
        X = np.column_stack(parts)
        zs = [sz]
        random_slices = {}
        k = sz.shape[1]
        extra = []
        if self.genotype_as_random:
            geno = self._encode(data, self.genotype)
            if self._populations:
                populations = list(
                    dict.fromkeys(self._populations[g] for g in self._genotype_levels)
                )
                for pop in populations:
                    cols = [
                        i
                        for i, g in enumerate(self._genotype_levels)
                        if self._populations[g] == pop
                    ]
                    extra.append((f"{self.genotype}:{pop}", geno[:, cols]))
            else:
                extra.append((self.genotype, geno))
        extra += [(c, self._encode(data, c)) for c in self.random]
        if len({name for name, _ in extra}) != len(extra):
            raise ValueError("Random terms duplicate genotype or population terms")
        for name, block in extra:
            zs.append(block)
            random_slices[name] = slice(k, k + block.shape[1])
            k += block.shape[1]
        Z = np.column_stack(zs)
        if training:
            if not self.genotype_as_random:
                self._genotype_indices = np.arange(len(self._genotype_levels))
            else:
                self._genotype_indices = (
                    X.shape[1] + sz.shape[1] + np.arange(len(self._genotype_levels))
                )
            self._spatial_fixed = slice(start, X.shape[1])
            self._spatial_q = sz.shape[1]
            self._random_slices = random_slices
            self._fixed_names = names
            ns = len(self._basis.names)
            self._variance_names = self._basis.names + list(random_slices)
            if len(set(self._variance_names)) != len(self._variance_names):
                raise ValueError(
                    "Model term names collide with reserved spatial component names"
                )
            self._penalties = np.zeros((ns + len(random_slices), k))
            self._penalties[:ns, : sz.shape[1]] = self._basis.penalties
            for i, sl in enumerate(random_slices.values(), ns):
                self._penalties[i, sl] = 1
        return X, Z

    def predict(
        self,
        newdata: pd.DataFrame | None = None,
        *,
        offset=None,
        return_se: bool = False,
    ) -> np.ndarray | pd.DataFrame:
        """Predict plot means using training knots, contrasts and level ordering.

        New-data offsets default to zero; supply them explicitly if applicable.
        Unseen factor levels and coordinates outside the field raise errors.
        SE describes the latent predictor (link scale), not a future observation.
        """
        if newdata is None:
            if not return_se:
                return self.fitted_values.copy()
            A = np.column_stack((self._X, self._Z))
            se = np.full(len(self.data), np.nan)
            se[self.valid_obs] = np.sqrt(
                np.maximum(np.einsum("ij,jk,ik->i", A, self.covariance, A), 0)
            )
            return pd.DataFrame(
                {"predicted": self.fitted_values, "se_link": se}, index=self.data.index
            )
        X, Z = self._design(newdata)
        A = np.column_stack((X, Z))
        off = self._vector(offset, 0, "offset", newdata)
        pred = self.family.inverse_link(A @ self.coefficients + off)
        if return_se:
            se = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", A, self.covariance, A), 0))
            return pd.DataFrame({"predicted": pred, "se_link": se}, index=newdata.index)
        return pred

    def genotype_predictions(self) -> pd.DataFrame:
        """Genotype effects and conditional SEs on the link scale.

        Fixed genotypes: coefficient at zero values of the other model columns.
        Random genotypes: zero-centered BLUP, excluding the intercept. These
        are effects, not marginal means averaged over treatments or locations.
        """
        if not self.genotype_as_random:
            indices = np.arange(len(self._genotype_levels))
            levels = self._genotype_levels
        else:
            indices, levels = [], []
            p = self._X.shape[1]
            if self._populations:
                for name, sl in self._random_slices.items():
                    if not name.startswith(self.genotype + ":"):
                        continue
                    pop_levels = [
                        g
                        for g in self._genotype_levels
                        if f"{self.genotype}:{self._populations[g]}" == name
                    ]
                    levels.extend(pop_levels)
                    indices.extend(range(p + sl.start, p + sl.stop))
            else:
                sl = self._random_slices[self.genotype]
                indices = np.arange(p + sl.start, p + sl.stop)
                levels = self._genotype_levels
        indices = np.asarray(indices, dtype=int)
        return pd.DataFrame(
            {
                self.genotype: levels,
                "effect": self.coefficients[indices],
                "se": np.sqrt(np.maximum(np.diag(self.covariance)[indices], 0)),
                "type": "BLUP" if self.genotype_as_random else "BLUE",
            }
        )

    def get_heritability(self) -> float | dict[str, float]:
        """Generalized H² = genotype effective dimension / estimable dimension.

        Requires random genotypes. Estimable dimension is rank([X,Zg])-rank(X),
        as in R, accounting for confounding with all fixed effects. This is
        trial-specific generalized heritability, not a universal genetic trait.
        """
        if not self.genotype_as_random:
            raise ValueError("Heritability requires genotype_as_random=True")
        names = [
            n
            for n in self._random_slices
            if n == self.genotype
            or (self.geno_decomp and n.startswith(self.genotype + ":"))
        ]
        values = {}
        for name in names:
            nominal = self._nominal[name]
            if nominal == 0:
                raise ValueError(f"Heritability is not identifiable for {name}")
            values[name] = self.effective_dim[name] / nominal
        return values if self.geno_decomp else values[self.genotype]

    @property
    def heritability(self):
        return self.get_heritability()

    def to_frame(self) -> pd.DataFrame:
        """Original plots plus fitted, residual, spatial trend, adjusted and used columns.

        ``adjusted`` removes only the centered spatial trend; it retains
        treatment, block, genotype and residual contributions.
        """
        result = self.data.assign(
            fitted=self.fitted_values,
            residual=self.residuals,
            spatial_trend=self.spatial_trend,
            used_for_fit=self.observed,
        )
        if self.adjusted_values is not None:
            result["adjusted"] = self.adjusted_values
        return result

    def summary(self, which: str = "all") -> pd.DataFrame:
        """Return a table suitable for logs, notebooks and CSV export."""
        table = pd.DataFrame(
            {
                "variance": self.var_comp,
                "effective_dimension": {
                    k: v for k, v in self.effective_dim.items() if k != "fixed"
                },
            }
        )
        table.attrs.update(
            converged=self.converged,
            n_obs=self.n_obs,
            dispersion=self.psi,
            iterations=self.n_iterations,
            residual_df=self.residual_df,
        )
        return table

    def summary_ed(self):
        return pd.Series(self.effective_dim, name="effective_dimension")

    def plot(self, show=True, figsize=(12, 8), **kwargs):
        """Plot observed, fitted, spatial trend and residual values at plot coordinates."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        for ax, values, label in zip(
            axes.flat,
            [
                self.data[self.response],
                self.fitted_values,
                self.spatial_trend,
                self.residuals,
            ],
            ["Observed", "Fitted", "Spatial trend", "Residual"],
        ):
            scatter = ax.scatter(
                self.data[self.spec.x], self.data[self.spec.y], c=values, marker="s"
            )
            ax.set(xlabel=self.spec.x, ylabel=self.spec.y, title=label)
            fig.colorbar(scatter, ax=ax)
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    def __repr__(self):
        return (
            f"SpATS(response={self.response!r}, genotype={self.genotype!r}, "
            f"n_obs={self.n_obs}, converged={self.converged}, iterations={self.n_iterations})"
        )


def fit_trial(
    *,
    data: pd.DataFrame,
    response: str,
    genotype: str,
    spatial: SpatialSpec | tuple[str, str] | dict,
    **kwargs,
) -> SpATS:
    """Fit one trial; keyword-only entry point returning a :class:`SpATS` result."""
    return SpATS(
        response=response, genotype=genotype, spatial=spatial, data=data, **kwargs
    )
