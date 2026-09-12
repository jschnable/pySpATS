"""Exact SAP updates, eliminating the diagonal genotype block by a Schur solve.

Uses coefficient-space sufficient statistics, never an observation covariance
or hat matrix. All estimates refer to the same variance iterate. Equations
follow SpATS 1.0-20; diagonal scaling and trace rearrangement improve numerical
stability without changing the statistical model.
"""

from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_factor, cho_solve


class ConvergenceWarning(UserWarning):
    """The iteration limit was reached; inspect the result before using it."""


@dataclass
class EngineResult:
    coefficients: np.ndarray
    covariance: np.ndarray
    variances: np.ndarray
    dispersion: float
    effective_dimensions: np.ndarray
    objective: float
    history: list
    converged: bool


def inverse_spd(matrix):
    scale = 1 / np.sqrt(np.diag(matrix))
    factor = cho_factor(
        scale[:, None] * matrix * scale[None, :], lower=True, check_finite=False
    )
    inverse = (
        scale[:, None]
        * cho_solve(factor, np.eye(len(matrix)), check_finite=False)
        * scale[None, :]
    )
    logdet = 2 * (np.log(np.diag(factor[0])).sum() - np.log(scale).sum())
    return inverse, logdet


def fit_mixed(
    X,
    Z,
    y,
    weights,
    penalties,
    control,
    initial=None,
    update_dispersion=True,
    genotype_indices=None,
):
    p, q = X.shape[1], Z.shape[1]
    A = np.column_stack((X, Z))
    n = np.count_nonzero(weights)
    if n <= p:
        raise ValueError(
            "No residual degrees of freedom; more observations or fewer fixed effects are required"
        )
    gi = np.asarray([] if genotype_indices is None else genotype_indices, dtype=int)
    ri = np.setdiff1d(np.arange(p + q), gi)
    # Genotype indicators have disjoint support: cross-products by group sums.
    if len(gi):
        codes = A[:, gi].argmax(axis=1)
        if not np.all(A[np.arange(len(A)), gi[codes]] == 1):
            raise ValueError("Genotype design must be one-hot")
        R = A[:, ri]
        diagonal = np.bincount(codes, weights=weights, minlength=len(gi))
        cross = np.zeros((len(gi), len(ri)))
        np.add.at(cross, codes, weights[:, None] * R)
        rr = R.T @ (weights[:, None] * R)
        gy = np.bincount(codes, weights=weights * y, minlength=len(gi))
        ry = R.T @ (weights * y)
        # Validate fixed rank after eliminating full fixed genotype indicators.
        if np.all(gi < p):
            fixed_rest = ri < p
            fixed_gram = (
                rr[np.ix_(fixed_rest, fixed_rest)]
                - (cross[:, fixed_rest].T / diagonal) @ cross[:, fixed_rest]
            )
        else:
            fixed_gram = X.T @ (weights[:, None] * X)
    else:
        rr = A.T @ (weights[:, None] * A)
        ry = A.T @ (weights * y)
        fixed_gram = rr[:p, :p]
    if len(fixed_gram):
        fixed_scale = np.sqrt(np.maximum(np.diag(fixed_gram), np.finfo(float).tiny))
        normalized = fixed_gram / fixed_scale[:, None] / fixed_scale[None, :]
        if np.linalg.eigvalsh(normalized).min() < 1e-10:
            raise ValueError(
                "Fixed design is rank deficient: remove redundant fixed effects or confounded genotypes"
            )
    variances = np.ones(len(penalties)) if initial is None else initial[1].copy()
    psi = 1.0 if initial is None else initial[0]
    history = []
    previous = np.inf
    converged = False
    for iteration in range(control.max_iter):
        precision = (penalties / variances[:, None]).sum(0)
        full_precision = np.r_[np.zeros(p), precision]
        influence = np.empty(p + q)
        coef = np.empty(p + q)
        if len(gi):
            d = diagonal / psi + full_precision[gi]
            b = cross / psi
            v = b / d[:, None]
            data_schur = rr / psi - b.T @ v
            K = data_schur.copy()
            K[np.diag_indices_from(K)] += full_precision[ri]
            Kinv, logdet = inverse_spd(K)
            coef[ri] = Kinv @ (ry / psi - v.T @ (gy / psi))
            coef[gi] = (gy / psi - b @ coef[ri]) / d
            vk = v @ Kinv
            correction = np.einsum("ij,ij->i", vk, v)
            influence[gi] = diagonal / (psi * d) - full_precision[gi] * correction
            influence[ri] = np.einsum("ij,ji->i", Kinv, data_schur)
            logdet += np.log(d).sum()
        else:
            K = rr / psi
            K[np.diag_indices_from(K)] += full_precision
            Kinv, logdet = inverse_spd(K)
            coef = Kinv @ (ry / psi)
            influence = np.einsum("ij,ji->i", Kinv, rr) / psi
        ed = (penalties / variances[:, None]) @ (
            np.maximum(influence[p:], 0) / precision
        )
        residual = y - A @ coef
        rss = np.dot(weights, residual**2)
        objective = (
            logdet
            - np.log(precision).sum()
            + n * np.log(psi)
            - np.log(weights[weights > 0]).sum()
            + rss / psi
            + np.dot(precision, coef[p:] ** 2)
        )
        history.append(
            {
                "iteration": iteration + 1,
                "objective": float(objective),
                "dispersion": float(psi),
                "variances": variances.copy(),
                "ed": ed.copy(),
            }
        )
        if control.monitoring:
            print(f"SAP {iteration + 1}: objective={objective:.8g}")
        if abs(previous - objective) < control.tolerance:
            converged = True
            break
        if iteration == control.max_iter - 1:
            break
        df = n - p - ed.sum()
        if df <= 0:
            raise ValueError(
                "Nonpositive residual effective dimension; model is not identifiable"
            )
        newpsi = rss / df if update_dispersion else 1.0
        if newpsi <= 0 or not np.isfinite(newpsi):
            raise ValueError(
                "Residual variance is zero or nonfinite; check the response and model"
            )
        variances = np.maximum(
            (penalties @ (coef[p:] ** 2)) / np.maximum(ed, 1e-50), 1e-50
        )
        psi = newpsi
        previous = objective
    if len(gi):
        covariance = np.empty((p + q, p + q))
        covariance[np.ix_(ri, ri)] = Kinv
        covariance[np.ix_(gi, ri)] = -vk
        covariance[np.ix_(ri, gi)] = -vk.T
        covariance[np.ix_(gi, gi)] = vk @ v.T + np.diag(1 / d)
    else:
        covariance = Kinv
    return EngineResult(
        coef,
        covariance,
        variances,
        float(psi),
        ed,
        float(objective),
        history,
        converged,
    )
