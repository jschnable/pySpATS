"""
Variogram analysis for spatial correlation assessment.
"""

import numpy as np
from scipy.optimize import curve_fit
from typing import Dict, Any
import warnings


class Variogram:
    """
    Empirical variogram for spatial correlation analysis.

    Attributes
    ----------
    distances : np.ndarray
        Distance bins
    gamma : np.ndarray
        Semivariance values
    n_pairs : np.ndarray
        Number of pairs in each distance bin
    """

    def __init__(self, distances: np.ndarray, gamma: np.ndarray, n_pairs: np.ndarray):
        self.distances = distances
        self.gamma = gamma
        self.n_pairs = n_pairs
        self.fitted_model = None
        self.fitted_params = None


def _pairs(x, y, residual, direction=None, tolerance=22.5):
    # Bounded temporary storage; each unordered pair appears exactly once.
    n = len(x)
    for start in range(0, n, 128):
        stop = min(start + 128, n)
        i = np.arange(start, stop)[:, None]
        j = np.arange(n)[None, :]
        dx = x[None, :] - x[start:stop, None]
        dy = y[None, :] - y[start:stop, None]
        mask = j > i
        if direction is not None:
            angles = np.degrees(np.arctan2(dy, dx)) % 180
            delta = np.abs(angles - direction % 180)
            mask &= np.minimum(delta, 180 - delta) <= tolerance
        distances = np.hypot(dx, dy)[mask]
        gamma = (0.5 * (residual[None, :] - residual[start:stop, None]) ** 2)[mask]
        yield distances, gamma


def _empirical(model, max_dist, n_bins, direction=None, tolerance=22.5):
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    used = model.observed & np.isfinite(model.residuals)
    x = model.data.loc[used, model.spec.x].to_numpy(float)
    y = model.data.loc[used, model.spec.y].to_numpy(float)
    residual = model.residuals[used]
    if len(x) < 10:
        raise ValueError("Insufficient valid observations for variogram computation")
    if max_dist is None:
        max_dist = (
            max(
                (
                    d.max(initial=0)
                    for d, _ in _pairs(x, y, residual, direction, tolerance)
                ),
                default=0,
            )
            / 3
        )
    if not np.isfinite(max_dist) or max_dist <= 0:
        if direction is not None and max_dist == 0:
            return Variogram(np.array([]), np.array([]), np.array([], dtype=int))
        raise ValueError("max_dist must be finite and positive")
    edges = np.linspace(0, max_dist, n_bins + 1)
    counts = np.zeros(n_bins, dtype=int)
    sums = np.zeros(n_bins)
    for distance, gamma in _pairs(x, y, residual, direction, tolerance):
        counts += np.histogram(distance, bins=edges)[0]
        sums += np.histogram(distance, bins=edges, weights=gamma)[0]
    use = counts > 0
    if not use.any() and direction is None:
        raise ValueError("No pairs within specified maximum distance")
    return Variogram(
        ((edges[:-1] + edges[1:]) / 2)[use], sums[use] / counts[use], counts[use]
    )


def variogram(model, max_dist=None, n_bins=15, cutoff=None):
    """Exact empirical semivariogram of response residuals on fitted plots.

    Uses raw coordinate distances (use physical units for unequal plot spacing).
    Pair computations take quadratic time with bounded working memory. The
    rightmost bin includes its endpoint; zero-weight plots are excluded.
    """
    return _empirical(model, cutoff if cutoff is not None else max_dist, n_bins)


def directional_variogram(model, direction, tolerance=22.5, max_dist=None, n_bins=15):
    """Axial empirical variogram: directions 0 and 180 degrees are equivalent.

    Each unordered pair counts once, independent of input row ordering.
    """
    if (
        not np.isfinite(direction)
        or not np.isfinite(tolerance)
        or not 0 <= tolerance <= 90
    ):
        raise ValueError("direction must be finite and tolerance must be in [0,90]")
    return _empirical(model, max_dist, n_bins, direction, tolerance)


def fit_variogram_model(
    variogram_obj: Variogram, model: str = "spherical"
) -> Dict[str, Any]:
    """
    Fit theoretical variogram model to empirical variogram.

    Parameters
    ----------
    variogram_obj : Variogram
        Empirical variogram object
    model : str, default='spherical'
        Variogram model type: 'spherical', 'exponential', or 'gaussian'

    Returns
    -------
    dict
        Fitted model parameters and goodness of fit
    """
    distances = variogram_obj.distances
    gamma = variogram_obj.gamma
    n_pairs = variogram_obj.n_pairs

    # Weight by number of pairs
    weights = np.sqrt(n_pairs)

    # Initial parameter estimates
    nugget_init = np.min(gamma) if np.min(gamma) > 0 else gamma[0] * 0.1
    sill_init = np.max(gamma) - nugget_init
    range_init = (
        distances[np.argmax(gamma > 0.95 * np.max(gamma))]
        if len(distances) > 1
        else np.max(distances) / 3
    )

    # Select model function
    if model == "spherical":
        model_func = _spherical_model
    elif model == "exponential":
        model_func = _exponential_model
    elif model == "gaussian":
        model_func = _gaussian_model
    else:
        raise ValueError(f"Unknown variogram model: {model}")

    # Fit model
    try:
        initial_params = [nugget_init, sill_init, range_init]
        fitted_params, covariance = curve_fit(
            model_func,
            distances,
            gamma,
            p0=initial_params,
            sigma=1 / weights,
            absolute_sigma=False,
            bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
        )

        # Compute fitted values
        fitted_gamma = model_func(distances, *fitted_params)

        # Compute R²
        ss_res = np.sum((gamma - fitted_gamma) ** 2)
        ss_tot = np.sum((gamma - np.mean(gamma)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Store results in variogram object
        variogram_obj.fitted_model = model
        variogram_obj.fitted_params = {
            "nugget": fitted_params[0],
            "sill": fitted_params[1],
            "range": fitted_params[2],
        }
        variogram_obj.fitted_gamma = fitted_gamma

        return {
            "model": model,
            "nugget": fitted_params[0],
            "sill": fitted_params[1],
            "range": fitted_params[2],
            "r_squared": r_squared,
            "covariance": covariance,
        }

    except Exception as e:
        warnings.warn(f"Variogram model fitting failed: {str(e)}")
        return {
            "model": model,
            "nugget": nugget_init,
            "sill": sill_init,
            "range": range_init,
            "r_squared": 0.0,
            "fitted": False,
        }


def _spherical_model(
    h: np.ndarray, nugget: float, sill: float, range_param: float
) -> np.ndarray:
    """Spherical variogram model."""
    gamma = np.full_like(h, nugget + sill)

    mask = h < range_param
    gamma[mask] = nugget + sill * (
        1.5 * h[mask] / range_param - 0.5 * (h[mask] / range_param) ** 3
    )

    return gamma


def _exponential_model(
    h: np.ndarray, nugget: float, sill: float, range_param: float
) -> np.ndarray:
    """Exponential variogram model."""
    return nugget + sill * (1 - np.exp(-h / range_param))


def _gaussian_model(
    h: np.ndarray, nugget: float, sill: float, range_param: float
) -> np.ndarray:
    """Gaussian variogram model."""
    return nugget + sill * (1 - np.exp(-((h / range_param) ** 2)))
