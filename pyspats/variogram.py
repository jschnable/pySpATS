"""
Variogram analysis for spatial correlation assessment.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit
from typing import Optional, Tuple, Dict, Any
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


def variogram(model: 'SpATS', max_dist: Optional[float] = None, 
              n_bins: int = 15, cutoff: Optional[float] = None) -> Variogram:
    """
    Compute empirical variogram from SpATS model residuals.
    
    Parameters
    ----------
    model : SpATS
        Fitted SpATS model
    max_dist : float, optional
        Maximum distance for variogram computation
    n_bins : int, default=15
        Number of distance bins
    cutoff : float, optional
        Distance cutoff (alternative to max_dist)
        
    Returns
    -------
    Variogram
        Variogram object containing distances and semivariances
        
    Examples
    --------
    >>> var_obj = variogram(fitted_model)
    >>> plot_variogram(var_obj)
    """
    # Extract spatial coordinates and residuals
    data = model.data[model.valid_obs]
    
    # Get spatial coordinates
    if isinstance(model.spatial, tuple):
        x_coord, y_coord = model.spatial
    elif isinstance(model.spatial, dict):
        x_coord = model.spatial['x_coord']
        y_coord = model.spatial['y_coord']
    else:
        raise ValueError("Cannot determine spatial coordinates from model")
    
    x = data[x_coord].values
    y = data[y_coord].values
    residuals = model.residuals[model.valid_obs]
    
    # Remove missing values
    valid_idx = ~(np.isnan(x) | np.isnan(y) | np.isnan(residuals))
    x = x[valid_idx]
    y = y[valid_idx]
    residuals = residuals[valid_idx]
    
    if len(x) < 10:
        raise ValueError("Insufficient valid observations for variogram computation")
    
    # Compute pairwise distances
    coords = np.column_stack([x, y])
    distances = pdist(coords)
    
    # Compute pairwise semivariances  
    residual_diffs = pdist(residuals.reshape(-1, 1), metric='sqeuclidean')
    semivariances = 0.5 * residual_diffs
    
    # Set maximum distance
    if cutoff is not None:
        max_dist = cutoff
    elif max_dist is None:
        max_dist = np.max(distances) / 3  # Common rule of thumb
    
    # Filter by maximum distance
    valid_pairs = distances <= max_dist
    distances = distances[valid_pairs]
    semivariances = semivariances[valid_pairs]
    
    if len(distances) == 0:
        raise ValueError("No pairs within specified maximum distance")
    
    # Create distance bins
    bin_edges = np.linspace(0, max_dist, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Compute empirical variogram
    gamma_values = np.zeros(n_bins)
    n_pairs_values = np.zeros(n_bins, dtype=int)
    
    for i in range(n_bins):
        bin_mask = (distances >= bin_edges[i]) & (distances < bin_edges[i + 1])
        
        if np.sum(bin_mask) > 0:
            gamma_values[i] = np.mean(semivariances[bin_mask])
            n_pairs_values[i] = np.sum(bin_mask)
    
    # Remove empty bins
    non_empty = n_pairs_values > 0
    bin_centers = bin_centers[non_empty]
    gamma_values = gamma_values[non_empty]
    n_pairs_values = n_pairs_values[non_empty]
    
    return Variogram(bin_centers, gamma_values, n_pairs_values)


def fit_variogram_model(variogram_obj: Variogram, model: str = 'spherical') -> Dict[str, Any]:
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
    range_init = distances[np.argmax(gamma > 0.95 * np.max(gamma))] if len(distances) > 1 else np.max(distances) / 3
    
    # Select model function
    if model == 'spherical':
        model_func = _spherical_model
    elif model == 'exponential':
        model_func = _exponential_model
    elif model == 'gaussian':
        model_func = _gaussian_model
    else:
        raise ValueError(f"Unknown variogram model: {model}")
    
    # Fit model
    try:
        initial_params = [nugget_init, sill_init, range_init]
        fitted_params, covariance = curve_fit(
            model_func, distances, gamma, p0=initial_params,
            sigma=1/weights, absolute_sigma=False,
            bounds=([0, 0, 0], [np.inf, np.inf, np.inf])
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
            'nugget': fitted_params[0],
            'sill': fitted_params[1], 
            'range': fitted_params[2]
        }
        variogram_obj.fitted_gamma = fitted_gamma
        
        return {
            'model': model,
            'nugget': fitted_params[0],
            'sill': fitted_params[1],
            'range': fitted_params[2],
            'r_squared': r_squared,
            'covariance': covariance
        }
        
    except Exception as e:
        warnings.warn(f"Variogram model fitting failed: {str(e)}")
        return {
            'model': model,
            'nugget': nugget_init,
            'sill': sill_init,
            'range': range_init,
            'r_squared': 0.0,
            'fitted': False
        }


def _spherical_model(h: np.ndarray, nugget: float, sill: float, range_param: float) -> np.ndarray:
    """Spherical variogram model."""
    gamma = np.full_like(h, nugget + sill)
    
    mask = h < range_param
    gamma[mask] = nugget + sill * (1.5 * h[mask] / range_param - 0.5 * (h[mask] / range_param) ** 3)
    
    return gamma


def _exponential_model(h: np.ndarray, nugget: float, sill: float, range_param: float) -> np.ndarray:
    """Exponential variogram model."""
    return nugget + sill * (1 - np.exp(-h / range_param))


def _gaussian_model(h: np.ndarray, nugget: float, sill: float, range_param: float) -> np.ndarray:
    """Gaussian variogram model.""" 
    return nugget + sill * (1 - np.exp(-(h / range_param) ** 2))


def directional_variogram(model: 'SpATS', direction: float, tolerance: float = 22.5,
                         max_dist: Optional[float] = None, n_bins: int = 15) -> Variogram:
    """
    Compute directional variogram.
    
    Parameters
    ----------
    model : SpATS
        Fitted SpATS model
    direction : float
        Direction in degrees (0 = East, 90 = North)
    tolerance : float, default=22.5
        Angular tolerance in degrees
    max_dist : float, optional
        Maximum distance
    n_bins : int, default=15
        Number of distance bins
        
    Returns
    -------
    Variogram
        Directional variogram object
    """
    # Extract spatial coordinates and residuals
    data = model.data[model.valid_obs]
    
    if isinstance(model.spatial, tuple):
        x_coord, y_coord = model.spatial
    elif isinstance(model.spatial, dict):
        x_coord = model.spatial['x_coord']
        y_coord = model.spatial['y_coord']
    else:
        raise ValueError("Cannot determine spatial coordinates from model")
    
    x = data[x_coord].values
    y = data[y_coord].values
    residuals = model.residuals[model.valid_obs]
    
    # Remove missing values
    valid_idx = ~(np.isnan(x) | np.isnan(y) | np.isnan(residuals))
    x = x[valid_idx]
    y = y[valid_idx]
    residuals = residuals[valid_idx]
    
    n_points = len(x)
    
    # Compute all pairwise vectors
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    
    # Compute distances and angles
    distances = np.sqrt(dx**2 + dy**2)
    angles = np.degrees(np.arctan2(dy, dx))
    
    # Normalize angles to [0, 360)
    angles = (angles + 360) % 360
    
    # Filter by direction
    direction = direction % 360
    angle_diff = np.minimum(
        np.abs(angles - direction),
        360 - np.abs(angles - direction)
    )
    directional_mask = angle_diff <= tolerance
    
    # Get upper triangular indices (avoid double counting)
    triu_indices = np.triu_indices(n_points, k=1)
    mask = directional_mask[triu_indices]
    
    # Extract directional pairs
    pair_distances = distances[triu_indices][mask]
    
    # Compute semivariances
    residual_diffs = (residuals[:, np.newaxis] - residuals[np.newaxis, :]) ** 2
    pair_semivariances = 0.5 * residual_diffs[triu_indices][mask]
    
    # Set maximum distance
    if max_dist is None:
        if len(pair_distances) > 0:
            max_dist = np.max(pair_distances) / 3
        else:
            max_dist = 1.0  # Default fallback
    
    # Filter by maximum distance
    valid_pairs = pair_distances <= max_dist
    pair_distances = pair_distances[valid_pairs]
    pair_semivariances = pair_semivariances[valid_pairs]
    
    if len(pair_distances) == 0:
        # Return empty variogram for directions with no data
        return Variogram(np.array([]), np.array([]), np.array([], dtype=int))
    
    # Create distance bins
    bin_edges = np.linspace(0, max_dist, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Compute empirical variogram
    gamma_values = np.zeros(n_bins)
    n_pairs_values = np.zeros(n_bins, dtype=int)
    
    for i in range(n_bins):
        bin_mask = (pair_distances >= bin_edges[i]) & (pair_distances < bin_edges[i + 1])
        
        if np.sum(bin_mask) > 0:
            gamma_values[i] = np.mean(pair_semivariances[bin_mask])
            n_pairs_values[i] = np.sum(bin_mask)
    
    # Remove empty bins
    non_empty = n_pairs_values > 0
    bin_centers = bin_centers[non_empty]
    gamma_values = gamma_values[non_empty]
    n_pairs_values = n_pairs_values[non_empty]
    
    return Variogram(bin_centers, gamma_values, n_pairs_values)