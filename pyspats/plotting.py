"""
Plotting functions for SpATS models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Optional, Tuple, List, Union
import warnings


def plot_spats(model: 'SpATS', which: str = 'spatial', figsize: Tuple[int, int] = (12, 8),
               all_in_one: bool = True) -> plt.Figure:
    """
    Plot SpATS model results.
    
    Parameters
    ----------
    model : SpATS
        Fitted SpATS model
    which : str, default='spatial'
        Type of plot: 'spatial', 'residuals', 'fitted', 'all'
    figsize : tuple, default=(12, 8)
        Figure size
    all_in_one : bool, default=True
        Whether to show all plots in one figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    if which == 'all':
        return _plot_all(model, figsize, all_in_one)
    elif which == 'spatial':
        return _plot_spatial_trend(model, figsize)
    elif which == 'residuals':
        return _plot_residuals(model, figsize)
    elif which == 'fitted':
        return _plot_fitted_vs_observed(model, figsize)
    else:
        raise ValueError(f"Unknown plot type: {which}")


def _plot_all(model: 'SpATS', figsize: Tuple[int, int], 
              all_in_one: bool = True) -> plt.Figure:
    """Plot all diagnostic plots."""
    if all_in_one:
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        _plot_spatial_trend(model, figsize=None, ax=axes[0])
        _plot_fitted_vs_observed(model, figsize=None, ax=axes[1])
        _plot_residuals(model, figsize=None, ax=axes[2])
        _plot_residual_spatial(model, figsize=None, ax=axes[3])
        
        plt.tight_layout()
        return fig
    else:
        # Create separate figures
        figs = []
        figs.append(_plot_spatial_trend(model, figsize))
        figs.append(_plot_fitted_vs_observed(model, figsize))
        figs.append(_plot_residuals(model, figsize))
        figs.append(_plot_residual_spatial(model, figsize))
        return figs


def _plot_spatial_trend(model: 'SpATS', figsize: Optional[Tuple[int, int]] = None,
                       ax: Optional[plt.Axes] = None) -> plt.Figure:
    """Plot spatial trend."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (10, 6))
    else:
        fig = ax.get_figure()
    
    # Extract spatial coordinates
    data = model.data[model.valid_obs]
    spatial_spec = model.spatial if isinstance(model.spatial, dict) else {'x_coord': model.spatial[0], 'y_coord': model.spatial[1]}
    
    x_coord = spatial_spec['x_coord'] if 'x_coord' in spatial_spec else model.spatial[0]
    y_coord = spatial_spec['y_coord'] if 'y_coord' in spatial_spec else model.spatial[1]
    
    x = data[x_coord].values
    y = data[y_coord].values
    fitted = model.fitted_values[model.valid_obs]
    
    # Create scatter plot with color representing fitted values
    scatter = ax.scatter(x, y, c=fitted, cmap='viridis', alpha=0.7, s=30)
    plt.colorbar(scatter, ax=ax, label='Fitted Values')
    
    ax.set_xlabel(x_coord.capitalize())
    ax.set_ylabel(y_coord.capitalize())
    ax.set_title('Spatial Trend')
    ax.grid(True, alpha=0.3)
    
    return fig


def _plot_fitted_vs_observed(model: 'SpATS', figsize: Optional[Tuple[int, int]] = None,
                            ax: Optional[plt.Axes] = None) -> plt.Figure:
    """Plot fitted vs observed values."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (8, 6))
    else:
        fig = ax.get_figure()
    
    valid_obs = model.valid_obs & ~pd.isnull(model.data[model.response])
    observed = model.data[model.response].values[valid_obs]
    fitted = model.fitted_values[valid_obs]
    
    # Remove NaN values
    valid_fitted = ~np.isnan(fitted) & ~np.isnan(observed)
    observed = observed[valid_fitted]
    fitted = fitted[valid_fitted]
    
    # Scatter plot
    ax.scatter(observed, fitted, alpha=0.6, s=20)
    
    # Add diagonal line
    min_val = min(np.min(observed), np.min(fitted))
    max_val = max(np.max(observed), np.max(fitted))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    
    # Calculate R²
    r_squared = np.corrcoef(observed, fitted)[0, 1]**2
    ax.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Observed')
    ax.set_ylabel('Fitted')
    ax.set_title('Fitted vs Observed Values')
    ax.grid(True, alpha=0.3)
    
    return fig


def _plot_residuals(model: 'SpATS', figsize: Optional[Tuple[int, int]] = None,
                   ax: Optional[plt.Axes] = None) -> plt.Figure:
    """Plot residuals vs fitted values."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (8, 6))
    else:
        fig = ax.get_figure()
    
    valid_obs = model.valid_obs & ~pd.isnull(model.data[model.response])
    fitted = model.fitted_values[valid_obs]
    residuals = model.residuals[valid_obs]
    
    # Remove NaN values
    valid_vals = ~np.isnan(fitted) & ~np.isnan(residuals)
    fitted = fitted[valid_vals]
    residuals = residuals[valid_vals]
    
    # Scatter plot
    ax.scatter(fitted, residuals, alpha=0.6, s=20)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals vs Fitted Values')
    ax.grid(True, alpha=0.3)
    
    return fig


def _plot_residual_spatial(model: 'SpATS', figsize: Optional[Tuple[int, int]] = None,
                          ax: Optional[plt.Axes] = None) -> plt.Figure:
    """Plot spatial distribution of residuals."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (10, 6))
    else:
        fig = ax.get_figure()
    
    # Extract spatial coordinates and residuals
    data = model.data[model.valid_obs]
    spatial_spec = model.spatial if isinstance(model.spatial, dict) else {'x_coord': model.spatial[0], 'y_coord': model.spatial[1]}
    
    x_coord = spatial_spec['x_coord'] if 'x_coord' in spatial_spec else model.spatial[0]
    y_coord = spatial_spec['y_coord'] if 'y_coord' in spatial_spec else model.spatial[1]
    
    x = data[x_coord].values
    y = data[y_coord].values
    residuals = model.residuals[model.valid_obs]
    
    # Remove NaN values
    valid_vals = ~np.isnan(residuals)
    x = x[valid_vals]
    y = y[valid_vals]
    residuals = residuals[valid_vals]
    
    # Create scatter plot with color representing residuals
    scatter = ax.scatter(x, y, c=residuals, cmap='RdBu_r', alpha=0.7, s=30,
                        vmin=-np.max(np.abs(residuals)), vmax=np.max(np.abs(residuals)))
    plt.colorbar(scatter, ax=ax, label='Residuals')
    
    ax.set_xlabel(x_coord.capitalize())
    ax.set_ylabel(y_coord.capitalize())
    ax.set_title('Spatial Distribution of Residuals')
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_variogram(variogram_obj: 'Variogram', figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot empirical variogram.
    
    Parameters
    ----------
    variogram_obj : Variogram
        Variogram object from variogram() function
    figsize : tuple, default=(10, 6)
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot empirical variogram
    ax.plot(variogram_obj.distances, variogram_obj.gamma, 'bo-', 
            label='Empirical Variogram', markersize=4)
    
    # Add fitted model if available
    if hasattr(variogram_obj, 'fitted_gamma'):
        ax.plot(variogram_obj.distances, variogram_obj.fitted_gamma, 'r-',
                label='Fitted Model', linewidth=2)
    
    ax.set_xlabel('Distance')
    ax.set_ylabel('Semivariance')
    ax.set_title('Empirical Variogram')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return fig


def plot_spatial_surface(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                        figsize: Tuple[int, int] = (12, 8),
                        title: str = 'Spatial Surface') -> plt.Figure:
    """
    Plot spatial surface as contour plot.
    
    Parameters
    ----------
    x, y, z : np.ndarray
        Spatial coordinates and values
    figsize : tuple, default=(12, 8)
        Figure size
    title : str, default='Spatial Surface'
        Plot title
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create regular grid for interpolation
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    xi = np.linspace(x_min, x_max, 50)
    yi = np.linspace(y_min, y_max, 50)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate z values to regular grid
    from scipy.interpolate import griddata
    Zi = griddata((x, y), z, (Xi, Yi), method='cubic', fill_value=np.nan)
    
    # Create contour plot
    contour = ax.contourf(Xi, Yi, Zi, levels=20, cmap='viridis', alpha=0.8)
    plt.colorbar(contour, ax=ax)
    
    # Add data points
    ax.scatter(x, y, c='white', s=10, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(title)
    
    return fig


def diagnostic_plots(model: 'SpATS', figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Create comprehensive diagnostic plots for SpATS model.
    
    Parameters
    ----------
    model : SpATS
        Fitted SpATS model
    figsize : tuple, default=(15, 10)
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object with multiple subplots
    """
    fig = plt.figure(figsize=figsize)
    
    # Create grid for subplots
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Fitted vs Observed
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_fitted_vs_observed(model, ax=ax1)
    
    # 2. Residuals vs Fitted
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_residuals(model, ax=ax2)
    
    # 3. QQ plot of residuals
    ax3 = fig.add_subplot(gs[0, 2])
    _plot_qq_residuals(model, ax=ax3)
    
    # 4. Spatial trend
    ax4 = fig.add_subplot(gs[1, :2])
    _plot_spatial_trend(model, ax=ax4)
    
    # 5. Residual spatial distribution
    ax5 = fig.add_subplot(gs[2, :2])
    _plot_residual_spatial(model, ax=ax5)
    
    # 6. Histogram of residuals
    ax6 = fig.add_subplot(gs[1:, 2])
    _plot_residual_histogram(model, ax=ax6)
    
    fig.suptitle(f'SpATS Model Diagnostics - {model.response}', fontsize=16)
    
    return fig


def _plot_qq_residuals(model: 'SpATS', ax: plt.Axes) -> None:
    """Plot Q-Q plot of residuals."""
    from scipy import stats
    
    valid_obs = model.valid_obs & ~pd.isnull(model.data[model.response])
    residuals = model.residuals[valid_obs]
    residuals = residuals[~np.isnan(residuals)]
    
    if len(residuals) > 0:
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot of Residuals')
        ax.grid(True, alpha=0.3)


def _plot_residual_histogram(model: 'SpATS', ax: plt.Axes) -> None:
    """Plot histogram of residuals."""
    valid_obs = model.valid_obs & ~pd.isnull(model.data[model.response])
    residuals = model.residuals[valid_obs]
    residuals = residuals[~np.isnan(residuals)]
    
    if len(residuals) > 0:
        ax.hist(residuals, bins=30, alpha=0.7, density=True, color='skyblue', edgecolor='black')
        
        # Overlay normal distribution
        x = np.linspace(residuals.min(), residuals.max(), 100)
        y = stats.norm.pdf(x, residuals.mean(), residuals.std())
        ax.plot(x, y, 'r-', linewidth=2, label='Normal')
        
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Density')
        ax.set_title('Histogram of Residuals')
        ax.legend()
        ax.grid(True, alpha=0.3)


def plot_spats_full(model: 'SpATS', all_in_one: bool = True, 
                   figsize: Tuple[int, int] = (15, 10), 
                   spa_trend: str = 'raw') -> plt.Figure:
    """
    Create full SpATS diagnostic plot with 6 panels (matching R SpATS behavior).
    
    Creates the following plots:
    1. Raw data
    2. Fitted data
    3. Residuals
    4. Spatial trend
    5. Genotypic predictions (BLUPs/BLUEs)
    6. Histogram of genotype coefficients
    
    Parameters
    ----------
    model : SpATS
        Fitted SpATS model
    all_in_one : bool, default=True
        Whether to show all plots in one figure
    figsize : tuple, default=(15, 10)
        Figure size
    spa_trend : str, default='raw'
        Format for spatial trend: 'raw' or 'percentage'
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    if all_in_one:
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        # 1. Raw data
        _plot_spatial_image(model, model.data[model.response].values, 
                           'Raw Data', axes[0])
        
        # 2. Fitted data
        _plot_spatial_image(model, model.fitted_values, 
                           'Fitted Data', axes[1])
        
        # 3. Residuals
        _plot_spatial_image(model, model.residuals, 
                           'Residuals', axes[2])
        
        # 4. Spatial trend
        _plot_spatial_trend_image(model, spa_trend, axes[3])
        
        # 5. Genotypic predictions
        _plot_genotype_predictions(model, axes[4])
        
        # 6. Histogram of genotype coefficients
        _plot_genotype_histogram(model, axes[5])
        
        plt.tight_layout()
        return fig
    else:
        # Create separate figures for each plot
        figs = []
        
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        _plot_spatial_image(model, model.data[model.response].values, 'Raw Data', ax1)
        figs.append(fig1)
        
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        _plot_spatial_image(model, model.fitted_values, 'Fitted Data', ax2)
        figs.append(fig2)
        
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        _plot_spatial_image(model, model.residuals, 'Residuals', ax3)
        figs.append(fig3)
        
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        _plot_spatial_trend_image(model, spa_trend, ax4)
        figs.append(fig4)
        
        fig5, ax5 = plt.subplots(figsize=(8, 6))
        _plot_genotype_predictions(model, ax5)
        figs.append(fig5)
        
        fig6, ax6 = plt.subplots(figsize=(8, 6))
        _plot_genotype_histogram(model, ax6)
        figs.append(fig6)
        
        return figs


def _plot_spatial_image(model: 'SpATS', values: np.ndarray, title: str, 
                       ax: plt.Axes) -> None:
    """Create a spatial image plot (similar to R's image.plot)."""
    # Get spatial coordinates
    data = model.data
    spatial_spec = model.spatial if isinstance(model.spatial, dict) else {
        'x_coord': model.spatial[0], 'y_coord': model.spatial[1]
    }
    
    x_coord = spatial_spec.get('x_coord', model.spatial[0])
    y_coord = spatial_spec.get('y_coord', model.spatial[1])
    
    x = data[x_coord].values
    y = data[y_coord].values
    
    # Create regular grid for plotting
    x_unique = np.sort(np.unique(x))
    y_unique = np.sort(np.unique(y))
    
    # Initialize grid with NaN
    grid = np.full((len(y_unique), len(x_unique)), np.nan)
    
    # Fill grid with values
    for i, val in enumerate(values):
        if not np.isnan(val):
            x_idx = np.where(x_unique == x[i])[0]
            y_idx = np.where(y_unique == y[i])[0]
            if len(x_idx) > 0 and len(y_idx) > 0:
                grid[y_idx[0], x_idx[0]] = val
    
    # Create image plot
    im = ax.imshow(grid, aspect='auto', origin='lower', 
                   extent=[x_unique.min(), x_unique.max(), 
                          y_unique.min(), y_unique.max()],
                   cmap='viridis')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    ax.set_xlabel(x_coord.capitalize())
    ax.set_ylabel(y_coord.capitalize())
    ax.set_title(title)


def _plot_smooth_spatial_surface(x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                                 title: str, ax: plt.Axes) -> None:
    """Create smooth spatial surface with high resolution."""
    from scipy.interpolate import griddata
    
    # Remove NaN values
    valid_mask = ~np.isnan(z)
    x_valid = x[valid_mask]
    y_valid = y[valid_mask] 
    z_valid = z[valid_mask]
    
    if len(z_valid) == 0:
        ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes, 
                ha='center', va='center')
        ax.set_title(title)
        return
    
    # Create high-resolution grid
    x_min, x_max = x_valid.min(), x_valid.max()
    y_min, y_max = y_valid.min(), y_valid.max()
    
    # Higher resolution for smooth gradients
    resolution = 100
    xi = np.linspace(x_min, x_max, resolution)
    yi = np.linspace(y_min, y_max, resolution)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate to create smooth surface
    try:
        Zi = griddata((x_valid, y_valid), z_valid, (Xi, Yi), 
                      method='cubic', fill_value=np.nan)
        
        # Create smooth surface plot
        im = ax.imshow(Zi, aspect='auto', origin='lower',
                       extent=[x_min, x_max, y_min, y_max],
                       cmap='viridis', interpolation='bilinear')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.7)
        
        # Add original data points as overlay
        ax.scatter(x_valid, y_valid, c='white', s=10, alpha=0.7, 
                  edgecolors='black', linewidth=0.5)
        
    except Exception:
        # Fallback to simple scatter plot if interpolation fails
        scatter = ax.scatter(x_valid, y_valid, c=z_valid, cmap='viridis', s=30)
        plt.colorbar(scatter, ax=ax, shrink=0.7)
    
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_title(title)


def _plot_spatial_trend_image(model: 'SpATS', spa_trend: str, ax: plt.Axes) -> None:
    """Plot spatial trend as high-resolution smooth surface."""
    # Get spatial coordinates
    data = model.data
    spatial_spec = model.spatial if isinstance(model.spatial, dict) else {
        'x_coord': model.spatial[0], 'y_coord': model.spatial[1]
    }
    
    x_coord = spatial_spec.get('x_coord', model.spatial[0])
    y_coord = spatial_spec.get('y_coord', model.spatial[1])
    
    x = data[x_coord].values
    y = data[y_coord].values
    
    # Use the extracted spatial effects (2D splines only)
    if hasattr(model, 'spatial_effects'):
        spatial_values = model.spatial_effects.copy()
    else:
        # Fallback to fitted values if spatial effects not available
        spatial_values = model.fitted_values.copy()
        # Try to approximate spatial component by removing fixed effects
        if hasattr(model, 'fixed_effects'):
            spatial_values = spatial_values - model.fixed_effects
    
    if spa_trend == 'percentage':
        # Convert to percentage relative to overall mean
        # Use mean of original response for percentage calculation
        mean_response = np.nanmean(model.data[model.response].values)
        spatial_values = (spatial_values / mean_response) * 100
        title = 'Spatial Trend (%)'
    else:
        title = 'Spatial Trend'
    
    # Create high-resolution smooth surface
    _plot_smooth_spatial_surface(x, y, spatial_values, title, ax)


def _plot_genotype_predictions(model: 'SpATS', ax: plt.Axes) -> None:
    """Plot genotypic predictions (BLUPs/BLUEs) spatially by row/column position."""
    data = model.data
    
    # Get proper BLUEs from fixed effects coefficients
    try:
        blues = model.get_BLUEs()
        blues_dict = blues.to_dict()
        
        # Create array with genotype BLUEs at each spatial position
        geno_predictions = np.full(len(data), np.nan)
        for i, geno in enumerate(data[model.genotype]):
            if geno in blues_dict:
                geno_predictions[i] = blues_dict[geno]
                
        title = 'Genotypic Predictions (BLUEs)'
        
    except (ValueError, AttributeError):
        # Fallback to old method if BLUEs not available
        genotypes = data[model.genotype].unique()
        geno_means = {}
        for geno in genotypes:
            geno_mask = (data[model.genotype] == geno) & model.valid_obs
            if np.any(geno_mask):
                geno_means[geno] = np.nanmean(model.fitted_values[geno_mask])
        
        geno_predictions = np.full(len(data), np.nan)
        for i, geno in enumerate(data[model.genotype]):
            if geno in geno_means:
                geno_predictions[i] = geno_means[geno]
                
        title = 'Genotypic Predictions (Fitted Averages)'
    
    # Plot spatially using the same format as raw/fitted data
    _plot_spatial_image(model, geno_predictions, title, ax)


def _plot_genotype_histogram(model: 'SpATS', ax: plt.Axes) -> None:
    """Plot histogram of genotype coefficients."""
    data = model.data
    genotypes = data[model.genotype].unique()
    
    # Calculate mean fitted value per genotype
    geno_means = []
    for geno in genotypes:
        geno_mask = (data[model.genotype] == geno) & model.valid_obs
        if np.any(geno_mask):
            geno_means.append(np.nanmean(model.fitted_values[geno_mask]))
    
    geno_means = np.array(geno_means)
    valid_means = geno_means[~np.isnan(geno_means)]
    
    if len(valid_means) > 0:
        ax.hist(valid_means, bins=min(20, len(valid_means)), 
                alpha=0.7, color='lightcoral', edgecolor='black')
        
        ax.set_xlabel('Genotype Coefficient')
        ax.set_ylabel('Frequency')
        ax.set_title('Histogram of Genotype Coefficients')
        ax.grid(True, alpha=0.3)