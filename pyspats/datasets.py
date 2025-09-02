"""
Example datasets for SpATS package.
"""

import pandas as pd
import numpy as np
from typing import Optional


def load_wheatdata() -> pd.DataFrame:
    """
    Load the wheat field trial dataset.
    
    This is a simulated version of the wheat dataset from the R SpATS package,
    containing yield measurements from a field trial with spatial coordinates.
    
    Returns
    -------
    pd.DataFrame
        Wheat field trial data with columns:
        - yield: grain yield (response variable)
        - geno: genotype identifier (factor)
        - rep: replicate number (factor) 
        - row: row position (integer)
        - col: column position (integer)
        - rowcode: row coding factor
        - colcode: column coding factor
        
    Examples
    --------
    >>> data = load_wheatdata()
    >>> print(data.head())
    >>> print(data.describe())
    """
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Field dimensions
    n_rows = 22
    n_cols = 15
    n_obs = n_rows * n_cols
    
    # Generate spatial coordinates
    rows = np.repeat(np.arange(1, n_rows + 1), n_cols)
    cols = np.tile(np.arange(1, n_cols + 1), n_rows)
    
    # Generate genotypes (107 unique genotypes)
    n_genotypes = 107
    genotypes = np.random.choice(np.arange(1, n_genotypes + 1), size=n_obs, replace=True)
    
    # Generate replicates (3 replicates)
    reps = np.random.choice([1, 2, 3], size=n_obs, p=[0.4, 0.35, 0.25])
    
    # Generate row and column codes
    rowcode = np.random.choice([1, 2], size=n_obs, p=[0.6, 0.4])
    colcode = np.random.choice([1, 2, 3, 4], size=n_obs, p=[0.3, 0.3, 0.2, 0.2])
    
    # Generate spatial trend (smooth 2D function)
    x_scaled = (cols - 1) / (n_cols - 1)  # Scale to [0, 1]
    y_scaled = (rows - 1) / (n_rows - 1)  # Scale to [0, 1]
    
    # Smooth spatial trend
    spatial_trend = (50 * np.sin(2 * np.pi * x_scaled) * np.cos(2 * np.pi * y_scaled) +
                    30 * np.exp(-((x_scaled - 0.5)**2 + (y_scaled - 0.5)**2) / 0.2))
    
    # Genotype effects (some genotypes perform better)
    genotype_effects = np.random.normal(0, 40, n_genotypes)
    geno_effect = genotype_effects[genotypes - 1]  # Map to observations
    
    # Rep effects
    rep_effects = {1: 0, 2: 15, 3: -10}
    rep_effect = np.array([rep_effects[r] for r in reps])
    
    # Row and column code effects
    rowcode_effect = np.where(rowcode == 1, -10, 10)
    colcode_effects = {1: 0, 2: 20, 3: -15, 4: 5}
    colcode_effect = np.array([colcode_effects[c] for c in colcode])
    
    # Generate yields
    base_yield = 450
    error = np.random.normal(0, 35, n_obs)
    
    yield_values = (base_yield + spatial_trend + geno_effect + rep_effect + 
                   rowcode_effect + colcode_effect + error)
    
    # Ensure positive yields
    yield_values = np.maximum(yield_values, 50)
    
    # Round to integers
    yield_values = np.round(yield_values).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        'yield': yield_values,
        'geno': pd.Categorical(genotypes.astype(str)),
        'rep': pd.Categorical(reps.astype(str)), 
        'row': rows,
        'col': cols,
        'rowcode': pd.Categorical(rowcode.astype(str)),
        'colcode': pd.Categorical(colcode.astype(str))
    })
    
    # Add row and column factors (as in original R data)
    data['R'] = pd.Categorical(data['row'].astype(str))
    data['C'] = pd.Categorical(data['col'].astype(str))
    
    return data


def generate_field_trial_data(
    n_rows: int = 20,
    n_cols: int = 15, 
    n_genotypes: int = 50,
    spatial_variance: float = 100.0,
    genotype_variance: float = 50.0,
    error_variance: float = 25.0,
    missing_rate: float = 0.0,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate simulated field trial data.
    
    Parameters
    ----------
    n_rows : int, default=20
        Number of rows in field
    n_cols : int, default=15
        Number of columns in field  
    n_genotypes : int, default=50
        Number of genotypes
    spatial_variance : float, default=100.0
        Variance of spatial trend
    genotype_variance : float, default=50.0
        Variance of genotype effects
    error_variance : float, default=25.0
        Error variance
    missing_rate : float, default=0.0
        Proportion of missing observations
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    pd.DataFrame
        Simulated field trial data
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_obs = n_rows * n_cols
    
    # Spatial coordinates
    rows = np.repeat(np.arange(1, n_rows + 1), n_cols)
    cols = np.tile(np.arange(1, n_cols + 1), n_rows)
    
    # Normalize coordinates for spatial trend
    x_norm = (cols - 1) / (n_cols - 1)
    y_norm = (rows - 1) / (n_rows - 1)
    
    # Generate spatial trend
    spatial_trend = np.sqrt(spatial_variance) * (
        0.5 * np.sin(2 * np.pi * x_norm) + 
        0.3 * np.cos(2 * np.pi * y_norm) +
        0.4 * np.sin(np.pi * x_norm) * np.cos(np.pi * y_norm)
    )
    
    # Assign genotypes
    genotypes = np.random.choice(np.arange(1, n_genotypes + 1), size=n_obs, replace=True)
    
    # Genotype effects
    genotype_effects = np.random.normal(0, np.sqrt(genotype_variance), n_genotypes)
    geno_effect = genotype_effects[genotypes - 1]
    
    # Generate response
    base_response = 100
    error = np.random.normal(0, np.sqrt(error_variance), n_obs)
    response = base_response + spatial_trend + geno_effect + error
    
    # Add missing values
    if missing_rate > 0:
        missing_idx = np.random.choice(n_obs, size=int(n_obs * missing_rate), replace=False)
        response[missing_idx] = np.nan
    
    # Create DataFrame
    data = pd.DataFrame({
        'response': response,
        'genotype': pd.Categorical(genotypes.astype(str)),
        'row': rows,
        'col': cols,
        'block': pd.Categorical(np.random.choice(['A', 'B', 'C'], size=n_obs)),
        'treatment': pd.Categorical(np.random.choice(['T1', 'T2'], size=n_obs))
    })
    
    return data


def load_example_spatial_data(dataset: str = 'wheat') -> pd.DataFrame:
    """
    Load example spatial datasets.
    
    Parameters
    ----------
    dataset : str, default='wheat'
        Dataset name: 'wheat' or 'simulated'
        
    Returns
    -------
    pd.DataFrame
        Example dataset
    """
    if dataset == 'wheat':
        return load_wheatdata()
    elif dataset == 'simulated':
        return generate_field_trial_data(seed=123)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def create_toy_example() -> pd.DataFrame:
    """
    Create a small toy example for testing and demonstrations.
    
    Returns
    -------
    pd.DataFrame
        Small toy dataset
    """
    return generate_field_trial_data(
        n_rows=8, 
        n_cols=6,
        n_genotypes=15,
        seed=42
    )