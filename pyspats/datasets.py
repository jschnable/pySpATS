"""
Example datasets for SpATS package.
"""

import pandas as pd
import numpy as np
from typing import Optional


def load_wheatdata() -> pd.DataFrame:
    """Load the real wheat trial distributed with R SpATS 1.0-20 (GPL).

    This replaces the synthetic stand-in used in pySpATS 0.1. The R and C
    columns are categorical copies of the numeric field coordinates, useful
    for independent row/column random effects. Returns a fresh DataFrame.
    """
    from importlib.resources import files

    with files("pyspats").joinpath("data/wheat.csv").open("r") as stream:
        data = pd.read_csv(stream)
    for c in ("geno", "rep", "rowcode", "colcode"):
        data[c] = data[c].astype(str).astype("category")
    data["R"] = data["row"].astype(str).astype("category")
    data["C"] = data["col"].astype(str).astype("category")
    return data


def generate_field_trial_data(
    n_rows: int = 20,
    n_cols: int = 15,
    n_genotypes: int = 50,
    spatial_variance: float = 100.0,
    genotype_variance: float = 50.0,
    error_variance: float = 25.0,
    missing_rate: float = 0.0,
    seed: Optional[int] = None,
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
    rng = np.random.RandomState(seed)

    n_obs = n_rows * n_cols

    # Spatial coordinates
    rows = np.repeat(np.arange(1, n_rows + 1), n_cols)
    cols = np.tile(np.arange(1, n_cols + 1), n_rows)

    # Normalize coordinates for spatial trend
    x_norm = (cols - 1) / (n_cols - 1)
    y_norm = (rows - 1) / (n_rows - 1)

    # Generate spatial trend
    spatial_trend = np.sqrt(spatial_variance) * (
        0.5 * np.sin(2 * np.pi * x_norm)
        + 0.3 * np.cos(2 * np.pi * y_norm)
        + 0.4 * np.sin(np.pi * x_norm) * np.cos(np.pi * y_norm)
    )

    # Assign genotypes
    genotypes = rng.choice(np.arange(1, n_genotypes + 1), size=n_obs, replace=True)

    # Genotype effects
    genotype_effects = rng.normal(0, np.sqrt(genotype_variance), n_genotypes)
    geno_effect = genotype_effects[genotypes - 1]

    # Generate response
    base_response = 100
    error = rng.normal(0, np.sqrt(error_variance), n_obs)
    response = base_response + spatial_trend + geno_effect + error

    # Add missing values
    if missing_rate > 0:
        missing_idx = rng.choice(n_obs, size=int(n_obs * missing_rate), replace=False)
        response[missing_idx] = np.nan

    # Create DataFrame
    data = pd.DataFrame(
        {
            "response": response,
            "genotype": pd.Categorical(genotypes.astype(str)),
            "row": rows,
            "col": cols,
            "block": pd.Categorical(rng.choice(["A", "B", "C"], size=n_obs)),
            "treatment": pd.Categorical(rng.choice(["T1", "T2"], size=n_obs)),
        }
    )

    return data


def load_example_spatial_data(dataset: str = "wheat") -> pd.DataFrame:
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
    if dataset == "wheat":
        return load_wheatdata()
    elif dataset == "simulated":
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
    return generate_field_trial_data(n_rows=8, n_cols=6, n_genotypes=15, seed=42)
