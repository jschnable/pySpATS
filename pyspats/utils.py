"""
Utility functions for SpATS package.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Union, Dict, Any


def SAP(x_coord: str, y_coord: str, nseg: Tuple[int, int] = (10, 10), 
        degree: int = 3, penalty_order: int = 2) -> Dict[str, Any]:
    """
    Specify Separable Anisotropic Penalty (SAP) spatial model.
    
    This function creates a spatial model specification for 2D P-splines
    using the SAP algorithm for efficient estimation.
    
    Parameters
    ----------
    x_coord : str
        Name of x-coordinate variable
    y_coord : str  
        Name of y-coordinate variable
    nseg : tuple of int, default=(10, 10)
        Number of segments in (x, y) directions
    degree : int, default=3
        Degree of B-spline basis (usually 1, 2, or 3)
    penalty_order : int, default=2
        Order of difference penalty (1 or 2)
        
    Returns
    -------
    dict
        Spatial model specification
        
    Examples
    --------
    >>> spatial_spec = SAP('col', 'row', nseg=(15, 20), degree=3)
    """
    return {
        'type': 'SAP',
        'x_coord': x_coord,
        'y_coord': y_coord, 
        'nseg': nseg,
        'degree': degree,
        'penalty_order': penalty_order
    }


def PSANOVA(x_coord: str, y_coord: str, nseg: Tuple[int, int] = (10, 10),
            degree: int = 3, penalty_order: int = 2) -> Dict[str, Any]:
    """
    Specify P-spline ANOVA spatial model.
    
    This function creates a P-spline ANOVA decomposition for spatial modeling,
    separating main effects and interaction terms.
    
    Parameters
    ---------- 
    x_coord : str
        Name of x-coordinate variable
    y_coord : str
        Name of y-coordinate variable
    nseg : tuple of int, default=(10, 10)
        Number of segments in (x, y) directions
    degree : int, default=3
        Degree of B-spline basis
    penalty_order : int, default=2
        Order of difference penalty
        
    Returns
    -------
    dict
        P-spline ANOVA model specification
        
    Examples
    --------
    >>> spatial_spec = PSANOVA('col', 'row', nseg=(10, 15))
    """
    return {
        'type': 'PSANOVA',
        'x_coord': x_coord,
        'y_coord': y_coord,
        'nseg': nseg, 
        'degree': degree,
        'penalty_order': penalty_order
    }


def interpret_formula(formula: Union[str, Tuple[str, str], Dict[str, Any]]) -> Dict[str, Any]:
    """
    Interpret spatial formula specification.
    
    Parameters
    ----------
    formula : str, tuple, or dict
        Spatial model specification
        
    Returns
    -------
    dict
        Parsed spatial specification
    """
    if isinstance(formula, dict):
        return formula
    elif isinstance(formula, tuple) and len(formula) == 2:
        return SAP(formula[0], formula[1])
    elif isinstance(formula, str):
        # Simple parsing - in full implementation would parse R-style formulas
        raise NotImplementedError("String formula parsing not yet implemented")
    else:
        raise ValueError(f"Invalid spatial specification: {formula}")


def create_position_indicator(dimensions: np.ndarray, is_random: np.ndarray) -> np.ndarray:
    """
    Create position indicators for random effects in coefficient vector.
    
    Parameters
    ----------
    dimensions : np.ndarray
        Dimension of each model component
    is_random : np.ndarray
        Boolean indicator for random effects
        
    Returns
    -------
    np.ndarray
        Position indices for random effects
    """
    positions = []
    cumsum = 0
    
    for i, (dim, random) in enumerate(zip(dimensions, is_random)):
        if random:
            positions.extend(range(cumsum, cumsum + int(dim)))
        cumsum += int(dim)
    
    return np.array(positions)


def get_attribute(obj: Any, attr: str) -> Any:
    """
    Safely get attribute from object.
    
    Parameters
    ----------
    obj : Any
        Object to get attribute from
    attr : str
        Attribute name
        
    Returns
    -------
    Any
        Attribute value or None if not found
    """
    return getattr(obj, attr, None)


def nominal_dimension(design_matrix: np.ndarray, is_fixed: np.ndarray) -> int:
    """
    Compute nominal dimension for random effects.
    
    The nominal dimension is the maximum effective dimension a random 
    effect can achieve, computed as rank[X, Z] - rank[X].
    
    Parameters
    ----------
    design_matrix : np.ndarray
        Combined design matrix [X, Z]
    is_fixed : np.ndarray
        Boolean indicator for fixed effects columns
        
    Returns
    -------
    int
        Nominal dimension
    """
    X = design_matrix[:, is_fixed]
    XZ = design_matrix
    
    rank_X = np.linalg.matrix_rank(X)
    rank_XZ = np.linalg.matrix_rank(XZ)
    
    return rank_XZ - rank_X


def deviance_residuals(y: np.ndarray, mu: np.ndarray, family: 'Family', 
                      weights: np.ndarray = None) -> np.ndarray:
    """
    Compute deviance residuals.
    
    Parameters
    ----------
    y : np.ndarray
        Observed response
    mu : np.ndarray  
        Fitted values
    family : Family
        Distribution family
    weights : np.ndarray, optional
        Observation weights
        
    Returns
    -------
    np.ndarray
        Deviance residuals
    """
    if weights is None:
        weights = np.ones_like(y)
    
    # Individual deviances
    dev_contributions = family.deviance_residuals(y, mu, weights)
    
    # Sign based on raw residuals
    signs = np.sign(y - mu)
    
    # Deviance residuals
    dev_resid = signs * np.sqrt(np.maximum(dev_contributions, 0))
    
    return dev_resid


def extract_spatial_coordinates(data: pd.DataFrame, 
                               spatial_spec: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract spatial coordinates from data based on specification.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    spatial_spec : dict
        Spatial model specification
        
    Returns
    -------
    tuple
        (x_coordinates, y_coordinates)
    """
    x_coord = spatial_spec['x_coord']
    y_coord = spatial_spec['y_coord']
    
    if x_coord not in data.columns:
        raise ValueError(f"x-coordinate '{x_coord}' not found in data")
    if y_coord not in data.columns:
        raise ValueError(f"y-coordinate '{y_coord}' not found in data")
    
    return data[x_coord].values, data[y_coord].values


def validate_data_structure(data: pd.DataFrame, response: str, 
                           genotype: str, spatial_spec: Dict[str, Any],
                           fixed: list = None, random: list = None) -> None:
    """
    Validate input data structure and variables.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    response : str
        Response variable name
    genotype : str
        Genotype variable name
    spatial_spec : dict
        Spatial specification
    fixed : list, optional
        Fixed effect variable names
    random : list, optional
        Random effect variable names
    """
    fixed = fixed or []
    random = random or []
    
    # Check required columns exist
    required_cols = [response, genotype, spatial_spec['x_coord'], spatial_spec['y_coord']]
    required_cols.extend(fixed)
    required_cols.extend(random)
    
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in data: {missing_cols}")
    
    # Check data types
    if not pd.api.types.is_numeric_dtype(data[response]):
        raise ValueError(f"Response variable '{response}' must be numeric")
    
    if not pd.api.types.is_numeric_dtype(data[spatial_spec['x_coord']]):
        raise ValueError(f"x-coordinate '{spatial_spec['x_coord']}' must be numeric")
    
    if not pd.api.types.is_numeric_dtype(data[spatial_spec['y_coord']]):
        raise ValueError(f"y-coordinate '{spatial_spec['y_coord']}' must be numeric")
    
    # Check for sufficient data
    if len(data) < 10:
        raise ValueError("Insufficient data: need at least 10 observations")
    
    # Check for missing spatial coordinates
    spatial_missing = (data[spatial_spec['x_coord']].isnull() | 
                      data[spatial_spec['y_coord']].isnull())
    if spatial_missing.sum() > 0.1 * len(data):
        raise ValueError("Too many missing spatial coordinates")


def compute_aic_bic(deviance: float, effective_dim: float, n_obs: int) -> Tuple[float, float]:
    """
    Compute AIC and BIC from model deviance.
    
    Parameters
    ----------
    deviance : float
        Model deviance
    effective_dim : float
        Effective number of parameters
    n_obs : int
        Number of observations
        
    Returns
    -------
    tuple
        (AIC, BIC)
    """
    aic = deviance + 2 * effective_dim
    bic = deviance + np.log(n_obs) * effective_dim
    
    return aic, bic