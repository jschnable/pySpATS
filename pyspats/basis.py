"""
Basis construction for P-splines and design matrices.
"""

import numpy as np
from scipy import sparse
from scipy.interpolate import BSpline
from typing import Tuple, List, Union


def bspline_basis(x: np.ndarray, knots: np.ndarray, degree: int = 3) -> np.ndarray:
    """
    Construct B-spline basis functions.
    
    Parameters
    ----------
    x : np.ndarray
        Evaluation points
    knots : np.ndarray
        Knot sequence
    degree : int, default=3
        Spline degree
        
    Returns
    -------
    np.ndarray
        B-spline basis matrix
    """
    n_basis = len(knots) - degree - 1
    basis_matrix = np.zeros((len(x), n_basis))
    
    for i in range(n_basis):
        # Construct B-spline basis function i
        coeff = np.zeros(n_basis)
        coeff[i] = 1.0
        spline = BSpline(knots, coeff, degree)
        basis_matrix[:, i] = spline(x)
    
    return basis_matrix


def construct_knots(x: np.ndarray, nseg: int, degree: int = 3) -> np.ndarray:
    """
    Construct knot sequence for P-splines.
    
    Parameters
    ----------
    x : np.ndarray
        Data points
    nseg : int
        Number of segments
    degree : int, default=3
        Spline degree
        
    Returns
    -------
    np.ndarray
        Knot sequence
    """
    x_min, x_max = np.min(x), np.max(x)
    
    # Handle degenerate case where all x values are the same
    if x_max - x_min < 1e-10:
        # Create artificial range
        x_range = max(1.0, abs(x_min))
        x_min = x_min - 0.5 * x_range
        x_max = x_min + x_range
    
    # Interior knots
    interior_knots = np.linspace(x_min, x_max, nseg + 1)
    
    # Add boundary knots
    dx = (x_max - x_min) / nseg
    left_knots = np.array([x_min - (degree - i) * dx for i in range(degree)])
    right_knots = np.array([x_max + (i + 1) * dx for i in range(degree)])
    
    knots = np.concatenate([left_knots, interior_knots, right_knots])
    return np.sort(knots)


def penalty_matrix(n_basis: int, order: int = 2) -> sparse.csr_matrix:
    """
    Construct difference penalty matrix.
    
    Parameters
    ----------
    n_basis : int
        Number of basis functions
    order : int, default=2
        Order of differences
        
    Returns
    -------
    sparse.csr_matrix
        Penalty matrix (n_basis x n_basis)
    """
    if order == 0 or n_basis <= order:
        return sparse.eye(n_basis)
    
    # Build difference matrix using numpy and convert to sparse
    D = np.zeros((n_basis - order, n_basis))
    
    if order == 1:
        # First differences: [1, -1, 0, ...], [0, 1, -1, 0, ...], etc.
        for i in range(n_basis - 1):
            D[i, i] = 1
            D[i, i+1] = -1
    elif order == 2:
        # Second differences: [1, -2, 1, 0, ...], [0, 1, -2, 1, 0, ...], etc.
        for i in range(n_basis - 2):
            D[i, i] = 1
            D[i, i+1] = -2
            D[i, i+2] = 1
    else:
        # Higher order differences using binomial coefficients
        # For order k, coefficients are (-1)^i * C(k,i)
        from scipy.special import comb
        coeffs = np.array([(-1)**i * comb(order, i) for i in range(order + 1)])
        
        for i in range(n_basis - order):
            for j in range(order + 1):
                D[i, i + j] = coeffs[j]
    
    # Convert to sparse and compute penalty
    D_sparse = sparse.csr_matrix(D)
    P = D_sparse.T @ D_sparse
    
    return P


def construct_2d_pspline(
    x: np.ndarray, 
    y: np.ndarray,
    nseg: Tuple[int, int] = (10, 10),
    degree: int = 3,
    penalty_order: int = 2
) -> Tuple[np.ndarray, List[sparse.csr_matrix]]:
    """
    Construct 2D P-spline basis and penalty matrices.
    
    Parameters
    ----------
    x, y : np.ndarray
        Spatial coordinates
    nseg : tuple of int, default=(10, 10)
        Number of segments in each dimension
    degree : int, default=3
        Spline degree
    penalty_order : int, default=2
        Order of penalty
        
    Returns
    -------
    tuple
        - Basis matrix (n_obs x n_basis)
        - List of penalty matrices
    """
    nseg_x, nseg_y = nseg
    
    # Construct 1D bases
    knots_x = construct_knots(x, nseg_x, degree)
    knots_y = construct_knots(y, nseg_y, degree)
    
    B_x = bspline_basis(x, knots_x, degree)
    B_y = bspline_basis(y, knots_y, degree)
    
    n_basis_x = B_x.shape[1]
    n_basis_y = B_y.shape[1]
    
    # Tensor product basis
    n_obs = len(x)
    n_basis_total = n_basis_x * n_basis_y
    
    basis_2d = np.zeros((n_obs, n_basis_total))
    
    for i in range(n_obs):
        basis_2d[i, :] = np.kron(B_x[i, :], B_y[i, :])
    
    # Penalty matrices
    P_x = penalty_matrix(n_basis_x, penalty_order)
    P_y = penalty_matrix(n_basis_y, penalty_order)
    
    # 2D penalties
    I_x = sparse.eye(n_basis_x)
    I_y = sparse.eye(n_basis_y)
    
    # Penalty in x-direction: I_y ⊗ P_x
    P_2d_x = sparse.kron(I_y, P_x)
    
    # Penalty in y-direction: P_y ⊗ I_x  
    P_2d_y = sparse.kron(P_y, I_x)
    
    penalties = [P_2d_x, P_2d_y]
    
    return basis_2d, penalties


def construct_design_matrix(
    genotype: str,
    spatial_coords: Tuple[str, str],
    fixed_vars: List[str],
    random_vars: List[str],
    data: 'pd.DataFrame',
    genotype_as_random: bool = False
) -> dict:
    """
    Construct full design matrix for SpATS model.
    
    Parameters
    ----------
    genotype : str
        Genotype variable name
    spatial_coords : tuple of str
        Spatial coordinate variable names
    fixed_vars : list of str
        Fixed effect variable names
    random_vars : list of str
        Random effect variable names
    data : pd.DataFrame
        Input data
    genotype_as_random : bool, default=False
        Whether genotype is random
        
    Returns
    -------
    dict
        Dictionary containing design matrices and metadata
    """
    import pandas as pd
    
    n_obs = len(data)
    
    # Initialize lists for matrix blocks
    X_blocks = []  # Fixed effects
    Z_blocks = []  # Random effects
    penalties = []
    
    # Intercept
    X_blocks.append(np.ones((n_obs, 1)))
    
    # Genotype effects
    if genotype_as_random:
        # Random genotype effects
        geno_dummies = pd.get_dummies(data[genotype], drop_first=False)
        Z_blocks.append(geno_dummies.values)
        penalties.append(sparse.eye(geno_dummies.shape[1]))
    else:
        # Fixed genotype effects (drop first for identifiability)
        geno_dummies = pd.get_dummies(data[genotype], drop_first=True)
        if geno_dummies.shape[1] > 0:
            X_blocks.append(geno_dummies.values)
    
    # Other fixed effects
    for var in fixed_vars:
        if data[var].dtype in ['object', 'category']:
            dummies = pd.get_dummies(data[var], drop_first=True)
            if dummies.shape[1] > 0:
                X_blocks.append(dummies.values)
        else:
            X_blocks.append(data[var].values.reshape(-1, 1))
    
    # Spatial effects (always random with P-spline penalties)
    x_coord, y_coord = spatial_coords
    spatial_basis, spatial_penalties = construct_2d_pspline(
        data[x_coord].values,
        data[y_coord].values
    )
    Z_blocks.append(spatial_basis)
    penalties.extend(spatial_penalties)
    
    # Other random effects
    for var in random_vars:
        dummies = pd.get_dummies(data[var], drop_first=False)
        Z_blocks.append(dummies.values)
        penalties.append(sparse.eye(dummies.shape[1]))
    
    # Combine blocks
    X = np.hstack(X_blocks) if X_blocks else np.ones((n_obs, 1))
    Z = np.hstack(Z_blocks) if Z_blocks else np.zeros((n_obs, 0))
    
    return {
        'X': X,
        'Z': Z,
        'penalties': penalties,
        'n_fixed': X.shape[1],
        'n_random': Z.shape[1]
    }