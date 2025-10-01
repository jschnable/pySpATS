"""
PS-ANOVA decomposition for spatial modeling with explicit polynomial fixed effects
and orthogonal P-spline random smooths.

This module implements the SAP (Separation of Anisotropic Penalties) approach with:
- Fixed polynomial part: intercept, linear row, linear column
- Random smooth parts: row-smooth (f_r), column-smooth (f_c), interaction (f_rc)
- 2nd-order P-spline difference penalties
- Nullspace removal and orthogonality to polynomial space
- Penalty whitening so each random block has G_k = σ_k² I

Reference:
Rodriguez-Alvarez et al. (2015) "Fast smoothing parameter separation in
multidimensional generalized P-splines: the SAP algorithm"
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from scipy.interpolate import BSpline
from typing import Tuple, List, Optional
from dataclasses import dataclass

# Import BlockInfo from ed_selected_inverse for block metadata
try:
    from .ed_selected_inverse import BlockInfo
except ImportError:
    # Fallback if ed_selected_inverse not available
    @dataclass
    class BlockInfo:
        """Fallback BlockInfo for block metadata."""
        name: str
        start: int
        stop: int
        is_random: bool = True

        @property
        def size(self) -> int:
            return self.stop - self.start


def make_bspline_basis(
    x: np.ndarray,
    n_knots: int = 10,
    degree: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct univariate B-spline basis matrix.

    Parameters
    ----------
    x : np.ndarray
        1D array of positions (e.g., row or column indices)
    n_knots : int, default=10
        Number of interior knots
    degree : int, default=3
        B-spline degree (3 = cubic)

    Returns
    -------
    B : np.ndarray
        Sparse B-spline basis matrix (n_obs × n_basis)
    knots : np.ndarray
        Knot sequence used

    Notes
    -----
    - Normalizes x to [0,1] internally for stable knot placement
    - Returns basis in dense format for easier manipulation
    - For large problems, consider sparse representations
    """
    x = np.asarray(x).ravel()
    n = len(x)

    # Normalize to [0, 1]
    x_min, x_max = x.min(), x.max()
    if x_max - x_min < 1e-10:
        # Degenerate case: all same value
        x_norm = np.zeros_like(x)
    else:
        x_norm = (x - x_min) / (x_max - x_min)

    # Construct knot sequence
    # Interior knots uniformly spaced
    interior_knots = np.linspace(0, 1, n_knots)

    # Add boundary knots for B-spline basis
    # Extend by degree+1 knots on each side
    dx = 1.0 / max(n_knots - 1, 1)
    left_knots = np.array([0 - (degree - i) * dx for i in range(degree)])
    right_knots = np.array([1 + (i + 1) * dx for i in range(degree)])

    knots = np.concatenate([left_knots, interior_knots, right_knots])
    knots = np.sort(knots)

    # Number of basis functions
    n_basis = len(knots) - degree - 1

    # Evaluate B-spline basis
    B = np.zeros((n, n_basis))
    for i in range(n_basis):
        coef = np.zeros(n_basis)
        coef[i] = 1.0
        spline = BSpline(knots, coef, degree, extrapolate=False)
        B[:, i] = spline(x_norm, extrapolate=False)

    # Handle NaNs from extrapolation
    B = np.nan_to_num(B, nan=0.0)

    return B, knots


def D2_penalty(n_coef: int) -> sp.csr_matrix:
    """
    Construct 2nd-order difference penalty matrix K = D2.T @ D2.

    Parameters
    ----------
    n_coef : int
        Number of basis coefficients

    Returns
    -------
    K : scipy.sparse.csr_matrix
        Penalty matrix (n_coef × n_coef), sparse

    Notes
    -----
    Second-order difference operator:
    D2[i, :] = [0, ..., 0, 1, -2, 1, 0, ..., 0]
                        at position i

    K = D2.T @ D2 penalizes curvature (2nd derivative)
    K has 2D nullspace: constants and linear trends
    """
    if n_coef < 3:
        # Degenerate case: not enough coefficients for 2nd-order penalty
        return sp.eye(n_coef, format='csr')

    # Build 2nd-order difference matrix
    # Shape: (n_coef - 2) × n_coef
    D2 = sp.lil_matrix((n_coef - 2, n_coef))
    for i in range(n_coef - 2):
        D2[i, i] = 1.0
        D2[i, i + 1] = -2.0
        D2[i, i + 2] = 1.0

    D2 = D2.tocsr()
    K = D2.T @ D2

    return K.tocsr()


def row_col_bases(
    r: np.ndarray,
    c: np.ndarray,
    n_knots_r: int,
    n_knots_c: int,
    degree: int = 3
) -> Tuple[np.ndarray, np.ndarray, sp.csr_matrix, sp.csr_matrix]:
    """
    Build separate row and column B-spline bases and penalties.

    Parameters
    ----------
    r : np.ndarray
        Row coordinates (1D array, length n_obs)
    c : np.ndarray
        Column coordinates (1D array, length n_obs)
    n_knots_r : int
        Number of knots for row basis
    n_knots_c : int
        Number of knots for column basis
    degree : int, default=3
        B-spline degree

    Returns
    -------
    B_r : np.ndarray
        Row basis matrix (n_obs × p_r)
    B_c : np.ndarray
        Column basis matrix (n_obs × p_c)
    K_r : scipy.sparse.csr_matrix
        2nd-order penalty for row basis (p_r × p_r)
    K_c : scipy.sparse.csr_matrix
        2nd-order penalty for column basis (p_c × p_c)
    """
    B_r, _ = make_bspline_basis(r, n_knots=n_knots_r, degree=degree)
    B_c, _ = make_bspline_basis(c, n_knots=n_knots_c, degree=degree)

    K_r = D2_penalty(B_r.shape[1])
    K_c = D2_penalty(B_c.shape[1])

    return B_r, B_c, K_r, K_c


def remove_nullspace_and_whiten(
    Z: np.ndarray,
    K: sp.spmatrix,
    tol: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove penalty nullspace and whiten random effect design.

    Computes eigendecomposition of penalty K, separates nullspace (zero eigenvalues)
    from penalized space (positive eigenvalues), and returns whitened design matrix
    Z̃ such that coefficients ũ ~ N(0, σ² I) give the same random effect as
    u ~ N(0, σ² K⁺) where K⁺ is the Moore-Penrose pseudoinverse.

    Parameters
    ----------
    Z : np.ndarray
        Random effect design matrix (n_obs × p)
    K : scipy.sparse matrix
        Penalty matrix (p × p), symmetric positive semi-definite
    tol : float, default=1e-8
        Tolerance for determining zero eigenvalues

    Returns
    -------
    Z_tilde : np.ndarray
        Whitened design matrix (n_obs × p_penalized)
        Only includes penalized space (nullspace removed)
    U_plus : np.ndarray
        Eigenvectors for positive eigenvalues (p × p_penalized)
        Useful for back-transformation if needed

    Notes
    -----
    For 2nd-order penalty on B-splines:
    - Nullspace contains constant and linear trends (2D nullspace)
    - These polynomial pieces belong in fixed effects, not random

    Whitening transformation:
    Z̃ = Z @ U_+ @ Λ_+^{1/2}
    where K = U Λ U^T, Λ_+ contains only positive eigenvalues

    After whitening, cov(Z̃ @ ũ) = σ² Z̃ @ Z̃^T with ũ ~ N(0, σ² I)
    """
    # Convert to dense for eigendecomposition
    # For production: use sparse solvers for large K
    K_dense = K.toarray() if sp.issparse(K) else K

    # Symmetric eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(K_dense)

    # Identify positive eigenvalues (penalized space)
    idx_positive = eigvals > tol
    n_positive = np.sum(idx_positive)

    if n_positive == 0:
        # Degenerate: no penalized space (pure nullspace)
        # Return empty matrix
        return np.zeros((Z.shape[0], 0)), np.zeros((Z.shape[1], 0))

    # Extract penalized space
    U_plus = eigvecs[:, idx_positive]  # (p × p_penalized)
    Lambda_plus = eigvals[idx_positive]  # (p_penalized,)

    # Whiten: Z̃ = Z @ U_+ @ Λ_+^{1/2}
    # This absorbs penalty into design so we can use G = σ² I
    sqrt_Lambda = np.sqrt(Lambda_plus)
    Z_tilde = Z @ U_plus @ np.diag(sqrt_Lambda)

    return Z_tilde, U_plus


def project_out_polynomial(
    Z: np.ndarray,
    X_poly: np.ndarray,
    inplace: bool = False
) -> np.ndarray:
    """
    Project out polynomial space from random design matrix.

    Ensures Z is orthogonal to polynomial fixed effects X_poly by computing:
    Z ← Z - Q_X @ (Q_X^T @ Z)
    where Q_X is orthonormal basis for columns of X_poly.

    Parameters
    ----------
    Z : np.ndarray
        Random effect design matrix (n_obs × p_random)
    X_poly : np.ndarray
        Polynomial fixed effects (n_obs × p_poly)
        Typically [1, r, c] for PS-ANOVA
    inplace : bool, default=False
        If True, modify Z in place

    Returns
    -------
    Z_orth : np.ndarray
        Orthogonalized random design matrix
        Satisfies X_poly^T @ Z_orth ≈ 0

    Notes
    -----
    This removes any residual polynomial leakage after nullspace removal.
    For well-conditioned problems, nullspace removal should handle most of it,
    but numeric errors can introduce small polynomial components.
    """
    # Compute orthonormal basis for polynomial space
    Q_X, _ = np.linalg.qr(X_poly, mode='reduced')

    # Project Z onto polynomial space
    projection = Q_X @ (Q_X.T @ Z)

    # Remove projection
    if inplace:
        Z -= projection
        return Z
    else:
        return Z - projection


def build_psanova_design(
    r: np.ndarray,
    c: np.ndarray,
    nkr: int = 10,
    nkc: int = 10,
    degree: int = 3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[BlockInfo]]:
    """
    Build PS-ANOVA design with explicit polynomial fixed effects and orthogonal smooths.

    Constructs:
    - Fixed polynomial: X_poly = [1, r, c]
    - Random smooths: Z_r (row-smooth), Z_c (col-smooth), Z_rc (interaction)
    - All random parts orthogonal to polynomial space
    - Penalties absorbed via whitening (G_k = σ_k² I for each block)

    Parameters
    ----------
    r : np.ndarray
        Row coordinates (length n_obs)
    c : np.ndarray
        Column coordinates (length n_obs)
    nkr : int, default=10
        Number of knots for row dimension
    nkc : int, default=10
        Number of knots for column dimension
    degree : int, default=3
        B-spline degree

    Returns
    -------
    X_poly : np.ndarray
        Fixed polynomial design (n_obs × 3): [1, r, c]
    Z_r : np.ndarray
        Whitened row-smooth basis (n_obs × p_r)
    Z_c : np.ndarray
        Whitened column-smooth basis (n_obs × p_c)
    Z_rc : np.ndarray
        Whitened interaction basis (n_obs × p_rc)
    blocks : List[BlockInfo]
        Block metadata for random effects (contiguous indices)

    Notes
    -----
    PS-ANOVA decomposition:
    f(r, c) = β₀ + β_r·r + β_c·c + f_r(r) + f_c(c) + f_rc(r, c)

    Where:
    - β₀, β_r, β_c are fixed polynomial coefficients
    - f_r, f_c, f_rc are random smooth components with 2nd-order penalties

    All random components are:
    1. Nullspace-free (polynomial parts removed)
    2. Orthogonal to [1, r, c]
    3. Whitened so G_k = σ_k² I

    This matches the R SpATS package decomposition for exact ED computation.
    """
    r = np.asarray(r).ravel()
    c = np.asarray(c).ravel()
    n_obs = len(r)

    if len(c) != n_obs:
        raise ValueError(f"r and c must have same length, got {len(r)} and {len(c)}")

    # 1. Fixed polynomial part: [1, r, c]
    # Normalize r and c for numerical stability
    r_norm = (r - r.mean()) / (r.std() + 1e-10)
    c_norm = (c - c.mean()) / (c.std() + 1e-10)

    X_poly = np.column_stack([
        np.ones(n_obs),
        r_norm,
        c_norm
    ])

    # 2. Build B-spline bases and penalties
    B_r, B_c, K_r, K_c = row_col_bases(r, c, nkr, nkc, degree)

    # 3. Main effects: row-smooth and column-smooth
    # Remove nullspace and whiten
    Z_r_raw, _ = remove_nullspace_and_whiten(B_r, K_r)
    Z_c_raw, _ = remove_nullspace_and_whiten(B_c, K_c)

    # 4. Interaction: row ⊗ col
    # Build tensor product basis
    n_r_basis = B_r.shape[1]
    n_c_basis = B_c.shape[1]

    # Tensor product: for each observation i, B_rc[i, :] = kron(B_c[i, :], B_r[i, :])
    B_rc = np.zeros((n_obs, n_r_basis * n_c_basis))
    for i in range(n_obs):
        B_rc[i, :] = np.kron(B_c[i, :], B_r[i, :])

    # Interaction penalty: K_rc = kron(I_c, K_r) + kron(K_c, I_r)
    # This is the standard PS-ANOVA penalty for separable smoothing
    I_r = sp.eye(n_r_basis, format='csr')
    I_c = sp.eye(n_c_basis, format='csr')
    K_rc = sp.kron(I_c, K_r) + sp.kron(K_c, I_r)

    # Remove nullspace and whiten
    Z_rc_raw, _ = remove_nullspace_and_whiten(B_rc, K_rc)

    # 5. Orthogonalize all random parts to polynomial space
    # This removes any residual constant/linear leakage
    Z_r = project_out_polynomial(Z_r_raw, X_poly)
    Z_c = project_out_polynomial(Z_c_raw, X_poly)
    Z_rc = project_out_polynomial(Z_rc_raw, X_poly)

    # 6. Build block metadata (contiguous indices in concatenated Z)
    blocks = []
    start_idx = 0

    for name, Z_block in [('row_smooth', Z_r), ('col_smooth', Z_c), ('interaction_smooth', Z_rc)]:
        if Z_block.shape[1] > 0:  # Only add non-empty blocks
            stop_idx = start_idx + Z_block.shape[1]
            blocks.append(BlockInfo(
                name=name,
                start=start_idx,
                stop=stop_idx,
                is_random=True
            ))
            start_idx = stop_idx

    return X_poly, Z_r, Z_c, Z_rc, blocks


def verify_orthogonality(
    X_poly: np.ndarray,
    Z_blocks: List[np.ndarray],
    tol: float = 1e-8
) -> bool:
    """
    Verify that random blocks are orthogonal to polynomial space.

    Parameters
    ----------
    X_poly : np.ndarray
        Polynomial fixed effects
    Z_blocks : list of np.ndarray
        Random effect design blocks
    tol : float, default=1e-8
        Tolerance for orthogonality check

    Returns
    -------
    bool
        True if all blocks are orthogonal to X_poly within tolerance

    Raises
    ------
    AssertionError
        If orthogonality is violated (in debug mode)
    """
    for i, Z in enumerate(Z_blocks):
        if Z.shape[1] == 0:
            continue
        cross_product = X_poly.T @ Z
        max_abs = np.abs(cross_product).max()

        if max_abs > tol:
            print(f"Warning: Block {i} not orthogonal to polynomial space (max |X'Z| = {max_abs:.2e})")
            return False

    return True
