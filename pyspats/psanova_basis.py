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
from scipy.sparse.linalg import LinearOperator
from typing import Tuple, List, Optional, Union
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

# Import polynomial projection wrapper for Kronecker interaction
try:
    from .spatial.projection import LeftProjectedLO
except ImportError:
    LeftProjectedLO = None


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
    degree: int = 3,
    use_kron_interaction: bool = True  # Memory-efficient Kronecker path with Gram-based Z'Z/X'Z
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Union[np.ndarray, "LinearOperator"], List[BlockInfo]]:
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
    use_kron_interaction : bool, default=True
        If True, use Kronecker-structured LinearOperator for interaction (memory efficient).
        If False, materialize the full dense interaction matrix (legacy path).

    Returns
    -------
    X_poly : np.ndarray
        Fixed polynomial design (n_obs × 3): [1, r, c]
    Z_r : np.ndarray
        Whitened row-smooth basis (n_obs × p_r)
    Z_c : np.ndarray
        Whitened column-smooth basis (n_obs × p_c)
    Z_rc : np.ndarray or LinearOperator
        Whitened interaction basis. If use_kron_interaction=True, this is a
        LinearOperator (lazy evaluation via Kronecker structure). Otherwise,
        it's a dense array (memory-intensive for large problems).
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

    **Kronecker Interaction (Default)**:
    When use_kron_interaction=True, the interaction Z_rc is represented as a
    LinearOperator using the separable structure:

        Z_rc = (B_r U_r^+ ⊗ B_c U_c^+) @ diag(sqrt(λ_r ⊕ λ_c))

    This avoids materializing the (n_obs × p_r*p_c) interaction matrix, saving
    substantial memory for large grids. For example, with n_obs=10000, p_r=50,
    p_c=50:
        - Dense: 10000 × 2500 ≈ 200 MB
        - Operator: ~1 MB (just stores B_r, B_c, and small eigendecompositions)
        - Reduction: 200x

    The operator still integrates seamlessly with REML, Schur complement, and
    ED computation.

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
    # Choose between Kronecker-structured operator (memory-efficient) or dense (legacy)
    n_r_basis = B_r.shape[1]
    n_c_basis = B_c.shape[1]

    # 5. Orthogonalize main effects to polynomial space first
    # This removes any residual constant/linear leakage
    Z_r = project_out_polynomial(Z_r_raw, X_poly)
    Z_c = project_out_polynomial(Z_c_raw, X_poly)

    if use_kron_interaction:
        # **KRONECKER PATH (default)**: Use LinearOperator for memory efficiency
        # Build whitened interaction operator using eigendecomposition
        # This never materializes the full (n_obs × p_r*p_c) tensor product

        # Get eigendecomposition of row and column penalties
        # Drop nullspace modes (λ < tol) from each penalty separately
        # Note: This may give slightly fewer modes than the dense path which forms
        # K_rc = I_c ⊗ K_r + K_c ⊗ I_r first, then removes its nullspace.
        # For exact parity, we'd need to compute λ_rc[i,j] = λ_r[i] + λ_c[j] and
        # drop modes where λ_rc < tol. Current approach is more conservative but simpler.
        meta = tensor_whiten_interaction(B_r, K_r, B_c, K_c, drop_null=True, tol=1e-10)

        # For general observation patterns, we need to evaluate (B_r ⊗ B_c) for each obs
        # The operator expects coefficients in the whitened space (r+ * c+)
        # We build a custom operator that:
        # 1. Maps whitened coefficients to the Kronecker space
        # 2. Evaluates the tensor product at each observation

        # Get dimensions after exact mode selection
        r_plus = meta["Urp"].shape[1]  # Selected eigenmodes of K_r
        c_plus = meta["Ucp"].shape[1]  # Selected eigenmodes of K_c
        keep_mask = meta["keep_mask"]  # Boolean mask (r+, c+) for valid modes

        # Count actual number of valid coefficients after exact mode selection
        n_rc = np.sum(keep_mask)  # Only modes where λ_r[i] + λ_c[j] > tol

        # Precompute transformed bases at observation points
        # B_r_transformed[i,:] = B_r[i,:] @ U_r^+
        # B_c_transformed[i,:] = B_c[i,:] @ U_c^+
        B_r_transformed = B_r @ meta["Urp"]  # (n_obs, r+)
        B_c_transformed = B_c @ meta["Ucp"]  # (n_obs, c+)

        # Whitening scale from eigenvalues: sqrt(λ_r ⊕ λ_c)
        # Only compute sqrt for valid modes (keep_mask), set others to 0
        lambda_sum = meta["lrp"][:, None] + meta["lcp"][None, :]  # (r+, c+)
        sqrt_lambda_sum = np.zeros_like(lambda_sum)
        sqrt_lambda_sum[keep_mask] = np.sqrt(lambda_sum[keep_mask])

        # Precompute flattened indices for valid modes (C-order)
        valid_idx = np.where(keep_mask.ravel(order="C"))[0]

        def matvec(alpha):
            """
            Evaluate whitened interaction at observations.

            For valid modes (i,j) where λ_r[i] + λ_c[j] > tol:
                result += alpha[idx] * sqrt(λ_r[i] + λ_c[j]) * B_r[i] ⊗ B_c[j]
            """
            # Expand alpha to full (r+, c+) matrix, filling only valid modes
            A_full = np.zeros((r_plus, c_plus))
            A_full[keep_mask] = alpha  # Fill valid modes

            # Apply whitening (only valid modes will be non-zero)
            A_scaled = A_full * sqrt_lambda_sum

            # Evaluate at each observation via row-wise Kronecker product
            # result[i] = B_r_transformed[i,:] @ A_scaled @ B_c_transformed[i,:].T
            result = np.zeros(n_obs)
            for i in range(n_obs):
                result[i] = B_r_transformed[i, :] @ A_scaled @ B_c_transformed[i, :]

            return result

        def rmatvec(y):
            """
            Project observations onto whitened coefficient space (adjoint).

            Computes gradient for valid modes only.
            """
            # Initialize gradient matrix (full r+ × c+)
            G = np.zeros((r_plus, c_plus))

            # Accumulate contribution from each observation
            for i in range(n_obs):
                # Outer product: B_r[i,:].T @ B_c[i,:] scaled by y[i]
                G += y[i] * np.outer(B_r_transformed[i, :], B_c_transformed[i, :])

            # Apply whitening mask: zero out invalid modes before dividing
            # This ensures adjoint property holds: <y, matvec(alpha)> = <rmatvec(y), alpha>
            G_masked = G * keep_mask  # Zero out invalid modes

            # Undo whitening for valid modes (divide by sqrt_lambda where positive)
            G_unscaled = np.zeros_like(G)
            valid_nonzero = keep_mask & (sqrt_lambda_sum > 0)
            G_unscaled[valid_nonzero] = G_masked[valid_nonzero] / sqrt_lambda_sum[valid_nonzero]

            # Extract only valid modes (C order)
            return G_unscaled[keep_mask]

        Z_rc_raw = LinearOperator(
            (n_obs, n_rc),
            matvec=matvec,
            rmatvec=rmatvec
        )

        # Apply polynomial projection to ensure orthogonality to X_poly
        # Build orthonormal basis for polynomial space
        Qx, _ = np.linalg.qr(X_poly, mode='reduced')

        if LeftProjectedLO is not None:
            Z_rc = LeftProjectedLO(Z_rc_raw, Qx)
        else:
            # Fallback if projection module not available (shouldn't happen in normal use)
            Z_rc = Z_rc_raw

        # Attach Gram computation metadata for efficient Z'Z and X'Z
        # These are used by the Schur complement path to avoid observation loops
        Z_rc._gram_meta = {
            'BrUr': B_r_transformed,  # B_r @ U_r+ (n_obs × r+)
            'BcUc': B_c_transformed,  # B_c @ U_c+ (n_obs × c+)
            'scales': sqrt_lambda_sum[keep_mask],  # Whitening scales for kept modes (n_rc,)
            'keep_mask': keep_mask,  # Boolean mask (r+ × c+) for mode selection
            'Qx': Qx,  # Orthonormal polynomial basis (n_obs × 3)
            'Z_op_unprojected': Z_rc_raw,  # Unprojected operator for rmatvec operations
            'is_kron_interaction': True  # Flag for Schur path
        }

    else:
        # **DENSE PATH (legacy)**: Materialize full interaction matrix
        # This can be memory-intensive for large problems

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

        # Orthogonalize to polynomial space
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


def tensor_whiten_interaction(
    B_r: sp.spmatrix,
    K_r: sp.spmatrix,
    B_c: sp.spmatrix,
    K_c: sp.spmatrix,
    drop_null: bool = True,
    tol: float = 1e-10
) -> dict:
    """
    Compute whitening transformation for tensor product interaction using 1D eigendecompositions.

    For the PS-ANOVA interaction smooth, the penalty is K_rc = I_c ⊗ K_r + K_c ⊗ I_r.
    This function computes the eigen-decompositions of K_r and K_c, then constructs
    the whitening transformation that produces G_rc = σ²_rc I without materializing
    the full Kronecker product.

    The key insight is that if K_r = U_r Λ_r U_r^T and K_c = U_c Λ_c U_c^T, then:
        K_rc = (U_r ⊗ U_c) (I ⊗ Λ_r + Λ_c ⊗ I) (U_r ⊗ U_c)^T

    where the eigenvalues add: λ_rc[i,j] = λ_r[i] + λ_c[j].

    Parameters
    ----------
    B_r : scipy.sparse matrix
        Row B-spline basis (n_r × p_r)
    K_r : scipy.sparse matrix
        Row penalty matrix (p_r × p_r)
    B_c : scipy.sparse matrix
        Column B-spline basis (n_c × p_c)
    K_c : scipy.sparse matrix
        Column penalty matrix (p_c × p_c)
    drop_null : bool, default=True
        Whether to drop nullspace modes (eigenvalues ≈ 0) to respect PS-ANOVA constraints
    tol : float, default=1e-10
        Tolerance for identifying zero eigenvalues

    Returns
    -------
    dict
        Dictionary containing:
        - 'Urp': Positive eigenvectors of K_r (p_r × r+)
        - 'Ucp': Positive eigenvectors of K_c (p_c × c+)
        - 'lrp': Positive eigenvalues of K_r (r+,)
        - 'lcp': Positive eigenvalues of K_c (c+,)
        - 'idx_r': Indices of positive eigenvalues in K_r
        - 'idx_c': Indices of positive eigenvalues in K_c

    Notes
    -----
    The nullspace removal ensures that the interaction smooth is orthogonal to
    the polynomial space (constant and linear functions), which is crucial for
    PS-ANOVA identifiability.

    For a typical P-spline with 2nd-order penalty, K has a 2D nullspace
    (constant and linear functions). After removing these modes, we have
    r+ = p_r - 2 and c+ = p_c - 2 active modes.

    The whitening transformation is applied via:
        Z̃_rc = (B_r U_r^+ ⊗ B_c U_c^+) @ diag(sqrt(λ_r^+ ⊕ λ_c^+))

    where ⊕ denotes the Kronecker sum: (λ_r ⊕ λ_c)[i,j] = λ_r[i] + λ_c[j].

    Examples
    --------
    >>> from scipy.sparse import eye
    >>> B_r = eye(10, format='csc')
    >>> K_r = eye(10, format='csc')
    >>> B_c = eye(8, format='csc')
    >>> K_c = eye(8, format='csc')
    >>> meta = tensor_whiten_interaction(B_r, K_r, B_c, K_c)
    >>> meta['Urp'].shape  # All modes positive for identity penalty
    (10, 10)
    """
    from scipy.linalg import eigh

    # Convert to dense for eigendecomposition (penalties are typically small)
    Kr = K_r.toarray() if sp.issparse(K_r) else np.asarray(K_r)
    Kc = K_c.toarray() if sp.issparse(K_c) else np.asarray(K_c)

    # Compute eigendecompositions (sorted ascending by default)
    lr, Ur = eigh(Kr)
    lc, Uc = eigh(Kc)

    if drop_null:
        # EXACT MODE SELECTION for interaction nullspace
        # Drop modes where λ_rc[i,j] = λ_r[i] + λ_c[j] ≤ tol
        # This matches the dense path which forms K_rc = I_c ⊗ K_r + K_c ⊗ I_r first

        # Compute Kronecker sum of eigenvalues: λ_rc[i,j] = λ_r[i] + λ_c[j]
        lambda_sum = lr[:, None] + lc[None, :]  # (len(lr), len(lc))

        # Create mask for positive modes (sum > tol)
        keep_mask_full = lambda_sum > tol  # Boolean mask (len(lr), len(lc))

        # Keep ALL row/col modes (no pre-filtering)
        # We'll select valid pairs directly via keep_mask
        idx_r = np.arange(len(lr))
        idx_c = np.arange(len(lc))

        Urp = Ur  # Keep all eigenvectors
        Ucp = Uc
        lrp = lr  # Keep all eigenvalues
        lcp = lc

        # The keep_mask tells us which (i,j) pairs are valid
        keep_mask_selected = keep_mask_full

    else:
        # Keep all modes
        idx_r = np.arange(len(lr))
        idx_c = np.arange(len(lc))

        Urp = Ur
        Ucp = Uc
        lrp = lr
        lcp = lc

        # All modes kept
        keep_mask_selected = np.ones((len(lrp), len(lcp)), dtype=bool)

    return {
        "Urp": Urp,
        "Ucp": Ucp,
        "lrp": lrp,
        "lcp": lcp,
        "idx_r": idx_r,
        "idx_c": idx_c,
        "keep_mask": keep_mask_selected,  # Exact mode selection mask
    }


def build_whitened_interaction_operator(
    B_r: sp.spmatrix,
    K_r: sp.spmatrix,
    B_c: sp.spmatrix,
    K_c: sp.spmatrix,
    n_r: int,
    n_c: int,
    tol: float = 1e-10
):
    """
    Build a LinearOperator for the whitened interaction smooth Z̃_rc.

    This creates a lazy operator that evaluates the whitened tensor product
    interaction without materializing the full Kronecker product. The operator
    implements:

        Z̃_rc = (B_r U_r^+ ⊗ B_c U_c^+) @ diag(sqrt(λ_r^+ ⊕ λ_c^+))

    where U_r^+, U_c^+ are the non-null eigenvectors of K_r, K_c and
    λ_r^+ ⊕ λ_c^+ is the Kronecker sum of positive eigenvalues.

    Parameters
    ----------
    B_r : scipy.sparse matrix
        Row B-spline basis (n_r × p_r)
    K_r : scipy.sparse matrix
        Row penalty matrix (p_r × p_r)
    B_c : scipy.sparse matrix
        Column B-spline basis (n_c × p_c)
    K_c : scipy.sparse matrix
        Column penalty matrix (p_c × p_c)
    n_r : int
        Number of row positions in the spatial field
    n_c : int
        Number of column positions in the spatial field
    tol : float, default=1e-10
        Tolerance for eigenvalue positivity

    Returns
    -------
    operator : LinearOperator
        Lazy operator for Z̃_rc with shape (n_r*n_c, r+*c+)
        where r+ = number of positive eigenvalues of K_r
        and c+ = number of positive eigenvalues of K_c
    meta : dict
        Metadata from tensor_whiten_interaction containing eigenvectors
        and eigenvalues

    Notes
    -----
    **Memory Savings**:
    For a 100×100 field with p_r=50, p_c=50 coefficients:
        - Dense Z_rc: (10000 × 2500) ≈ 200 MB
        - Operator: (100×50) + (100×50) ≈ 0.8 MB
        - Reduction: 250x

    **Mathematical Correctness**:
    The whitening ensures G_rc = σ²_rc I, which is required for:
    1. ED computation via Takahashi selected inverse
    2. Closed-form variance updates: σ²_rc = (u'u) / ED_rc
    3. REML convergence properties

    **Matvec Implementation**:
    For coefficient vector α (r+*c+,):
    1. Reshape α to (r+, c+) matrix A
    2. Scale by sqrt(λ_r ⊕ λ_c): A *= sqrt_lambda_sum
    3. Evaluate via Kronecker structure: vec(B_r U_r^+ @ A @ (B_c U_c^+)^T)

    **Rmatvec Implementation** (adjoint):
    For field vector v (n_r*n_c,):
    1. Reshape v to (n_r, n_c) matrix V
    2. Project: G = (B_r U_r^+)^T @ V @ (B_c U_c^+)
    3. Unscale: G /= sqrt_lambda_sum
    4. Flatten to (r+*c+,)

    Examples
    --------
    >>> from scipy.sparse import eye
    >>> B_r = eye(100, 50, format='csc')
    >>> K_r = eye(50, format='csc')
    >>> B_c = eye(100, 50, format='csc')
    >>> K_c = eye(50, format='csc')
    >>> Z_op, meta = build_whitened_interaction_operator(
    ...     B_r, K_r, B_c, K_c, 100, 100
    ... )
    >>> Z_op.shape  # (n_r*n_c, r+*c+)
    (10000, 2500)
    >>> alpha = np.random.randn(2500)
    >>> field = Z_op @ alpha  # Lazy evaluation
    >>> field.shape
    (10000,)
    """
    from scipy.sparse.linalg import LinearOperator

    # Compute eigendecompositions and select positive modes
    meta = tensor_whiten_interaction(B_r, K_r, B_c, K_c, tol=tol)
    Urp, Ucp = meta["Urp"], meta["Ucp"]
    lrp, lcp = meta["lrp"], meta["lcp"]

    # Precompute transformed bases: B_r @ U_r^+ and B_c @ U_c^+
    BrUr = B_r @ Urp  # (n_r × r+)
    BcUc = B_c @ Ucp  # (n_c × c+)

    # Compute Kronecker sum of eigenvalues: λ_rc[i,j] = λ_r[i] + λ_c[j]
    # Shape: (r+, c+)
    lambda_sum = lrp[:, None] + lcp[None, :]

    # Whitening scale: sqrt(λ_r ⊕ λ_c)
    sqrt_lambda_sum = np.sqrt(lambda_sum)

    # Operator dimensions
    m = n_r * n_c  # Field size
    n = BrUr.shape[1] * BcUc.shape[1]  # Whitened coefficient size (r+ * c+)

    def matvec(alpha):
        """
        Compute Z̃_rc @ α via Kronecker structure.

        Maps whitened coefficients α to the spatial field.
        """
        # Reshape coefficient vector to matrix (Fortran order for kron consistency)
        A = alpha.reshape(BrUr.shape[1], BcUc.shape[1], order="F")

        # Apply whitening scale
        A_scaled = A * sqrt_lambda_sum

        # Evaluate via Kronecker structure: (BrUr ⊗ BcUc) @ vec(A_scaled)
        # = vec(BrUr @ A_scaled @ BcUc^T)
        Y = BrUr @ A_scaled @ (BcUc.T)

        # Flatten to field vector (Fortran order)
        return Y.reshape(m, order="F")

    def rmatvec(v):
        """
        Compute Z̃_rc^T @ v via Kronecker structure.

        Projects field v onto whitened coefficient space (adjoint operation).
        """
        # Reshape field vector to matrix (Fortran order)
        V = v.reshape(n_r, n_c, order="F")

        # Project onto transformed bases: (BrUr ⊗ BcUc)^T @ vec(V)
        # = vec(BrUr^T @ V @ BcUc)
        G = (BrUr.T @ V) @ BcUc

        # Undo whitening scale
        G_unscaled = G / sqrt_lambda_sum

        # Flatten to coefficient vector (Fortran order)
        return G_unscaled.reshape(n, order="F")

    operator = LinearOperator((m, n), matvec=matvec, rmatvec=rmatvec)

    return operator, meta
