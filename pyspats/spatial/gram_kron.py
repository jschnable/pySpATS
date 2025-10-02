"""
Efficient Gram-matrix-based formulas for Kronecker interaction Z'Z and X'Z.

This module provides algebraic computation of normal equations terms for the
whitened, left-projected Kronecker interaction without expensive observation loops.

Key insight: For Z_rc = (B_r U_r+ ⊗ B_c U_c+) @ diag(scales), we have:
    Z'Z = (G_r ⊗ G_c) with column/row scaling by scales
where G_r = (B_r U_r+)' @ (B_r U_r+) and G_c similarly are small dense Gram matrices.

For left-projection P_⊥ = I - Qx Qx', we apply a cheap correction:
    Z_⊥' Z_⊥ = Z'Z - (Z' Qx)(Z' Qx)'

This avoids materializing the full interaction matrix or looping over observations.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from typing import Tuple, Union


def compute_GrGc(
    BrUr: Union[np.ndarray, sp.spmatrix],
    BcUc: Union[np.ndarray, sp.spmatrix]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute small dense Gram matrices for row and column bases.

    Parameters
    ----------
    BrUr : array or sparse matrix
        Row basis after eigenspace projection: B_r @ U_r+ (n_obs × r+)
    BcUc : array or sparse matrix
        Column basis after eigenspace projection: B_c @ U_c+ (n_obs × c+)

    Returns
    -------
    Gr : np.ndarray
        Row Gram matrix: (B_r U_r+)' @ (B_r U_r+), shape (r+, r+)
    Gc : np.ndarray
        Column Gram matrix: (B_c U_c+)' @ (B_c U_c+), shape (c+, c+)

    Notes
    -----
    These small dense matrices (typically 10-50 × 10-50) are the building blocks
    for efficient Z'Z computation via Kronecker algebra.

    Examples
    --------
    >>> BrUr = np.random.randn(100, 8)
    >>> BcUc = np.random.randn(100, 8)
    >>> Gr, Gc = compute_GrGc(BrUr, BcUc)
    >>> Gr.shape
    (8, 8)
    """
    # Convert to dense if sparse (Gram matrices are small)
    BrUr = BrUr.toarray() if sp.issparse(BrUr) else np.asarray(BrUr)
    BcUc = BcUc.toarray() if sp.issparse(BcUc) else np.asarray(BcUc)

    # Compute Gram matrices (symmetric positive semi-definite)
    Gr = BrUr.T @ BrUr  # (r+, r+)
    Gc = BcUc.T @ BcUc  # (c+, c+)

    return Gr, Gc


def kron_whitened_ZtZ(
    Gr: np.ndarray,
    Gc: np.ndarray,
    scales: np.ndarray
) -> np.ndarray:
    """
    Build Z'Z for whitened Kronecker interaction using Gram matrices.

    Computes: Z'Z = diag(scales) @ kron(Gr, Gc) @ diag(scales)

    Parameters
    ----------
    Gr : np.ndarray
        Row Gram matrix (r+ × r+)
    Gc : np.ndarray
        Column Gram matrix (c+ × c+)
    scales : np.ndarray
        Whitening scales: sqrt(λ_r[i] + λ_c[j]) for kept modes, flattened (p_rc,)

    Returns
    -------
    ZtZ : np.ndarray
        Z'Z matrix (p_rc × p_rc), dense but small due to mode selection

    Notes
    -----
    The Kronecker product kron(Gr, Gc) is computed directly, then scaled.
    For typical sizes (r+ ~ 10, c+ ~ 10), this gives ~100×100 dense matrix,
    which is much smaller than materializing Z itself (n_obs × p_rc).

    Memory: O(r+ * c+)² vs O(n_obs * r+ * c+) for explicit Z

    Examples
    --------
    >>> Gr = np.eye(8)
    >>> Gc = np.eye(8)
    >>> scales = np.ones(64)
    >>> ZtZ = kron_whitened_ZtZ(Gr, Gc, scales)
    >>> ZtZ.shape
    (64, 64)
    """
    # Kronecker product of Gram matrices (dense, but small)
    K = np.kron(Gr, Gc)  # (r+ * c+, r+ * c+)

    # Apply diagonal scaling: diag(scales) @ K @ diag(scales)
    s = scales.ravel()
    K = (s[:, None] * K) * s[None, :]

    return K


def projected_ZtZ(
    ZtZ_raw: np.ndarray,
    Z_op: LinearOperator,
    Qx: np.ndarray,
    symmetrize: bool = True,
    ensure_psd: bool = True,
    eps: float = 1e-12
) -> np.ndarray:
    """
    Apply left-projection correction to Z'Z with numerical safeguards.

    Computes: Z_⊥' Z_⊥ = Z'Z - (Z' Qx)(Z' Qx)'

    where Z_⊥ = P_⊥ Z = (I - Qx Qx') Z is the left-projected operator.

    Parameters
    ----------
    ZtZ_raw : np.ndarray
        Unprojected Z'Z from kron_whitened_ZtZ() (p_rc × p_rc)
    Z_op : LinearOperator
        Unprojected whitened interaction operator (before LeftProjectedLO wrapper)
    Qx : np.ndarray
        Orthonormal basis for polynomial space (n_obs × q), typically q=3 for [1, r, c]
    symmetrize : bool, default=True
        If True, enforce symmetry to eliminate numerical asymmetry
    ensure_psd : bool, default=True
        If True, add minimal jitter to ensure positive semi-definiteness
    eps : float, default=1e-12
        Minimum eigenvalue tolerance for PSD enforcement

    Returns
    -------
    ZtZ_proj : np.ndarray
        Projected Z_⊥' Z_⊥ (p_rc × p_rc), guaranteed symmetric PSD

    Notes
    -----
    Uses the identity P_⊥' P_⊥ = P_⊥ for orthogonal projectors:
        (P_⊥ Z)' (P_⊥ Z) = Z' P_⊥' P_⊥ Z = Z' P_⊥ Z = Z'Z - Z' Qx Qx' Z

    Only requires q rmatvec operations (typically q=3), not n_obs.

    Numerical safeguards:
    - Symmetrization eliminates round-off asymmetry
    - PSD enforcement adds minimal jitter if negative eigenvalues detected

    Examples
    --------
    >>> ZtZ_raw = np.eye(64)
    >>> Z_op = LinearOperator((100, 64), matvec=..., rmatvec=...)
    >>> Qx = np.random.randn(100, 3); Qx, _ = np.linalg.qr(Qx)
    >>> ZtZ_proj = projected_ZtZ(ZtZ_raw, Z_op, Qx)
    >>> ZtZ_proj.shape
    (64, 64)
    >>> # Check PSD
    >>> w = np.linalg.eigvalsh(ZtZ_proj)
    >>> assert w.min() >= -1e-10
    """
    # Re-orthogonalize Qx to ensure numerical orthonormality
    Qx, _ = np.linalg.qr(Qx, mode='reduced')

    # Compute D = Z' Qx via rmatvec (p_rc × q)
    # Each column: Z' @ Qx[:, j]
    # CRITICAL: Use unprojected Z_op here
    q = Qx.shape[1]
    D_cols = [Z_op.rmatvec(Qx[:, j]) for j in range(q)]
    D = np.column_stack(D_cols)  # (p_rc, q)

    # Apply correction: Z'Z - D @ D'
    correction = D @ D.T  # (p_rc, p_rc)
    ZtZ_proj = ZtZ_raw - correction

    # Symmetrize to eliminate numerical asymmetry
    if symmetrize:
        ZtZ_proj = 0.5 * (ZtZ_proj + ZtZ_proj.T)

    # Ensure PSD by adding minimal jitter if needed
    if ensure_psd:
        w = np.linalg.eigvalsh(ZtZ_proj)
        if w.min() < 0:
            # Add jitter to make smallest eigenvalue = eps
            jitter = (-w.min() + eps)
            ZtZ_proj += jitter * np.eye(ZtZ_proj.shape[0])

    return ZtZ_proj


def XtZ_poly_zero(n_poly: int, p_rc: int) -> np.ndarray:
    """
    Return X_poly' Z_⊥ = 0 by construction.

    Parameters
    ----------
    n_poly : int
        Number of polynomial fixed effects (typically 3 for [1, r, c])
    p_rc : int
        Number of interaction coefficients

    Returns
    -------
    XtZ : np.ndarray
        Zero matrix (n_poly × p_rc)

    Notes
    -----
    By PS-ANOVA construction, the interaction smooth is orthogonal to the
    polynomial space [1, r, c], so X_poly' Z_⊥ = 0 exactly.

    This avoids unnecessary computation and ensures exact zeros in the
    normal equations.

    Examples
    --------
    >>> XtZ = XtZ_poly_zero(3, 64)
    >>> XtZ.shape
    (3, 64)
    >>> np.max(np.abs(XtZ))
    0.0
    """
    return np.zeros((n_poly, p_rc))


def XtZ_other(
    Z_op: LinearOperator,
    X_other: Union[np.ndarray, sp.spmatrix],
    Qx: np.ndarray,
    chunk_size: int = 64
) -> np.ndarray:
    """
    Compute X_other' Z_⊥ for non-polynomial fixed effects.

    Uses: X_other' Z_⊥ = X_other' Z - (X_other' Qx)(Qx' Z)

    Parameters
    ----------
    Z_op : LinearOperator
        Unprojected whitened interaction operator
    X_other : array or sparse matrix
        Non-polynomial fixed effects (n_obs × p_other)
    Qx : np.ndarray
        Orthonormal polynomial basis (n_obs × q)
    chunk_size : int, default=64
        Number of columns to process at once for memory efficiency

    Returns
    -------
    XtZ : np.ndarray
        X_other' Z_⊥ matrix (p_other × p_rc)

    Notes
    -----
    Computes X' Z via batched rmatvec operations, then applies the
    polynomial projection correction.

    For typical cases with only polynomial fixed effects, X_other is empty
    and this function is not called.

    Examples
    --------
    >>> Z_op = LinearOperator((100, 64), matvec=..., rmatvec=...)
    >>> X_other = np.random.randn(100, 5)
    >>> Qx = np.random.randn(100, 3); Qx, _ = np.linalg.qr(Qx)
    >>> XtZ = XtZ_other(Z_op, X_other, Qx)
    >>> XtZ.shape
    (5, 64)
    """
    # Convert to dense if sparse
    Xo = X_other.toarray() if sp.issparse(X_other) else np.asarray(X_other)
    p_other = Xo.shape[1]

    if p_other == 0:
        # No other fixed effects
        return np.zeros((0, Z_op.shape[1]))

    # Compute Z' Qx once (p_rc × q)
    q = Qx.shape[1]
    D_cols = [Z_op.rmatvec(Qx[:, j]) for j in range(q)]
    ZtQ = np.column_stack(D_cols)  # (p_rc, q)

    # Compute X_other' Z via batched rmatvec
    # Process columns in chunks to limit memory usage
    chunks = []
    for i in range(0, p_other, chunk_size):
        Xi = Xo[:, i:i+chunk_size]  # (n_obs × chunk)
        n_cols = Xi.shape[1]

        # Z' @ Xi via rmatvec for each column
        Zi_cols = [Z_op.rmatvec(Xi[:, j]) for j in range(n_cols)]
        Zi = np.column_stack(Zi_cols).T  # (chunk × p_rc)

        chunks.append(Zi)

    XtZ = np.vstack(chunks)  # (p_other × p_rc)

    # Apply projection correction: subtract (X_other' Qx)(Qx' Z)
    XoTQ = Xo.T @ Qx  # (p_other × q)
    correction = XoTQ @ ZtQ.T  # (p_other × p_rc)

    return XtZ - correction


def build_interaction_gram_terms(
    BrUr: Union[np.ndarray, sp.spmatrix],
    BcUc: Union[np.ndarray, sp.spmatrix],
    scales: np.ndarray,
    Z_op_unprojected: LinearOperator,
    Qx: np.ndarray,
    X: Union[np.ndarray, sp.spmatrix],
    n_poly: int = 3,
    keep_mask: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build Z'Z and X'Z for projected Kronecker interaction using Gram matrices.

    This is the main entry point that combines all the algebraic formulas.

    Parameters
    ----------
    BrUr : array or sparse
        Row basis after eigenspace projection (n_obs × r+)
    BcUc : array or sparse
        Column basis after eigenspace projection (n_obs × c+)
    scales : np.ndarray
        Whitening scales for kept modes (p_rc,)
    Z_op_unprojected : LinearOperator
        Unprojected whitened interaction operator
    Qx : np.ndarray
        Orthonormal polynomial basis (n_obs × q)
    X : array or sparse
        Full fixed effects matrix (n_obs × p)
    n_poly : int, default=3
        Number of polynomial columns at start of X
    keep_mask : np.ndarray, optional
        Boolean mask (r+ × c+) indicating which modes are kept.
        If provided, only these modes are included in the Gram computation.

    Returns
    -------
    ZtZ : np.ndarray
        Projected Z_⊥' Z_⊥ (p_rc × p_rc)
    XtZ : np.ndarray
        X' Z_⊥ (p × p_rc), with polynomial block exactly zero

    Notes
    -----
    This function orchestrates the entire Gram-based computation:
    1. Compute small Gram matrices Gr, Gc
    2. Build raw Z'Z via Kronecker algebra (only for kept modes)
    3. Apply polynomial projection correction
    4. Build X'Z with exact zeros for polynomial block

    Memory: O((p_rc)²) vs O(n_obs * p_rc) for explicit Z

    Examples
    --------
    >>> # Setup (in practice, these come from build_psanova_design)
    >>> n_obs, r_plus, c_plus = 100, 8, 8
    >>> BrUr = np.random.randn(n_obs, r_plus)
    >>> BcUc = np.random.randn(n_obs, c_plus)
    >>> keep_mask = np.ones((r_plus, c_plus), dtype=bool)
    >>> scales = np.ones(64)  # All modes kept
    >>> Z_op = LinearOperator((n_obs, 64), matvec=..., rmatvec=...)
    >>> X_poly = np.column_stack([np.ones(n_obs), np.arange(n_obs), np.arange(n_obs)])
    >>> Qx, _ = np.linalg.qr(X_poly)
    >>> ZtZ, XtZ = build_interaction_gram_terms(BrUr, BcUc, scales, Z_op, Qx, X_poly, 3, keep_mask)
    >>> ZtZ.shape
    (64, 64)
    >>> XtZ.shape
    (3, 64)
    >>> np.max(np.abs(XtZ))  # Should be ~0
    0.0
    """
    p_rc = len(scales)

    # 1. Compute Gram matrices
    Gr, Gc = compute_GrGc(BrUr, BcUc)

    # 2. Build full Kronecker product
    K_full = np.kron(Gr, Gc)  # (r+ * c+, r+ * c+)

    # 3. If keep_mask provided, extract only kept modes
    if keep_mask is not None:
        # Convert mask to flat indices (C-order)
        kept_indices = np.where(keep_mask.ravel(order="C"))[0]

        # Extract submatrix for kept modes
        K_kept = K_full[np.ix_(kept_indices, kept_indices)]

        # Apply scaling
        s = scales.ravel()
        ZtZ_raw = (s[:, None] * K_kept) * s[None, :]
    else:
        # No masking - use full Kronecker product
        s = scales.ravel()
        ZtZ_raw = (s[:, None] * K_full) * s[None, :]

    # 4. Apply polynomial projection correction
    ZtZ = projected_ZtZ(ZtZ_raw, Z_op_unprojected, Qx)

    # 5. Build X'Z with polynomial block as exact zeros
    XtZ_poly = XtZ_poly_zero(n_poly, p_rc)

    # Split X into polynomial and other parts
    X_dense = X.toarray() if sp.issparse(X) else np.asarray(X)
    p_total = X_dense.shape[1]

    if p_total > n_poly:
        # Have non-polynomial fixed effects
        X_other = X_dense[:, n_poly:]
        XtZ_oth = XtZ_other(Z_op_unprojected, X_other, Qx)
        XtZ = np.vstack([XtZ_poly, XtZ_oth])
    else:
        # Only polynomial fixed effects
        XtZ = XtZ_poly

    return ZtZ, XtZ
