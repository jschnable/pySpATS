"""
Schur complement utilities for eliminating fixed-effects block.

The Schur complement reduces the mixed model equations by eliminating the fixed-effects
block, leaving only the sparse random-effects system to factorize with CHOLMOD.

Given normal equations:
    C [β] = [X'R^{-1}X    X'R^{-1}Z  ] [β] = [X'R^{-1}y]
      [u]   [Z'R^{-1}X  Z'R^{-1}Z+G^{-1}] [u]   [Z'R^{-1}y]

Form the Schur complement on the fixed block:
    S = Z'R^{-1}Z + G^{-1} - Z'R^{-1}X (X'R^{-1}X)^{-1} X'R^{-1}Z
    r = Z'R^{-1}y - Z'R^{-1}X (X'R^{-1}X)^{-1} X'R^{-1}y

Then solve:
    S u = r  (factorize S with CHOLMOD; reuse factorization)
    β = (X'R^{-1}X)^{-1} [X'R^{-1}(y - Z u)]  (small dense solve)

This is more efficient than factorizing the full C matrix because:
- S is typically large and sparse (random effects only)
- X'R^{-1}X is small and dense (fixed effects only)
- Avoids factorizing mixed sparse-dense structure
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Tuple


def schur_reduce(
    X: sp.spmatrix,
    Z: sp.spmatrix,
    Rinv_scale: float,
    Ginv_blocks: List[sp.spmatrix]
) -> Tuple[sp.csc_matrix, Dict]:
    """
    Build the Schur complement S and precompute parts for RHS construction.

    Computes:
        S = Z'R^{-1}Z + G^{-1} - Z'R^{-1}X (X'R^{-1}X)^{-1} X'R^{-1}Z

    where R = σ²_ε I, so R^{-1} = Rinv_scale * I.

    Parameters
    ----------
    X : scipy.sparse matrix
        Fixed effect design matrix (n × p)
    Z : scipy.sparse matrix
        Random effect design matrix (n × m), concatenated blocks
    Rinv_scale : float
        Precision scaling: 1/σ²_ε
        Since R = σ²_ε I, we have R^{-1} = Rinv_scale * I
    Ginv_blocks : list of scipy.sparse matrices
        Block-diagonal pieces of G^{-1}
        For each block k: G_k^{-1} = (1/σ²_k) I_k

    Returns
    -------
    S : scipy.sparse.csc_matrix
        Schur complement matrix (m × m), sparse
    parts : dict
        Precomputed components for RHS and β recovery:
        - "XtRinvX_inv": (p × p) dense, inverse of X'R^{-1}X
        - "XtRinvZ": (p × m) sparse/dense, X'R^{-1}Z
        - "ZtRinvX": (m × p) sparse/dense, Z'R^{-1}X

    Notes
    -----
    The Schur complement eliminates the fixed-effects block, leaving only
    the sparse random-effects system S to factorize with CHOLMOD.

    This is efficient because:
    - X'R^{-1}X is small (p × p) and dense
    - S is large (m × m) but sparse
    - Avoids factorizing mixed sparse-dense full C

    The matrix S has the same sparsity pattern as Z'R^{-1}Z + G^{-1}
    minus a low-rank correction (rank ≤ p).
    """
    # Ensure CSC format for efficient operations
    if not sp.issparse(X):
        X = sp.csc_matrix(X)
    elif not sp.isspmatrix_csc(X):
        X = X.tocsc()

    if not sp.isspmatrix_csc(Z):
        Z = Z.tocsc()

    n, p = X.shape
    m = Z.shape[1]

    # Transposes
    Xt = X.T
    Zt = Z.T

    # Compute X'R^{-1}X (small, dense)
    # R^{-1} = Rinv_scale * I, so X'R^{-1}X = Rinv_scale * X'X
    XtRinvX = Rinv_scale * (Xt @ X)

    # Convert to dense for inversion (it's small: p × p)
    if sp.issparse(XtRinvX):
        XtRinvX = XtRinvX.toarray()
    else:
        XtRinvX = np.asarray(XtRinvX)

    # Invert X'R^{-1}X (small dense inverse)
    XtRinvX_inv = np.linalg.inv(XtRinvX)

    # Compute X'R^{-1}Z (p × m, typically small × large)
    XtRinvZ = Rinv_scale * (Xt @ Z)

    # Compute Z'R^{-1}X (m × p, transpose of above)
    ZtRinvX = XtRinvZ.T

    # Compute Z'R^{-1}Z (m × m, sparse)
    ZtRinvZ = Rinv_scale * (Zt @ Z)

    # Assemble G^{-1} (block-diagonal)
    if len(Ginv_blocks) > 0:
        Ginv = sp.block_diag(Ginv_blocks, format="csc")
    else:
        # No random effects (shouldn't happen in practice)
        Ginv = sp.csc_matrix((m, m))

    # Compute Schur complement correction term:
    # M = Z'R^{-1}X (X'R^{-1}X)^{-1} X'R^{-1}Z
    # This is (m × p) @ (p × p) @ (p × m) = (m × m)
    # But it's rank ≤ p, so still relatively sparse after subtraction

    # Compute in stages to preserve sparsity
    # M = (Z'R^{-1}X @ (X'R^{-1}X)^{-1}) @ X'R^{-1}Z
    temp = ZtRinvX @ XtRinvX_inv  # (m × p) @ (p × p) = (m × p)

    # Convert temp to sparse if it came out dense
    if not sp.issparse(temp):
        temp = sp.csc_matrix(temp)

    M = temp @ XtRinvZ  # (m × p) @ (p × m) = (m × m)

    # Schur complement: S = Z'R^{-1}Z + G^{-1} - M
    S = (ZtRinvZ - M) + Ginv
    S = S.tocsc()

    # Return S and precomputed parts for later use
    parts = {
        "XtRinvX_inv": XtRinvX_inv,
        "XtRinvZ": XtRinvZ,
        "ZtRinvX": ZtRinvX,
    }

    return S, parts


def schur_rhs(
    X: sp.spmatrix,
    Z: sp.spmatrix,
    y: np.ndarray,
    Rinv_scale: float,
    XtRinvX_inv: np.ndarray,
    XtRinvZ: sp.spmatrix
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute reduced RHS for Schur complement system.

    Computes:
        r = Z'R^{-1}y - Z'R^{-1}X (X'R^{-1}X)^{-1} X'R^{-1}y

    Parameters
    ----------
    X : scipy.sparse matrix
        Fixed effect design matrix (n × p)
    Z : scipy.sparse matrix
        Random effect design matrix (n × m)
    y : np.ndarray
        Response vector (n,)
    Rinv_scale : float
        Precision scaling: 1/σ²_ε
    XtRinvX_inv : np.ndarray
        Inverse of X'R^{-1}X (p × p, dense)
    XtRinvZ : scipy.sparse matrix
        X'R^{-1}Z (p × m)

    Returns
    -------
    r : np.ndarray
        Reduced RHS for Schur system: S u = r (m,)
    XtRinvy : np.ndarray
        X'R^{-1}y for β recovery (p,)
    ZtRinvy : np.ndarray
        Z'R^{-1}y for residual computation (m,)

    Notes
    -----
    The reduced RHS removes the contribution of fixed effects,
    leaving only the random-effects system to solve.
    """
    # Ensure CSC format
    if not sp.issparse(X):
        X = sp.csc_matrix(X)
    elif not sp.isspmatrix_csc(X):
        X = X.tocsc()

    if not sp.isspmatrix_csc(Z):
        Z = Z.tocsc()

    Xt = X.T
    Zt = Z.T

    # Compute X'R^{-1}y and Z'R^{-1}y
    XtRinvy = Rinv_scale * (Xt @ y)  # (p,)
    ZtRinvy = Rinv_scale * (Zt @ y)  # (m,)

    # Compute reduced RHS:
    # r = Z'R^{-1}y - (Z'R^{-1}X @ (X'R^{-1}X)^{-1}) @ X'R^{-1}y
    # Using transpose: Z'R^{-1}X = (X'R^{-1}Z)^T
    temp = XtRinvZ.T @ XtRinvX_inv  # (m × p) @ (p × p) = (m × p)
    correction = temp @ XtRinvy      # (m × p) @ (p,) = (m,)

    if sp.issparse(correction):
        correction = correction.toarray().ravel()
    else:
        correction = np.asarray(correction).ravel()

    r = ZtRinvy - correction

    # Ensure proper array format
    XtRinvy = np.asarray(XtRinvy).ravel()
    ZtRinvy = np.asarray(ZtRinvy).ravel()
    r = np.asarray(r).ravel()

    return r, XtRinvy, ZtRinvy


def recover_beta(
    X: sp.spmatrix,
    y: np.ndarray,
    Z: sp.spmatrix,
    u: np.ndarray,
    Rinv_scale: float,
    XtRinvX_inv: np.ndarray
) -> np.ndarray:
    """
    Recover fixed-effect coefficients from random-effect solution.

    Computes:
        β = (X'R^{-1}X)^{-1} [X'R^{-1}(y - Z u)]

    Parameters
    ----------
    X : scipy.sparse matrix
        Fixed effect design matrix (n × p)
    y : np.ndarray
        Response vector (n,)
    Z : scipy.sparse matrix
        Random effect design matrix (n × m)
    u : np.ndarray
        Random effect solution from Schur system (m,)
    Rinv_scale : float
        Precision scaling: 1/σ²_ε
    XtRinvX_inv : np.ndarray
        Inverse of X'R^{-1}X (p × p, dense)

    Returns
    -------
    beta : np.ndarray
        Fixed effect coefficients (p,)

    Notes
    -----
    This is the back-substitution step after solving the Schur system.
    Since X'R^{-1}X is small and already inverted, this is very fast.
    """
    # Ensure CSC format
    if not sp.issparse(X):
        X = sp.csc_matrix(X)
    elif not sp.isspmatrix_csc(X):
        X = X.tocsc()

    if not sp.isspmatrix_csc(Z):
        Z = Z.tocsc()

    # Compute residual from random effects: y - Z u
    Zu = Z @ u
    if sp.issparse(Zu):
        Zu = Zu.toarray().ravel()
    else:
        Zu = np.asarray(Zu).ravel()

    residual = y - Zu

    # Compute RHS for β: X'R^{-1}(y - Z u)
    rhs_beta = Rinv_scale * (X.T @ residual)
    rhs_beta = np.asarray(rhs_beta).ravel()

    # Solve for β (small dense system, already inverted)
    beta = XtRinvX_inv @ rhs_beta

    return beta
