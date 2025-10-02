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
from scipy.sparse.linalg import LinearOperator
from typing import Dict, List, Tuple, Union

# Import Gram matrix utilities for efficient Kronecker interaction
try:
    from ..spatial.gram_kron import build_interaction_gram_terms
    GRAM_AVAILABLE = True
except ImportError:
    GRAM_AVAILABLE = False


def _compute_ZtZ_gram_aware(
    Z: Union[sp.spmatrix, LinearOperator, "ConcatenatedBlockOperator"],
    Rinv_scale: float,
    X: Union[sp.spmatrix, np.ndarray]
) -> Tuple[Union[sp.spmatrix, np.ndarray], Union[sp.spmatrix, np.ndarray]]:
    """
    Compute Z'R^{-1}Z and X'R^{-1}Z using Gram matrices when available.

    For blocks with `_gram_meta` attribute (Kronecker interaction), uses
    efficient Gram-matrix-based computation. For other blocks, uses standard
    sparse or loop-based methods.

    Parameters
    ----------
    Z : sparse matrix, LinearOperator, or ConcatenatedBlockOperator
        Random effect design matrix (n × m)
    Rinv_scale : float
        Precision scaling: 1/σ²_ε
    X : sparse matrix or ndarray
        Fixed effect design matrix (n × p)

    Returns
    -------
    ZtRinvZ : sparse matrix or ndarray
        Z'R^{-1}Z (m × m)
    XtRinvZ : sparse matrix or ndarray
        X'R^{-1}Z (p × m)

    Notes
    -----
    This function detects whether Z is a ConcatenatedBlockOperator containing
    blocks with Gram metadata, and if so, uses algebraic formulas instead of
    expensive observation loops.
    """
    m = Z.shape[1]

    # Check if Z is a ConcatenatedBlockOperator with potential Gram blocks
    if hasattr(Z, 'blocks') and hasattr(Z, 'block_info'):
        # Z is ConcatenatedBlockOperator
        # Build Z'Z and X'Z block by block
        ZtZ_blocks = []
        XtZ_blocks = []

        for block_wrapper, info in zip(Z.blocks, Z.block_info):
            block = block_wrapper.block
            start, stop = info.start, info.stop
            block_size = stop - start

            # Check if this block has Gram metadata
            if hasattr(block, '_gram_meta') and GRAM_AVAILABLE:
                # Use Gram-based computation
                meta = block._gram_meta
                if meta.get('is_kron_interaction', False):
                    # Build Z'Z and X'Z for this interaction block using Gram formulas
                    ZtZ_block, XtZ_block = build_interaction_gram_terms(
                        BrUr=meta['BrUr'],
                        BcUc=meta['BcUc'],
                        scales=meta['scales'],
                        Z_op_unprojected=meta['Z_op_unprojected'],
                        Qx=meta['Qx'],
                        X=X,
                        n_poly=3,  # Assuming 3 polynomial columns: [1, r, c]
                        keep_mask=meta.get('keep_mask', None)  # Pass keep_mask for mode selection
                    )

                    # Scale by Rinv_scale
                    ZtZ_block = Rinv_scale * ZtZ_block
                    XtZ_block = Rinv_scale * XtZ_block

                    ZtZ_blocks.append((start, stop, ZtZ_block))
                    XtZ_blocks.append((start, stop, XtZ_block))
                    continue

            # Standard computation for non-Gram blocks
            if sp.issparse(block):
                # Sparse block: use direct multiplication
                Zk_T = block.T
                ZtZ_block = Rinv_scale * (Zk_T @ block)
                XtZ_block = Rinv_scale * (X.T @ block).toarray() if sp.issparse(X.T @ block) else Rinv_scale * (X.T @ block)

                ZtZ_blocks.append((start, stop, ZtZ_block))
                XtZ_blocks.append((start, stop, XtZ_block))
            else:
                # LinearOperator without Gram metadata: use loop
                ZtZ_cols = []
                XtZ_cols = []

                for j in range(block_size):
                    e_j = np.zeros(block_size)
                    e_j[j] = 1.0

                    # Z_k @ e_j
                    z_ej = block @ e_j

                    # Z_k' @ (Z_k @ e_j)
                    zt_z_ej = block.rmatvec(z_ej)
                    ZtZ_cols.append(zt_z_ej)

                    # X' @ (Z_k @ e_j)
                    Xt = X.T
                    if sp.issparse(Xt):
                        x_z_ej = (Xt @ z_ej).toarray().ravel()
                    else:
                        x_z_ej = Xt @ z_ej
                    XtZ_cols.append(x_z_ej)

                ZtZ_block = Rinv_scale * np.column_stack(ZtZ_cols)
                XtZ_block = Rinv_scale * np.column_stack(XtZ_cols)

                ZtZ_blocks.append((start, stop, ZtZ_block))
                XtZ_blocks.append((start, stop, XtZ_block))

        # Assemble full Z'Z and X'Z from blocks
        ZtRinvZ = np.zeros((m, m))
        p = X.shape[1]
        XtRinvZ = np.zeros((p, m))

        for start, stop, ZtZ_block in ZtZ_blocks:
            # Convert to dense if sparse
            if sp.issparse(ZtZ_block):
                ZtZ_block = ZtZ_block.toarray()
            ZtRinvZ[start:stop, start:stop] = ZtZ_block

        for start, stop, XtZ_block in XtZ_blocks:
            # Convert to dense if sparse
            if sp.issparse(XtZ_block):
                XtZ_block = XtZ_block.toarray()
            XtRinvZ[:, start:stop] = XtZ_block

        ZtRinvZ = sp.csc_matrix(ZtRinvZ)
        # XtRinvZ stays dense (it's p × m, typically small × large)

    elif sp.issparse(Z):
        # Simple sparse case: use direct multiplication
        Zt = Z.T
        ZtRinvZ = Rinv_scale * (Zt @ Z)
        XtRinvZ = Rinv_scale * (X.T @ Z)
    else:
        # Generic LinearOperator: use loop (fallback)
        ZtRinvZ_list = []
        XtRinvZ_list = []

        for j in range(m):
            e_j = np.zeros(m)
            e_j[j] = 1.0

            z_ej = Z @ e_j
            zt_z_ej = Z.rmatvec(z_ej)
            ZtRinvZ_list.append(zt_z_ej)

            Xt = X.T
            if sp.issparse(Xt):
                x_z_ej = (Xt @ z_ej).toarray().ravel()
            else:
                x_z_ej = Xt @ z_ej
            XtRinvZ_list.append(x_z_ej)

        ZtRinvZ = Rinv_scale * np.column_stack(ZtRinvZ_list)
        ZtRinvZ = sp.csc_matrix(ZtRinvZ)
        XtRinvZ = Rinv_scale * np.column_stack(XtRinvZ_list)

    return ZtRinvZ, XtRinvZ


def schur_reduce(
    X: Union[sp.spmatrix, np.ndarray],
    Z: Union[sp.spmatrix, LinearOperator, "ConcatenatedBlockOperator"],
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
    X : scipy.sparse matrix or ndarray
        Fixed effect design matrix (n × p)
    Z : scipy.sparse matrix, LinearOperator, or ConcatenatedBlockOperator
        Random effect design matrix (n × m), concatenated blocks
        Can be a LinearOperator for efficient Kronecker products
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

    When Z is a LinearOperator (e.g., for Kronecker-structured interactions),
    matrix products are computed using matvec/rmatvec without materialization.
    """
    # Ensure CSC format for efficient operations
    if not sp.issparse(X):
        X = sp.csc_matrix(X)
    elif not sp.isspmatrix_csc(X):
        X = X.tocsc()

    n, p = X.shape

    # Get Z dimensions
    # Z can be sparse matrix, LinearOperator, or ConcatenatedBlockOperator (both have .shape)
    m = Z.shape[1]

    # Transpose of X
    Xt = X.T

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

    # Compute Z'R^{-1}Z and X'R^{-1}Z using Gram-aware helper
    # This uses efficient Gram matrices for Kronecker interaction blocks
    ZtRinvZ, XtRinvZ = _compute_ZtZ_gram_aware(Z, Rinv_scale, X)

    # Compute Z'R^{-1}X (m × p, transpose of X'R^{-1}Z)
    ZtRinvX = XtRinvZ.T

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
    if not sp.issparse(ZtRinvZ):
        ZtRinvZ = sp.csc_matrix(ZtRinvZ)
    if not sp.issparse(M):
        M = sp.csc_matrix(M)

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
    X: Union[sp.spmatrix, np.ndarray],
    Z: Union[sp.spmatrix, LinearOperator, "ConcatenatedBlockOperator"],
    y: np.ndarray,
    Rinv_scale: float,
    XtRinvX_inv: np.ndarray,
    XtRinvZ: Union[sp.spmatrix, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute reduced RHS for Schur complement system.

    Computes:
        r = Z'R^{-1}y - Z'R^{-1}X (X'R^{-1}X)^{-1} X'R^{-1}y

    Parameters
    ----------
    X : scipy.sparse matrix or ndarray
        Fixed effect design matrix (n × p)
    Z : scipy.sparse matrix, LinearOperator, or ConcatenatedBlockOperator
        Random effect design matrix (n × m)
    y : np.ndarray
        Response vector (n,)
    Rinv_scale : float
        Precision scaling: 1/σ²_ε
    XtRinvX_inv : np.ndarray
        Inverse of X'R^{-1}X (p × p, dense)
    XtRinvZ : scipy.sparse matrix or ndarray
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

    When Z is a LinearOperator, Z'y is computed using rmatvec.
    """
    # Ensure CSC format
    if not sp.issparse(X):
        X = sp.csc_matrix(X)
    elif not sp.isspmatrix_csc(X):
        X = X.tocsc()

    Xt = X.T

    # Compute X'R^{-1}y
    XtRinvy = Rinv_scale * (Xt @ y)  # (p,)

    # Compute Z'R^{-1}y
    if sp.issparse(Z):
        Zt = Z.T
        ZtRinvy = Rinv_scale * (Zt @ y)  # (m,)
    else:
        # Z is LinearOperator or ConcatenatedBlockOperator
        if hasattr(Z, 'rmatvec'):
            ZtRinvy = Rinv_scale * Z.rmatvec(y)
        else:
            ZtRinvy = Rinv_scale * Z.rmatvec(y)

    # Compute reduced RHS:
    # r = Z'R^{-1}y - (Z'R^{-1}X @ (X'R^{-1}X)^{-1}) @ X'R^{-1}y
    # Using transpose: Z'R^{-1}X = (X'R^{-1}Z)^T
    if sp.issparse(XtRinvZ):
        temp = XtRinvZ.T @ XtRinvX_inv  # (m × p) @ (p × p) = (m × p)
    else:
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
    X: Union[sp.spmatrix, np.ndarray],
    y: np.ndarray,
    Z: Union[sp.spmatrix, LinearOperator, "ConcatenatedBlockOperator"],
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
    X : scipy.sparse matrix or ndarray
        Fixed effect design matrix (n × p)
    y : np.ndarray
        Response vector (n,)
    Z : scipy.sparse matrix, LinearOperator, or ConcatenatedBlockOperator
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

    When Z is a LinearOperator, Z u is computed using matvec.
    """
    # Ensure CSC format
    if not sp.issparse(X):
        X = sp.csc_matrix(X)
    elif not sp.isspmatrix_csc(X):
        X = X.tocsc()

    # Compute residual from random effects: y - Z u
    if sp.issparse(Z):
        Zu = Z @ u
        if sp.issparse(Zu):
            Zu = Zu.toarray().ravel()
        else:
            Zu = np.asarray(Zu).ravel()
    else:
        # Z is LinearOperator or ConcatenatedBlockOperator
        if hasattr(Z, 'matvec'):
            Zu = Z.matvec(u)
        else:
            Zu = Z.matvec(u)
        Zu = np.asarray(Zu).ravel()

    residual = y - Zu

    # Compute RHS for β: X'R^{-1}(y - Z u)
    rhs_beta = Rinv_scale * (X.T @ residual)
    rhs_beta = np.asarray(rhs_beta).ravel()

    # Solve for β (small dense system, already inverted)
    beta = XtRinvX_inv @ rhs_beta

    return beta
