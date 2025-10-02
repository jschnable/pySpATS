"""
Exact Effective Dimension (ED) computation using CHOLMOD selected inverse.

This module computes ED_k = m_k - tr(G_k^{-1} C^{-1}_{kk}) for each random-effect block,
where C is the mixed-model coefficient matrix and C^{-1}_{kk} is obtained exactly via
the CHOLMOD sparse Cholesky factorization with selected inverse (diagonal extraction).

Requires: SuiteSparse/CHOLMOD installed system-wide, accessed via scikit-sparse.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Optional

try:
    from sksparse import cholmod
    CHOLMOD_AVAILABLE = True
except ImportError:
    CHOLMOD_AVAILABLE = False
    cholmod = None


class BlockInfo:
    """
    Holds index information for blocks inside C in *model order*.

    Attributes
    ----------
    name : str
        Block identifier (e.g., 'genotype', 'spatial', 'block')
    start : int
        Starting index in model order
    stop : int
        Ending index (exclusive) in model order
    is_random : bool
        Whether this is a random effect block
    """
    __slots__ = ("name", "start", "stop", "is_random")

    def __init__(self, name: str, start: int, stop: int, is_random: bool = True):
        self.name = name
        self.start = start
        self.stop = stop
        self.is_random = is_random

    @property
    def slice(self) -> slice:
        """Return slice object for this block."""
        return slice(self.start, self.stop)

    @property
    def size(self) -> int:
        """Return size of this block."""
        return self.stop - self.start

    def __repr__(self) -> str:
        return f"BlockInfo('{self.name}', {self.start}:{self.stop}, random={self.is_random})"


def factorize_C_cholmod(C: sp.spmatrix):
    """
    Factorize symmetric positive-definite C using CHOLMOD.

    CHOLMOD computes P'CP = LDL' where P is a fill-reducing permutation.

    Parameters
    ----------
    C : scipy.sparse matrix
        Symmetric positive-definite coefficient matrix in model order

    Returns
    -------
    factor : cholmod.Factor
        CHOLMOD factor object containing L, D, and permutation
    P : np.ndarray
        Permutation vector: C_cholmod[i,j] = C_model[P[i], P[j]]
    invP : np.ndarray
        Inverse permutation: invP[P[i]] = i

    Raises
    ------
    RuntimeError
        If CHOLMOD is not available or factorization fails
    """
    if not CHOLMOD_AVAILABLE:
        raise RuntimeError(
            "CHOLMOD not available. Install scikit-sparse>=0.4.14 and ensure "
            "SuiteSparse/CHOLMOD is installed system-wide."
        )

    if not sp.isspmatrix_csc(C):
        C = C.tocsc()

    # Factorize using CHOLMOD
    try:
        factor = cholmod.cholesky(C)
    except Exception as e:
        raise RuntimeError(f"CHOLMOD factorization failed: {e}")

    # Extract permutation
    # factor.P() returns the permutation vector where P[i] gives the model-order index
    # that maps to CHOLMOD-order index i
    P = factor.P()
    invP = np.argsort(P)

    return factor, P, invP


def _diag_selected_inverse(
    factor,
    idx_model: np.ndarray,
    P: np.ndarray,
    invP: np.ndarray
) -> np.ndarray:
    """
    Compute selected diagonal entries of inv(C) for model-order indices.

    Solves C * x_i = e_i for each requested index i to get diagonal entry C^{-1}_{ii} = x_i[i].

    Parameters
    ----------
    factor : cholmod.Factor
        CHOLMOD factorization object
    idx_model : np.ndarray
        Indices in model order for which to compute inv(C) diagonal
    P : np.ndarray
        Permutation vector from factorization
    invP : np.ndarray
        Inverse permutation

    Returns
    -------
    np.ndarray
        Diagonal entries of inv(C) at the requested model-order indices

    Notes
    -----
    CHOLMOD computes C = LL' factorization. For the diagonal of C^{-1}:
    - C^{-1} = L^{-T} L^{-1}
    - diag(C^{-1})[i] = sum_j (L^{-1}_{ji})^2

    Rather than computing L^{-1} (which would be dense), we solve C * x = e_i
    for each requested index i, giving diagonal entry C^{-1}_{ii} = x[i].
    This is efficient for sparse systems and leverages the already-computed factorization.
    """
    n_indices = len(idx_model)
    diag_inv = np.zeros(n_indices, dtype=np.float64)

    for k, i in enumerate(idx_model):
        # Create unit vector e_i in model order
        e_i = np.zeros(factor.L().shape[0], dtype=np.float64)
        e_i[i] = 1.0

        # Solve C * x = e_i using the factorization
        x_i = factor.solve_A(e_i)

        # Extract diagonal entry: C^{-1}_{ii} = x_i[i]
        diag_inv[k] = x_i[i]

    return diag_inv


def ed_components_from_selected_inverse(
    C: sp.spmatrix,
    blocks: List[BlockInfo],
    G_blocks: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, float]:
    """
    Compute exact ED_k = m_k - tr(G_k^{-1} C^{-1}_{kk}) for each random-effect block.

    Uses CHOLMOD sparse Cholesky factorization with selected inverse (Takahashi equations)
    to compute diagonal entries of C^{-1} without forming the full inverse.

    Parameters
    ----------
    C : scipy.sparse matrix
        Mixed-model coefficient matrix in model order. Should be symmetric positive-definite.
        Typically this is the Henderson/REML normal equations matrix:
        C = [[X'WX,  X'WZ  ],
             [Z'WX,  Z'WZ+G]]
    blocks : list of BlockInfo
        Ordered block metadata covering the random effects (and optionally fixed effects).
        Indices should be in model order matching C.
    G_blocks : dict, optional
        Dictionary mapping block name to penalty/precision matrix G_k.
        - If None: assumes G_k = I for all blocks
        - If scalar: assumes G_k = scalar * I
        - If 1D array: assumes G_k = diag(array)

    Returns
    -------
    dict
        Dictionary mapping random-effect block name -> ED_k value.
        Only includes random blocks (is_random=True).

    Raises
    ------
    RuntimeError
        If CHOLMOD is not available or factorization fails

    Notes
    -----
    The effective dimension formula is:

    ED_k = m_k - tr(G_k^{-1} C^{-1}_{kk})

    where:
    - m_k is the nominal dimension (number of parameters in block k)
    - G_k is the penalty/precision matrix for block k
    - C^{-1}_{kk} is the block-diagonal portion of inv(C) for block k
    - diag(C^{-1}) is obtained from CHOLMOD selected inverse

    For the common case where G_k = sigma_k^2 I:
    - tr(G_k^{-1} C^{-1}_{kk}) = (1/sigma_k^2) * sum(diag(C^{-1})[k])

    References
    ----------
    Takahashi, K., Fagan, J., & Chen, M.-S. (1973). Formation of a sparse bus
    impedance matrix and its application to short circuit study. 8th PICA Conf. Proc., 63-69.

    Examples
    --------
    >>> # Simple example with 2 random blocks
    >>> C = sp.eye(10, format='csc') * 2.0  # Simple SPD matrix
    >>> blocks = [
    ...     BlockInfo('fixed', 0, 3, is_random=False),
    ...     BlockInfo('genotype', 3, 6, is_random=True),
    ...     BlockInfo('spatial', 6, 10, is_random=True)
    ... ]
    >>> ed = ed_components_from_selected_inverse(C, blocks)
    >>> # ED_k ≈ m_k - m_k/2 = m_k/2 for well-conditioned I-like matrices
    """
    if not CHOLMOD_AVAILABLE:
        raise RuntimeError(
            "CHOLMOD not available. Install scikit-sparse>=0.4.14 with "
            "pip install scikit-sparse (requires SuiteSparse/CHOLMOD system library)."
        )

    # Factorize C once
    factor, P, invP = factorize_C_cholmod(C)

    ed = {}

    for b in blocks:
        if not b.is_random:
            continue

        # Get indices for this block in model order
        idx = np.arange(b.start, b.stop, dtype=int)

        # Compute diagonal of inv(C) for this block
        diag_inv = _diag_selected_inverse(factor, idx, P, invP)

        m_k = b.size

        # Get G_k for this block
        Gk = None if G_blocks is None else G_blocks.get(b.name, None)

        if Gk is None:
            # Default: G_k = I (or sigma^2 I with sigma^2=1)
            # tr(G_k^{-1} C^{-1}_{kk}) = tr(C^{-1}_{kk}) = sum(diag_inv)
            tr_term = float(np.sum(diag_inv))
        else:
            # Handle scalar or diagonal G_k
            if np.ndim(Gk) == 0:
                # Scalar: G_k = scalar * I
                # tr(G_k^{-1} C^{-1}_{kk}) = (1/scalar) * sum(diag_inv)
                tr_term = float(np.sum(diag_inv) / float(Gk))
            else:
                # Diagonal: G_k = diag(Gk)
                # tr(G_k^{-1} C^{-1}_{kk}) = sum(diag_inv / Gk)
                Gk = np.asarray(Gk)
                if Gk.shape != (m_k,):
                    raise ValueError(
                        f"G_blocks['{b.name}'] must be scalar or 1D array of length {m_k}, "
                        f"got shape {Gk.shape}"
                    )
                tr_term = float(np.sum(diag_inv / Gk))

        # ED_k = m_k - tr(G_k^{-1} C^{-1}_{kk})
        ed[b.name] = float(m_k - tr_term)

    return ed


def ed_components_from_schur_inverse(
    S: sp.spmatrix,
    blocks: List[BlockInfo],
    G_blocks: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, float]:
    """
    Compute exact ED_k from Schur complement matrix S (random effects only).

    This is a specialized version of ed_components_from_selected_inverse for use
    with the Schur complement elimination of fixed effects. The Schur complement
    S = Z'R^{-1}Z + G^{-1} - Z'R^{-1}X (X'R^{-1}X)^{-1} X'R^{-1}Z
    corresponds to (C^{-1})_uu, the random-random block of the full inverse.

    Parameters
    ----------
    S : scipy.sparse matrix
        Schur complement matrix (random effects only), symmetric positive-definite
    blocks : list of BlockInfo
        Random effect block metadata with contiguous indices in S
    G_blocks : dict, optional
        Dictionary mapping block name to penalty/precision matrix G_k.
        - If None: assumes G_k = I for all blocks
        - If scalar: assumes G_k = scalar * I
        - If 1D array: assumes G_k = diag(array)

    Returns
    -------
    dict
        Dictionary mapping random-effect block name -> ED_k value.

    Notes
    -----
    When using Schur complement, the fixed effects are eliminated and only the
    random effects remain. The inverse of S directly gives (C^{-1})_uu, so:

        ED_k = m_k - tr(G_k^{-1} (S^{-1})_{kk})

    This is mathematically equivalent to computing EDs from the full C matrix,
    but more efficient since S is typically smaller and sparser.

    See Also
    --------
    ed_components_from_selected_inverse : Full system version
    """
    # This is identical to the full version, just operating on S instead of C
    # The mathematics is the same: we compute diag(S^{-1}) for random blocks
    return ed_components_from_selected_inverse(S, blocks, G_blocks)


def is_cholmod_available() -> bool:
    """
    Check if CHOLMOD is available.

    Returns
    -------
    bool
        True if scikit-sparse with CHOLMOD support is available
    """
    return CHOLMOD_AVAILABLE
