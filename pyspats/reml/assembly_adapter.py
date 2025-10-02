"""
Assembly adapter for constructing mixed model coefficient matrix C(θ) and RHS.

This module provides a thin wrapper around existing design matrix builders
to construct the mixed model normal equations:

C(θ) = [[X'R^{-1}X,  X'R^{-1}Z  ],
        [Z'R^{-1}X,  Z'R^{-1}Z+G^{-1}]]

where:
- R = σ²_ε I (residual covariance)
- G = block_diag(σ²_k I_k) (random effect covariances)
- θ = {σ²_ε, σ²_k, ...} (variance components)

The adapter ensures:
- Contiguous block structure for random effects
- BlockInfo metadata for ED computation
- Proper slicing for extracting β and u from solution
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Tuple, Callable

from ..ed_selected_inverse import BlockInfo


def make_assemble_fn(builder: Callable) -> Callable:
    """
    Create an assembly function from a design matrix builder.

    The assembly function constructs C(θ), RHS, and metadata for given variance components θ.

    Parameters
    ----------
    builder : callable
        Function that builds design matrices from variance components.
        Signature: builder(theta) -> (X, Z_dict, y, block_order)
        Returns:
        - X : np.ndarray or sparse
            Fixed effect design matrix (n × p)
        - Z_dict : dict[str, sparse]
            Random effect design matrices by block name
        - y : np.ndarray
            Response vector (n,)
        - block_order : list[str]
            Ordered list of block names for assembly

    Returns
    -------
    assemble_fn : callable
        Assembly function compatible with fit_reml().
        Signature: assemble_fn(theta) -> (C, rhs, blocks, rank_X, n, X, Z_dict)
        Attributes:
        - assemble_fn.beta_slice : slice for β in solution
        - assemble_fn.random_slice : slice for u in solution
        - assemble_fn.y : response vector

    Notes
    -----
    The builder is responsible for creating the design matrices.
    The assembly adapter handles:
    - Constructing weighted normal equations with R^{-1} = (1/σ²_ε) I
    - Adding penalty matrices G^{-1} = diag(1/σ²_k I_k)
    - Creating BlockInfo metadata
    - Tracking slices for coefficient extraction

    Examples
    --------
    >>> def my_builder(theta):
    ...     # Build X, Z_dict, y for your model
    ...     return X, Z_dict, y, ["block1", "block2"]
    >>> assemble_fn = make_assemble_fn(my_builder)
    >>> C, rhs, blocks, rank_X, n, X, Z_dict = assemble_fn(theta)
    """

    def assemble_fn(theta: Dict[str, float]) -> Tuple:
        """
        Assemble mixed model coefficient matrix and RHS for given variance components.

        Parameters
        ----------
        theta : dict[str, float]
            Variance components: {"eps": σ²_ε, "<block>": σ²_k, ...}

        Returns
        -------
        C : scipy.sparse.csc_matrix
            Mixed model coefficient matrix in model order
        rhs : np.ndarray
            Right-hand side of normal equations
        blocks : list[BlockInfo]
            Random effect block metadata (contiguous indices)
        rank_X : int
            Rank of fixed effect design matrix
        n : int
            Number of observations
        X : np.ndarray or sparse
            Fixed effect design matrix (for residual computation)
        Z_dict : dict[str, sparse]
            Random effect design matrices by block name
        """
        # Call builder to get design matrices
        X, Z_dict, y, block_order = builder(theta)

        n = y.shape[0]
        p = X.shape[1]

        # Compute rank of X (convert to dense if sparse and small enough)
        if sp.issparse(X):
            if X.shape[1] < 1000:  # Only densify for reasonably sized X
                X_dense = X.toarray()
                rank_X = np.linalg.matrix_rank(X_dense)
            else:
                # For large X, use sparse SVD or assume full rank
                rank_X = min(X.shape)
        else:
            rank_X = np.linalg.matrix_rank(X)

        # Build concatenated Z matrix and create BlockInfo metadata
        Z_list = []
        blocks = []
        start = 0

        for name in block_order:
            Zk = Z_dict[name]
            m = Zk.shape[1]
            blocks.append(BlockInfo(name, start, start + m, is_random=True))
            Z_list.append(Zk)
            start += m

        # Concatenate all random effect design matrices
        if Z_list:
            Z = sp.hstack(Z_list, format="csc")
        else:
            Z = sp.csc_matrix((n, 0))

        # Precision (inverse covariance) weights
        # R^{-1} = (1/σ²_ε) I
        lam_eps = 1.0 / theta["eps"]

        # Construct weighted normal equations
        # For R = σ²_ε I, we have R^{-1} = lam_eps I
        # So X'R^{-1}X = lam_eps * X'X, etc.

        # Convert X to sparse if needed for efficient operations
        if not sp.issparse(X):
            X = sp.csr_matrix(X)

        Xt = X.T
        Zt = Z.T

        # Upper-left block: X'R^{-1}X
        XtX = lam_eps * (Xt @ X)

        # Upper-right block: X'R^{-1}Z
        if Z.shape[1] > 0:
            XtZ = lam_eps * (Xt @ Z)
        else:
            XtZ = sp.csc_matrix((p, 0))

        # Lower-left block: Z'R^{-1}X
        ZtX = XtZ.T

        # Lower-right block: Z'R^{-1}Z + G^{-1}
        if Z.shape[1] > 0:
            ZtZ = lam_eps * (Zt @ Z)

            # Add penalty matrices G^{-1} = diag(1/σ²_k I_k)
            # For each block k: G_k = σ²_k I, so G_k^{-1} = (1/σ²_k) I
            g_inv_blocks = []
            for b in blocks:
                lam_k = 1.0 / theta[b.name]
                g_inv_blocks.append(sp.eye(b.size, format="csc") * lam_k)

            Ginv = sp.block_diag(g_inv_blocks, format="csc")
            ZtZ_plus_Ginv = ZtZ + Ginv
        else:
            ZtZ_plus_Ginv = sp.csc_matrix((0, 0))

        # Assemble full coefficient matrix C(θ)
        if Z.shape[1] > 0:
            C = sp.bmat([
                [XtX, XtZ],
                [ZtX, ZtZ_plus_Ginv]
            ], format="csc")
        else:
            # No random effects
            C = XtX.tocsc()

        # Right-hand side
        rhs_top = lam_eps * (Xt @ y)
        if Z.shape[1] > 0:
            rhs_bot = lam_eps * (Zt @ y)
            rhs = np.concatenate([
                np.asarray(rhs_top).ravel(),
                np.asarray(rhs_bot).ravel()
            ])
        else:
            rhs = np.asarray(rhs_top).ravel()

        # Attach metadata as function attributes for fit_reml to access
        assemble_fn.beta_slice = slice(0, p)
        assemble_fn.random_slice = slice(p, p + Z.shape[1])
        assemble_fn.y = y

        return C, rhs, blocks, rank_X, n, X, Z_dict

    return assemble_fn


def make_builder_from_psanova(
    X_poly: np.ndarray,
    Z_r: np.ndarray,
    Z_c: np.ndarray,
    Z_rc: np.ndarray,
    y: np.ndarray,
    genotype_Z: np.ndarray = None,
    other_Z_dict: Dict[str, np.ndarray] = None
) -> Callable:
    """
    Create a builder function from PS-ANOVA design matrices.

    This is a convenience function for creating a builder from the output
    of build_psanova_design().

    Parameters
    ----------
    X_poly : np.ndarray
        Fixed polynomial design: [1, r_norm, c_norm]
    Z_r : np.ndarray
        Row-smooth random design
    Z_c : np.ndarray
        Column-smooth random design
    Z_rc : np.ndarray
        Interaction random design
    y : np.ndarray
        Response vector
    genotype_Z : np.ndarray, optional
        Genotype random effects (if genotype_as_random=True)
    other_Z_dict : dict, optional
        Other random effects: {name: Z_matrix}

    Returns
    -------
    builder : callable
        Builder function for make_assemble_fn()

    Examples
    --------
    >>> from pyspats.psanova_basis import build_psanova_design
    >>> X_poly, Z_r, Z_c, Z_rc, blocks = build_psanova_design(r, c, nkr=10, nkc=10)
    >>> builder = make_builder_from_psanova(X_poly, Z_r, Z_c, Z_rc, y)
    >>> assemble_fn = make_assemble_fn(builder)
    """
    # Convert to sparse for efficiency
    Z_r_sp = sp.csc_matrix(Z_r) if not sp.issparse(Z_r) else Z_r
    Z_c_sp = sp.csc_matrix(Z_c) if not sp.issparse(Z_c) else Z_c
    Z_rc_sp = sp.csc_matrix(Z_rc) if not sp.issparse(Z_rc) else Z_rc

    def builder(theta: Dict[str, float]) -> Tuple:
        """Build design matrices (independent of theta in this simple case)."""
        # Fixed effects
        X = X_poly.copy()

        # Random effects in order
        Z_dict = {}
        block_order = []

        # Spatial blocks (PS-ANOVA)
        if Z_r.shape[1] > 0:
            Z_dict["row_smooth"] = Z_r_sp
            block_order.append("row_smooth")

        if Z_c.shape[1] > 0:
            Z_dict["col_smooth"] = Z_c_sp
            block_order.append("col_smooth")

        if Z_rc.shape[1] > 0:
            Z_dict["interaction_smooth"] = Z_rc_sp
            block_order.append("interaction_smooth")

        # Genotype (if provided)
        if genotype_Z is not None:
            Z_dict["genotype"] = sp.csc_matrix(genotype_Z)
            block_order.append("genotype")

        # Other random effects
        if other_Z_dict is not None:
            for name, Z_mat in other_Z_dict.items():
                Z_dict[name] = sp.csc_matrix(Z_mat)
                block_order.append(name)

        return X, Z_dict, y, block_order

    return builder
