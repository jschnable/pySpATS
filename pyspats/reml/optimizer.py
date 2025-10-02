"""
REML optimizer with CHOLMOD factorization reuse and ED-based variance updates.

This module implements iterative REML estimation using:
- One CHOLMOD factorization per iteration of C(θ)
- Exact effective dimensions (ED) via Takahashi selected inverse
- Closed-form variance component updates: σ²_k = (u_k' u_k) / ED_k

The optimizer reuses the factorization for:
- Solving mixed model equations: C(θ) * [β; u] = rhs
- Computing exact EDs via diagonal of C(θ)^{-1}
- Residual computation and sum of squares

Reference:
Rodriguez-Alvarez et al. (2015) "Fast smoothing parameter separation in
multidimensional generalized P-splines: the SAP algorithm"
"""

from __future__ import annotations
import os
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Tuple, Optional, Union

try:
    from sksparse.cholmod import cholesky
    CHOLMOD_AVAILABLE = True
except ImportError:
    CHOLMOD_AVAILABLE = False
    cholesky = None

from ..ed_selected_inverse import ed_components_from_selected_inverse, ed_components_from_schur_inverse, BlockInfo
from ..spatial.block_operator import BlockLinearOperator, ConcatenatedBlockOperator
from .schur import schur_reduce, schur_rhs, recover_beta

# Environment flag to disable Schur complement (for debugging only)
DISABLE_SCHUR = os.getenv("PYSPATS_DISABLE_SCHUR", "0") == "1"


@dataclass
class REMLOptions:
    """
    Configuration options for REML optimizer.

    Attributes
    ----------
    max_iter : int, default=50
        Maximum number of REML iterations
    tol_rel : float, default=1e-4
        Relative tolerance for convergence (max relative change across all variance components)
    tol_abs : float, default=1e-8
        Absolute tolerance floor for convergence
    verbose : bool, default=False
        If True, print iteration progress
    safeguard_min : float, default=1e-12
        Minimum floor for variance components (prevents numerical issues)
    """
    max_iter: int = 50
    tol_rel: float = 1e-4
    tol_abs: float = 1e-8
    verbose: bool = False
    safeguard_min: float = 1e-12


@dataclass
class REMLResult:
    """
    REML optimization result.

    Attributes
    ----------
    beta : np.ndarray
        Fixed effect coefficients
    u : dict[str, np.ndarray]
        Random effect coefficients per block
    sigma2 : dict[str, float]
        Variance components: {"eps": σ²_ε, "<block_name>": σ²_k, ...}
    ed : dict[str, float]
        Effective dimensions per random block: {<block_name>: ED_k, ...}
    ed_residual : float
        Residual effective dimension: n - rank(X) - sum(ED_k)
    n_iter : int
        Number of iterations performed
    converged : bool
        Whether convergence was achieved within tolerance
    log : list[dict]
        Per-iteration diagnostics (optional, for debugging)
    """
    beta: np.ndarray
    u: Dict[str, np.ndarray]
    sigma2: Dict[str, float]
    ed: Dict[str, float]
    ed_residual: float
    n_iter: int
    converged: bool
    log: List[Dict] = field(default_factory=list)


def fit_reml(
    assemble_fn: Callable,
    init_sigma2: Dict[str, float],
    options: REMLOptions = REMLOptions(),
) -> REMLResult:
    """
    Iterative REML estimation using ED-based variance updates with CHOLMOD factorization reuse.

    Performs REML estimation by iteratively:
    1. Assembling mixed model coefficient matrix C(θ) and RHS
    2. Factorizing C(θ) once with CHOLMOD
    3. Solving for β̂, û using the factorization
    4. Computing exact EDs via Takahashi selected inverse (reuses factorization)
    5. Updating variance components using closed-form ED-based formulas:
       - σ²_k ← (u_k' u_k) / ED_k  for each random block k
       - σ²_ε ← (e' e) / (n - rank(X))  for residual

    Parameters
    ----------
    assemble_fn : callable
        Function that constructs mixed model system for given variance components.
        Signature: assemble_fn(theta: dict) -> tuple
        Must return:
        - C : scipy.sparse matrix
            Mixed model coefficient matrix in model order
        - rhs : np.ndarray
            Right-hand side of normal equations
        - blocks : list[BlockInfo]
            Random effect block metadata with contiguous indices
        - rank_X : int
            Rank of fixed effect design matrix
        - n : int
            Number of observations
        - X : np.ndarray or sparse
            Fixed effect design matrix (for residual computation)
        - Z_dict : dict[str, sparse]
            Random effect design matrices by block name
        Must also have attributes:
        - assemble_fn.beta_slice : slice for extracting β from solution
        - assemble_fn.random_slice : slice for extracting u from solution
        - assemble_fn.y : np.ndarray, response vector
    init_sigma2 : dict[str, float]
        Initial variance components: {"eps": σ²_ε, "<block>": σ²_k, ...}
    options : REMLOptions, optional
        Optimization configuration

    Returns
    -------
    REMLResult
        Optimization result with converged variance components, coefficients, and EDs

    Raises
    ------
    ImportError
        If CHOLMOD is not available (requires scikit-sparse with SuiteSparse)
    RuntimeError
        If CHOLMOD factorization fails

    Notes
    -----
    This implementation reuses one CHOLMOD factorization per iteration for:
    - Mixed model equation solve: C(θ)^{-1} @ rhs
    - Exact ED computation: diag(C(θ)^{-1}) via Takahashi selected inverse
    - No stochastic approximations; fully deterministic

    The ED-based variance updates are closed-form:
    - σ²_k = (u_k' u_k) / ED_k matches SpATS/LMMsolver practice
    - σ²_ε = (e' e) / (n - rank(X)) is standard residual variance

    Convergence is based on maximum relative change across all variance components.

    Examples
    --------
    >>> from pyspats.reml import fit_reml, REMLOptions
    >>> # Assuming you have an assemble_fn set up:
    >>> init = {"eps": 1.0, "spatial": 1.0, "genotype": 1.0}
    >>> result = fit_reml(assemble_fn, init, REMLOptions(max_iter=50, verbose=True))
    >>> print(f"Converged: {result.converged}, iterations: {result.n_iter}")
    >>> print(f"Variance components: {result.sigma2}")
    >>> print(f"Effective dimensions: {result.ed}")
    """
    if not CHOLMOD_AVAILABLE:
        raise ImportError(
            "REML optimizer requires CHOLMOD (scikit-sparse). "
            "Install with: pip install scikit-sparse\n"
            "System requirement: SuiteSparse/CHOLMOD library\n"
            "  - macOS: brew install suite-sparse\n"
            "  - Ubuntu/Debian: sudo apt-get install libsuitesparse-dev\n"
            "  - Then: pip install scikit-sparse"
        )

    # Initialize with safeguarded variance components
    theta = {k: max(v, options.safeguard_min) for k, v in init_sigma2.items()}
    log = []

    # Storage for final iteration values
    beta_hat = None
    u_hat = {}
    ed_k = {}
    ed_resid = 0.0

    for it in range(options.max_iter):
        # 1) Get design matrices and metadata for current variance components
        _, _, blocks, rank_X, n, X, Z_dict = assemble_fn(theta)

        # Build concatenated Z matrix (all random effects)
        # Check if any blocks are LinearOperators
        block_order = [b.name for b in blocks]
        Z_list = [Z_dict[name] for name in block_order]

        if not Z_list:
            # No random effects
            Z = sp.csc_matrix((n, 0))
            use_block_operator = False
        else:
            # Check if any block is a LinearOperator
            has_linear_operator = any(isinstance(z, LinearOperator) for z in Z_list)

            if has_linear_operator:
                # Use ConcatenatedBlockOperator for mixed sparse/LinearOperator blocks
                wrapped_blocks = []
                for name, z in zip(block_order, Z_list):
                    wrapped_blocks.append(BlockLinearOperator(z, name))
                Z = ConcatenatedBlockOperator(wrapped_blocks, blocks)
                use_block_operator = True
            else:
                # All blocks are sparse: use traditional hstack
                Z = sp.hstack(Z_list, format="csc")
                use_block_operator = False

        y = assemble_fn.y
        Rinv_scale = 1.0 / theta["eps"]

        # Build G^{-1} blocks (each G_k^{-1} = (1/σ²_k) I_k)
        Ginv_blocks = []
        for b in blocks:
            lam_k = 1.0 / theta[b.name]
            Ginv_blocks.append(sp.eye(b.size, format="csc") * lam_k)

        # Schur complement path (default) vs. full system (debug only)
        if not DISABLE_SCHUR:
            # **DEFAULT PATH: Schur complement**
            # Eliminates fixed effects, factorizes sparse random block S only

            # 2a) Build Schur complement S and precomputed parts
            S, parts = schur_reduce(X, Z, Rinv_scale, Ginv_blocks)

            # 2b) Compute reduced RHS
            r, XtRinvy, ZtRinvy = schur_rhs(
                X, Z, y, Rinv_scale,
                parts["XtRinvX_inv"],
                parts["XtRinvZ"]
            )

            # 3) Factorize S with CHOLMOD (sparse random block only)
            try:
                factor = cholesky(S)
            except Exception as e:
                raise RuntimeError(f"CHOLMOD factorization of Schur complement failed at iteration {it}: {e}")

            # 4) Solve for random effects: S u = r
            u_concat = factor(r)
            u_concat = np.asarray(u_concat).ravel()

            # 5) Recover fixed effects: β = (X'R^{-1}X)^{-1} [X'R^{-1}(y - Z u)]
            beta_hat = recover_beta(X, y, Z, u_concat, Rinv_scale, parts["XtRinvX_inv"])

            # Extract per-block random effects
            u_hat = {}
            for b in blocks:
                u_hat[b.name] = u_concat[b.start:b.stop]

            # 6) Compute exact EDs from S^{-1} (reuses same factorization)
            # For Schur complement: (C^{-1})_uu = S^{-1}
            G_blocks = {b.name: theta[b.name] for b in blocks}
            ed_k = ed_components_from_schur_inverse(S, blocks, G_blocks=G_blocks)

        else:
            # **DEBUG PATH ONLY: Full system C (for numerical verification)**
            # This path is only for testing/debugging via PYSPATS_DISABLE_SCHUR=1

            # Assemble full C and rhs
            C, rhs, _, _, _, _, _ = assemble_fn(theta)

            # Ensure CSC format
            if not sp.isspmatrix_csc(C):
                C = C.tocsc()

            # Factorize full C
            try:
                factor = cholesky(C)
            except Exception as e:
                raise RuntimeError(f"CHOLMOD factorization of full C failed at iteration {it}: {e}")

            # Solve full system
            sol = factor(rhs)

            # Extract β̂ and û
            beta_hat = np.asarray(sol[assemble_fn.beta_slice]).ravel()

            u_hat = {}
            for b in blocks:
                global_slice = slice(
                    assemble_fn.random_slice.start + (b.start - blocks[0].start),
                    assemble_fn.random_slice.start + (b.stop - blocks[0].start)
                )
                u_hat[b.name] = np.asarray(sol[global_slice]).ravel()

            # Compute EDs from full C
            G_blocks = {b.name: theta[b.name] for b in blocks}
            ed_k = ed_components_from_selected_inverse(C, blocks, G_blocks=G_blocks)

        # Residual effective dimension
        ed_sum = float(sum(ed_k.values()))
        ed_resid = float(n - rank_X - ed_sum)

        # 5) Compute residuals: e = y - X β̂ - Σ_k Z_k û_k
        y = assemble_fn.y
        mu = X @ beta_hat
        for name, uk in u_hat.items():
            Zk = Z_dict[name]
            if sp.issparse(Zk):
                mu += Zk @ uk
            elif isinstance(Zk, LinearOperator):
                # Use matvec for LinearOperator
                mu += Zk.matvec(uk)
            else:
                # Dense array
                mu += Zk @ uk

        e = y - mu
        ss_e = float(e @ e)

        # 6) Compute block sums of squares: u_k' u_k
        ss_u = {name: float(uk @ uk) for name, uk in u_hat.items()}

        # 7) ED-based closed-form variance component updates
        theta_new = dict(theta)

        # Residual variance: σ²_ε = (e' e) / (n - rank(X))
        denom_eps = max(n - rank_X, options.safeguard_min)
        theta_new["eps"] = max(ss_e / denom_eps, options.safeguard_min)

        # Random effect variances: σ²_k = (u_k' u_k) / ED_k
        for name in ss_u:
            denom_k = max(ed_k[name], options.safeguard_min)
            theta_new[name] = max(ss_u[name] / denom_k, options.safeguard_min)

        # 8) Convergence check: maximum relative change across all components
        diffs = []
        for k in theta:
            num = abs(theta_new[k] - theta[k])
            den = max(abs(theta[k]), 1.0)
            rel_change_k = num / den
            diffs.append(rel_change_k)

        rel_change = max(diffs) if diffs else 0.0

        # Log iteration diagnostics
        log.append({
            "iter": it,
            "sigma2": dict(theta),
            "sigma2_new": dict(theta_new),
            "ed": dict(ed_k),
            "ed_resid": ed_resid,
            "ss_e": ss_e,
            "ss_u": dict(ss_u),
            "rel_change": rel_change,
        })

        if options.verbose:
            print(f"[REML iter {it:2d}] rel_change={rel_change:.3e} "
                  f"σ²_ε={theta_new['eps']:.4f} ed_resid={ed_resid:.2f}")
            for name in sorted([k for k in theta_new.keys() if k != 'eps']):
                print(f"  {name:15s}: σ²={theta_new[name]:.4f} ED={ed_k.get(name, 0):.2f}")

        # Update theta for next iteration
        theta = theta_new

        # Check convergence
        if rel_change < options.tol_rel:
            if options.verbose:
                print(f"[REML] Converged at iteration {it + 1}")
            return REMLResult(
                beta=beta_hat,
                u=u_hat,
                sigma2=theta,
                ed=ed_k,
                ed_residual=ed_resid,
                n_iter=it + 1,
                converged=True,
                log=log
            )

    # Max iterations reached without convergence
    if options.verbose:
        print(f"[REML] Max iterations ({options.max_iter}) reached without convergence")

    return REMLResult(
        beta=beta_hat,
        u=u_hat,
        sigma2=theta,
        ed=ed_k,
        ed_residual=ed_resid,
        n_iter=options.max_iter,
        converged=False,
        log=log
    )
