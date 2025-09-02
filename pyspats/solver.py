"""
SAP (Separation of Anisotropic Penalties) algorithm solver.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Dict, List, Tuple, Any


class SAP_solver:
    """
    SAP (Separation of Anisotropic Penalties) algorithm for efficient
    estimation in multidimensional P-spline models.
    
    Based on Rodriguez-Alvarez et al. (2015) "Fast smoothing parameter 
    separation in multidimensional generalized P-splines: the SAP algorithm"
    """
    
    def __init__(self, tolerance: float = 1e-6, max_iter: int = 100):
        self.tolerance = tolerance
        self.max_iter = max_iter
    
    def solve(
        self,
        X: np.ndarray,
        Z: np.ndarray, 
        y: np.ndarray,
        weights: np.ndarray,
        penalties: List[sparse.csr_matrix],
        lambda_init: np.ndarray
    ) -> Dict[str, Any]:
        """
        Solve mixed model equations using SAP algorithm.
        
        Parameters
        ----------
        X : np.ndarray
            Fixed effects design matrix
        Z : np.ndarray  
            Random effects design matrix
        y : np.ndarray
            Response vector
        weights : np.ndarray
            Weight vector
        penalties : list of sparse matrices
            Penalty matrices for each random component
        lambda_init : np.ndarray
            Initial variance component values
            
        Returns
        -------
        dict
            Solution containing coefficients and variance components
        """
        n_obs, p_fixed = X.shape
        n_random = Z.shape[1] if Z.size > 0 else 0
        n_penalties = len(penalties)
        
        # Initialize
        lambda_vec = lambda_init.copy()
        beta = np.zeros(p_fixed)
        u = np.zeros(n_random) if n_random > 0 else np.array([])
        
        # Weight matrix
        W = sparse.diags(weights)
        
        # Precompute fixed quantities
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ y
        
        if n_random > 0:
            XtWZ = X.T @ W @ Z
            ZtWX = Z.T @ W @ X
            ZtWZ = Z.T @ W @ Z
            ZtWy = Z.T @ W @ y
        
        # Main SAP iteration
        for iteration in range(self.max_iter):
            lambda_old = lambda_vec.copy()
            
            if n_random > 0:
                # Construct combined penalty matrix
                G_inv = self._construct_penalty_matrix(penalties, lambda_vec[1:])
                
                # Mixed model equations
                # [X'WX   X'WZ ] [beta] = [X'Wy]
                # [Z'WX  Z'WZ+G] [u   ]   [Z'Wy]
                
                coeff_matrix = sparse.bmat([
                    [XtWX, XtWZ],
                    [ZtWX, ZtWZ + G_inv]
                ]).tocsr()
                
                rhs = np.concatenate([XtWy, ZtWy])
                
                # Solve system
                try:
                    solution = spsolve(coeff_matrix, rhs)
                    beta = solution[:p_fixed]
                    u = solution[p_fixed:]
                except:
                    # Fallback to dense solver
                    solution = np.linalg.solve(coeff_matrix.toarray(), rhs)
                    beta = solution[:p_fixed]
                    u = solution[p_fixed:]
            else:
                # Only fixed effects
                beta = np.linalg.solve(XtWX, XtWy)
                u = np.array([])
            
            # Update variance components
            lambda_vec = self._update_variance_components(
                X, Z, y, weights, beta, u, penalties, lambda_vec
            )
            
            # Check convergence
            if np.allclose(lambda_vec, lambda_old, rtol=self.tolerance):
                break
        
        # Compute effective dimensions and other diagnostics
        diagnostics = self._compute_diagnostics(
            X, Z, weights, penalties, lambda_vec, beta, u
        )
        
        return {
            'beta': beta,
            'u': u,
            'lambda': lambda_vec,
            'iterations': iteration + 1,
            'converged': iteration < self.max_iter - 1,
            'diagnostics': diagnostics
        }
    
    def _construct_penalty_matrix(
        self, 
        penalties: List[sparse.csr_matrix], 
        lambda_vec: np.ndarray
    ) -> sparse.csr_matrix:
        """Construct combined penalty matrix."""
        if len(penalties) == 0:
            return sparse.csr_matrix((0, 0))
        
        # Assume penalties apply to consecutive blocks
        G_inv_blocks = []
        for i, penalty in enumerate(penalties):
            G_inv_blocks.append(lambda_vec[i] * penalty)
        
        return sparse.block_diag(G_inv_blocks)
    
    def _update_variance_components(
        self,
        X: np.ndarray,
        Z: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        beta: np.ndarray,
        u: np.ndarray,
        penalties: List[sparse.csr_matrix],
        lambda_old: np.ndarray
    ) -> np.ndarray:
        """Update variance components using REML estimates."""
        
        # Residuals
        fitted = X @ beta
        if len(u) > 0:
            fitted += Z @ u
        residuals = y - fitted
        
        # Residual sum of squares
        rss = np.sum(weights * residuals**2)
        
        # Update dispersion parameter
        df_residual = len(y) - X.shape[1]
        psi_new = rss / max(df_residual, 1)
        
        # Update variance components for random effects
        lambda_new = np.zeros(len(lambda_old))
        lambda_new[0] = psi_new
        
        if len(u) > 0 and len(penalties) > 0:
            # Update each variance component
            start_idx = 0
            for i, penalty in enumerate(penalties):
                block_size = penalty.shape[0]
                u_block = u[start_idx:start_idx + block_size]
                
                # Quadratic form u'Pu
                quadratic_form = u_block.T @ penalty @ u_block
                
                # Effective degrees of freedom (simplified estimate)
                # In full implementation, this would use trace(hat matrix)
                eff_df = max(block_size * 0.5, 1)  # Simplified
                
                # Variance component estimate
                lambda_new[i + 1] = max(quadratic_form / eff_df, 1e-8)
                
                start_idx += block_size
        
        return lambda_new
    
    def _compute_diagnostics(
        self,
        X: np.ndarray,
        Z: np.ndarray, 
        weights: np.ndarray,
        penalties: List[sparse.csr_matrix],
        lambda_vec: np.ndarray,
        beta: np.ndarray,
        u: np.ndarray
    ) -> Dict[str, Any]:
        """Compute model diagnostics."""
        
        diagnostics = {}
        
        # Effective dimensions (simplified)
        diagnostics['eff_dim_fixed'] = X.shape[1]
        
        if len(u) > 0:
            # For random effects, effective dimension < number of parameters
            start_idx = 0
            eff_dims_random = []
            
            for i, penalty in enumerate(penalties):
                block_size = penalty.shape[0]
                # Simplified effective dimension calculation
                eff_dim = max(block_size * 0.5, 1)
                eff_dims_random.append(eff_dim)
                start_idx += block_size
                
            diagnostics['eff_dim_random'] = eff_dims_random
            diagnostics['eff_dim_total'] = diagnostics['eff_dim_fixed'] + sum(eff_dims_random)
        else:
            diagnostics['eff_dim_random'] = []
            diagnostics['eff_dim_total'] = diagnostics['eff_dim_fixed']
        
        return diagnostics