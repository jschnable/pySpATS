"""
Core SpATS implementation for spatial analysis of field trials.
"""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve, LinearOperator
from scipy.linalg import cholesky, solve_triangular
from scipy.stats import norm
import warnings
from typing import Optional, Union, List, Dict, Tuple, Any

from .control import SpATSControl
from .basis import construct_2d_pspline, construct_design_matrix
from .families import Family, gaussian
from .solver import SAP_solver
from .utils import interpret_formula, get_heritability
from . import plotting
from .psanova_basis import build_psanova_design
from .ed_selected_inverse import BlockInfo
from .spatial.block_operator import ConcatenatedBlockOperator, BlockLinearOperator


class SpATS:
    """
    Spatial Analysis of Field Trials with Splines
    
    Fits a (generalised) linear mixed model with spatial trends modelled 
    using two-dimensional P-splines.
    
    Parameters
    ----------
    response : str
        Name of the response variable column in data
    genotype : str
        Name of the genotype/variety column in data (must be categorical)
    spatial : tuple or str
        Spatial coordinates specification as (x_coord, y_coord) or formula
    genotype_as_random : bool, default=False
        Whether to treat genotype as random effect
    geno_decomp : str, optional
        Factor variable for genotype grouping (when genotype is random)
    fixed : list of str, optional
        Fixed effect variables
    random : list of str, optional
        Random effect variables (must be categorical)
    data : pd.DataFrame
        Input data containing all variables
    family : Family, default=gaussian()
        Distribution family and link function
    offset : array-like, optional
        Known offset to include in linear predictor
    weights : array-like, optional
        Observation weights (default: all ones)
    control : SpATSControl, optional
        Algorithm control parameters
        
    Attributes
    ----------
    fitted_values : np.ndarray
        Fitted values from the model
    residuals : np.ndarray
        Deviance residuals
    coefficients : np.ndarray
        Estimated coefficients (fixed and random)
    var_comp : dict
        Variance component estimates
    effective_dim : dict
        Effective dimensions for each model component
    deviance : float
        Model deviance at convergence
    """
    
    def __init__(
        self,
        response: str,
        genotype: str,
        spatial: Union[Tuple[str, str], str],
        data: pd.DataFrame,
        genotype_as_random: bool = False,
        geno_decomp: Optional[str] = None,
        fixed: Optional[List[str]] = None,
        random: Optional[List[str]] = None,
        family: Family = gaussian(),
        offset: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        control: Optional[SpATSControl] = None
    ):
        self.response = response
        self.genotype = genotype
        self.spatial = spatial
        self.genotype_as_random = genotype_as_random
        self.geno_decomp = geno_decomp
        self.fixed = fixed or []
        self.random = random or []
        self.family = family
        self.control = control or SpATSControl()
        
        # Validate inputs first
        self._validate_inputs(data)
        self.data = data.copy()
        
        # Set up weights and offset
        n_obs = len(self.data)
        self.weights = np.ones(n_obs) if weights is None else np.asarray(weights)
        self.offset = np.zeros(n_obs) if offset is None else np.asarray(offset)
        
        # Handle missing values
        self._handle_missing_data()
        
        # Check and clean covariates
        self._check_and_clean_covariates()
        
        # Fit the model
        self._fit_model()
    
    def _validate_inputs(self, data):
        """Validate input parameters and data."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame")
            
        required_cols = [self.response, self.genotype]
        if isinstance(self.spatial, tuple):
            required_cols.extend(self.spatial)
        required_cols.extend(self.fixed)
        required_cols.extend(self.random)
        if self.geno_decomp:
            required_cols.append(self.geno_decomp)
            
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in data: {missing_cols}")
            
        # Check that genotype is categorical
        if not pd.api.types.is_categorical_dtype(data[self.genotype]) and \
           not data[self.genotype].dtype == 'object':
            warnings.warn(f"Converting {self.genotype} to categorical")
            data[self.genotype] = data[self.genotype].astype('category')
            
        # Check random effects are categorical
        for col in self.random:
            if not pd.api.types.is_categorical_dtype(data[col]) and \
               not data[col].dtype == 'object':
                warnings.warn(f"Converting {col} to categorical")
                data[col] = data[col].astype('category')
    
    def _check_and_clean_covariates(self):
        """
        Check covariates for sufficient variation and remove problematic ones.
        Provides informative messages about removed factors.
        """
        import warnings
        
        # Check fixed effects
        fixed_to_remove = []
        
        for factor in self.fixed[:]:  # Create copy to modify during iteration
            if factor in self.data.columns:
                # Get valid (non-NA) values for this factor
                valid_values = self.data[factor].dropna()
                
                # Check for sufficient levels
                if pd.api.types.is_categorical_dtype(valid_values) or valid_values.dtype == 'object':
                    unique_levels = valid_values.nunique()
                    if unique_levels < 2:
                        warnings.warn(f"Fixed effect '{factor}' has insufficient levels ({unique_levels}). Removing from model.")
                        fixed_to_remove.append(factor)
                        continue
                        
                # Check for zero variance (for numeric factors)
                elif pd.api.types.is_numeric_dtype(valid_values):
                    if valid_values.std() == 0:
                        warnings.warn(f"Fixed effect '{factor}' has zero variance. Removing from model.")
                        fixed_to_remove.append(factor)
                        continue
                        
                # Check if factor becomes problematic after subsetting to valid observations
                if hasattr(self, 'valid_obs'):
                    subset_values = self.data[factor][self.valid_obs].dropna()
                    if len(subset_values) == 0:
                        warnings.warn(f"Fixed effect '{factor}' has no valid observations. Removing from model.")
                        fixed_to_remove.append(factor)
                        continue
                    elif pd.api.types.is_categorical_dtype(subset_values) or subset_values.dtype == 'object':
                        subset_levels = subset_values.nunique()
                        if subset_levels < 2:
                            warnings.warn(f"Fixed effect '{factor}' has insufficient levels after subsetting ({subset_levels}). Removing from model.")
                            fixed_to_remove.append(factor)
                            continue
        
        # Remove problematic fixed effects
        for factor in fixed_to_remove:
            self.fixed.remove(factor)
            
        # Check random effects
        random_to_remove = []
        
        for factor in self.random[:]:  # Create copy to modify during iteration
            if factor in self.data.columns:
                # Get valid (non-NA) values for this factor
                valid_values = self.data[factor].dropna()
                
                # Random effects must be categorical
                if not (pd.api.types.is_categorical_dtype(valid_values) or valid_values.dtype == 'object'):
                    warnings.warn(f"Random effect '{factor}' is not categorical. Converting to categorical.")
                    self.data[factor] = self.data[factor].astype('category')
                    valid_values = self.data[factor].dropna()
                
                # Check for sufficient levels
                unique_levels = valid_values.nunique()
                if unique_levels < 2:
                    warnings.warn(f"Random effect '{factor}' has insufficient levels ({unique_levels}). Removing from model.")
                    random_to_remove.append(factor)
                    continue
                    
                # Check if factor becomes problematic after subsetting to valid observations  
                if hasattr(self, 'valid_obs'):
                    subset_values = self.data[factor][self.valid_obs].dropna()
                    if len(subset_values) == 0:
                        warnings.warn(f"Random effect '{factor}' has no valid observations. Removing from model.")
                        random_to_remove.append(factor)
                        continue
                    else:
                        subset_levels = subset_values.nunique()
                        if subset_levels < 2:
                            warnings.warn(f"Random effect '{factor}' has insufficient levels after subsetting ({subset_levels}). Removing from model.")
                            random_to_remove.append(factor)
                            continue
        
        # Remove problematic random effects
        for factor in random_to_remove:
            self.random.remove(factor)
            
        # Inform user of final model specification
        if fixed_to_remove or random_to_remove:
            print("SpATS model updated:")
            print(f"  Fixed effects: {self.fixed if self.fixed else 'None'}")
            print(f"  Random effects: {self.random if self.random else 'None'}")
    
    def _handle_missing_data(self):
        """Handle missing values in predictors and response."""
        # Identify rows with missing predictors
        predictor_cols = [self.genotype]
        if isinstance(self.spatial, tuple):
            predictor_cols.extend(self.spatial)
        predictor_cols.extend(self.fixed)
        predictor_cols.extend(self.random)
        if self.geno_decomp:
            predictor_cols.append(self.geno_decomp)
            
        missing_predictors = self.data[predictor_cols].isnull().any(axis=1)
        missing_response = self.data[self.response].isnull()
        
        # Update weights to handle missing data
        self.weights = self.weights * (~missing_predictors) * (~missing_response)
        
        # Store original indices
        self.valid_obs = ~missing_predictors
        self.n_obs = np.sum(self.weights > 0)
        
    def _fit_model(self):
        """Main model fitting procedure."""
        if self.control.monitoring:
            print("Starting SpATS model fitting...")
            
        # Construct design matrices
        design_info = self._construct_design_matrices()
        
        # Initialize parameters
        y = self.data[self.response].values[self.valid_obs]
        n_params = design_info['X'].shape[1] + design_info['Z'].shape[1]
        
        # Initialize coefficients and variance components
        beta = np.zeros(design_info['X'].shape[1])
        u = np.zeros(design_info['Z'].shape[1])
        lambda_params = np.ones(len(design_info['penalty_matrices']) + 1)
        
        # Initialize linear predictor and mean
        eta = design_info['X'] @ beta + design_info['Z'] @ u + self.offset[self.valid_obs]
        mu = self.family.inverse_link(eta)
        
        # Main iteration loop
        deviance_old = np.inf
        
        for iteration in range(self.control.max_iter):
            # Update working variables
            mu_eta = self.family.d_inverse_link(eta)
            var_mu = self.family.variance(mu)
            
            # Working response and weights
            z = eta + (y - mu) / mu_eta
            w = (mu_eta ** 2) / var_mu * self.weights[self.valid_obs]
            
            # Solve mixed model equations
            solution = self._solve_mixed_model_equations(
                design_info, z, w, lambda_params
            )
            
            beta = solution['beta']
            u = solution['u'] 
            lambda_params = solution['lambda']
            
            # Update linear predictor
            eta = design_info['X'] @ beta + design_info['Z'] @ u + self.offset[self.valid_obs]
            mu = self.family.inverse_link(eta)
            
            # Check convergence
            deviance = self._compute_deviance(y, mu, w)
            
            if self.control.monitoring:
                print(f"Iteration {iteration + 1}: Deviance = {deviance:.6f}")
                
            if abs(deviance_old - deviance) < self.control.tolerance:
                break
                
            deviance_old = deviance
            
            # For Gaussian with identity link, converge after one iteration
            if self.family.family == 'gaussian' and self.family.link == 'identity':
                break
        
        # Store results
        self._store_results(design_info, beta, u, lambda_params, mu, deviance, iteration + 1)
        
    def _construct_design_matrices(self) -> Dict[str, Any]:
        """Construct design matrices for fixed and random effects using PS-ANOVA decomposition."""
        valid_data = self.data[self.valid_obs]

        # PS-ANOVA decomposition for spatial component
        # Fixed polynomial: [1, r, c]
        # Random smooths: row-smooth, col-smooth, interaction (orthogonal to polynomial)
        spatial_blocks = []
        n_geno_fixed = None
        n_geno_random = None

        if isinstance(self.spatial, tuple):
            x_coord, y_coord = self.spatial
            r_vals = valid_data[x_coord].values
            c_vals = valid_data[y_coord].values

            # Build PS-ANOVA design with explicit polynomial fixed effects
            # and orthogonal random smooths
            X_poly, Z_r, Z_c, Z_rc, spatial_blocks = build_psanova_design(
                r_vals, c_vals,
                nkr=10,  # Default knots for row
                nkc=10,  # Default knots for column
                degree=3
            )

            # Fixed effects: start with polynomial part
            X_parts = [X_poly]
        else:
            # No spatial component - just intercept
            X_parts = [np.ones((len(valid_data), 1))]

        # Genotype (if fixed)
        if not self.genotype_as_random:
            geno_dummies = pd.get_dummies(valid_data[self.genotype], drop_first=True)
            X_parts.append(geno_dummies.values)
            # Track number of genotypes (including baseline)
            n_geno_fixed = valid_data[self.genotype].nunique()

        # Other fixed effects
        for var in self.fixed:
            if valid_data[var].dtype in ['object', 'category']:
                dummies = pd.get_dummies(valid_data[var], drop_first=True)
                X_parts.append(dummies.values)
            else:
                X_parts.append(valid_data[var].values.reshape(-1, 1))

        X = np.hstack(X_parts) if X_parts else np.ones((len(valid_data), 1))

        # Random effects design matrix
        Z_parts = []
        penalty_matrices = []
        block_info = []  # Track block metadata for ED computation

        # Add spatial random smooths (already whitened, so penalty = identity)
        if isinstance(self.spatial, tuple):
            current_idx = 0
            for Z_block, block in zip([Z_r, Z_c, Z_rc], spatial_blocks):
                if Z_block.shape[1] > 0:
                    Z_parts.append(Z_block)
                    # Penalty is identity (already whitened)
                    penalty_matrices.append(sparse.eye(Z_block.shape[1]))
                    # Update block indices to account for global position
                    block_info.append(BlockInfo(
                        name=block.name,
                        start=current_idx,
                        stop=current_idx + Z_block.shape[1],
                        is_random=True
                    ))
                    current_idx += Z_block.shape[1]

        # Genotype (if random)
        if self.genotype_as_random:
            geno_dummies = pd.get_dummies(valid_data[self.genotype], drop_first=False)
            Z_parts.append(geno_dummies.values)
            # Add identity penalty for genotype random effects
            n_geno_random = geno_dummies.shape[1]
            penalty_matrices.append(sparse.eye(n_geno_random))
            block_info.append(BlockInfo(
                name='genotype',
                start=current_idx,
                stop=current_idx + n_geno_random,
                is_random=True
            ))
            current_idx += n_geno_random

        # Other random effects
        for var in self.random:
            dummies = pd.get_dummies(valid_data[var], drop_first=False)
            n_levels = dummies.shape[1]
            Z_parts.append(dummies.values)
            # Add identity penalty
            penalty_matrices.append(sparse.eye(n_levels))
            block_info.append(BlockInfo(
                name=var,
                start=current_idx,
                stop=current_idx + n_levels,
                is_random=True
            ))
            current_idx += n_levels

        # Handle Z matrix construction - may contain LinearOperators from Kronecker path
        if Z_parts:
            # Check if any parts are LinearOperators
            has_linear_operators = any(isinstance(z, LinearOperator) for z in Z_parts)

            if has_linear_operators:
                # Use ConcatenatedBlockOperator for mixed dense/LinearOperator blocks
                wrapped_blocks = []
                for z_block, block in zip(Z_parts, block_info):
                    wrapped_blocks.append(BlockLinearOperator(z_block, block.name))
                Z = ConcatenatedBlockOperator(wrapped_blocks, block_info)
            else:
                # All dense - use normal hstack
                Z = np.hstack(Z_parts)
        else:
            Z = np.zeros((len(valid_data), 0))

        return {
            'X': X,
            'Z': Z,
            'penalty_matrices': penalty_matrices,
            'block_info': block_info,  # Add block metadata
            'valid_data': valid_data,
            'n_geno_fixed': n_geno_fixed,
            'n_geno_random': n_geno_random
        }
    
    def _solve_mixed_model_equations(self, design_info, z, w, lambda_params):
        """Solve mixed model equations using SAP algorithm."""
        X = design_info['X']
        Z = design_info['Z']
        penalties = design_info['penalty_matrices']

        # Convert LinearOperator to dense array if needed (for compatibility with simplified solver)
        if isinstance(Z, LinearOperator) or isinstance(Z, ConcatenatedBlockOperator):
            # Materialize the full matrix for this simple solver
            # (The REML optimizer path handles LinearOperators efficiently)
            n_obs = X.shape[0]
            n_random = Z.shape[1]
            Z_dense = np.zeros((n_obs, n_random))
            for j in range(n_random):
                e_j = np.zeros(n_random)
                e_j[j] = 1.0
                Z_dense[:, j] = Z @ e_j
            Z = Z_dense

        # Weight matrices
        W = sparse.diags(w)
        
        # Construct penalty matrix
        G_inv = sparse.block_diag([lambda_params[i+1] * P for i, P in enumerate(penalties)])
        
        # Mixed model equations
        # [X'WX   X'WZ] [beta] = [X'Wz]
        # [Z'WX  Z'WZ + G_inv] [u]     [Z'Wz]
        
        XtWX = X.T @ W @ X
        XtWZ = X.T @ W @ Z
        ZtWX = Z.T @ W @ X  
        ZtWZ = Z.T @ W @ Z
        
        XtWz = X.T @ W @ z
        ZtWz = Z.T @ W @ z
        
        # Construct coefficient matrix
        if Z.shape[1] > 0:
            coeff_matrix = sparse.bmat([
                [XtWX, XtWZ],
                [ZtWX, ZtWZ + G_inv]
            ]).tocsr()
            rhs = np.concatenate([XtWz, ZtWz])
        else:
            coeff_matrix = XtWX
            rhs = XtWz
            
        # Solve system
        try:
            solution = spsolve(coeff_matrix, rhs)
        except:
            # Fallback to dense solver with regularization
            try:
                coeff_dense = coeff_matrix.toarray()
                # Add small regularization to diagonal
                np.fill_diagonal(coeff_dense, coeff_dense.diagonal() + 1e-8)
                solution = np.linalg.solve(coeff_dense, rhs)
            except:
                # Last resort: use least squares
                solution, _, _, _ = np.linalg.lstsq(coeff_matrix.toarray(), rhs, rcond=None)
            
        if Z.shape[1] > 0:
            beta = solution[:X.shape[1]]
            u = solution[X.shape[1]:]
        else:
            beta = solution
            u = np.array([])
            
        # Update variance components (simplified REML estimation)
        new_lambda = self._update_variance_components(
            design_info, beta, u, z, w, lambda_params
        )
        
        return {
            'beta': beta,
            'u': u, 
            'lambda': new_lambda
        }
    
    def _update_variance_components(self, design_info, beta, u, z, w, lambda_params):
        """Update variance components using REML."""
        # Simplified variance component update
        # In practice, this would use the full REML equations

        X = design_info['X']
        Z = design_info['Z']

        # Convert LinearOperator to dense array if needed
        if isinstance(Z, LinearOperator) or isinstance(Z, ConcatenatedBlockOperator):
            n_obs = X.shape[0]
            n_random = Z.shape[1]
            Z_dense = np.zeros((n_obs, n_random))
            for j in range(n_random):
                e_j = np.zeros(n_random)
                e_j[j] = 1.0
                Z_dense[:, j] = Z @ e_j
            Z = Z_dense

        # Residuals
        residuals = z - X @ beta - Z @ u
        
        # Residual sum of squares
        rss = np.sum(w * residuals**2)
        
        # Degrees of freedom
        df_residual = len(z) - X.shape[1]
        
        # Update dispersion parameter
        psi = rss / df_residual
        
        # Update variance components (simplified)
        new_lambda = lambda_params.copy()
        new_lambda[0] = psi
        
        if len(u) > 0:
            # Simple update for random effect variances
            n_penalties = len(design_info['penalty_matrices'])
            block_sizes = [P.shape[0] for P in design_info['penalty_matrices']]
            
            start_idx = 0
            for i, block_size in enumerate(block_sizes):
                u_block = u[start_idx:start_idx + block_size]
                penalty = design_info['penalty_matrices'][i]
                
                # Variance component estimate
                quadratic_form = u_block.T @ penalty @ u_block
                effective_df = block_size  # Simplified
                
                if effective_df > 0:
                    new_lambda[i + 1] = max(quadratic_form / effective_df, 1e-8)
                
                start_idx += block_size
        
        return new_lambda
    
    def _compute_deviance(self, y, mu, w):
        """Compute model deviance."""
        return self.family.deviance(y, mu, w)
    
    def _store_results(self, design_info, beta, u, lambda_params, mu, deviance, n_iter):
        """Store model fitting results."""
        # Store design matrices for component decomposition
        self._design_info = design_info
        self._beta = beta
        self._u = u

        # Fitted values for all observations
        self.fitted_values = np.full(len(self.data), np.nan)
        self.fitted_values[self.valid_obs] = mu

        # Decompose fitted values into components
        self._decompose_fitted_components(design_info, beta, u)

        # Residuals
        y_full = self.data[self.response].values
        self.residuals = np.full(len(self.data), np.nan)
        valid_idx = self.valid_obs & ~pd.isnull(y_full)
        self.residuals[valid_idx] = (y_full[valid_idx] - self.fitted_values[valid_idx])

        # Coefficients
        self.coefficients = np.concatenate([beta, u])

        # Variance components
        self.var_comp = {f'component_{i}': lambda_params[i+1]
                        for i in range(len(lambda_params)-1)}
        self.psi = lambda_params[0]

        # Model information
        self.deviance = deviance
        self.n_iterations = n_iter
        self.n_obs = self.n_obs

        # Store genotype counts for heritability calculation
        self._n_geno = design_info.get('n_geno_fixed') or design_info.get('n_geno_random')

        # Effective dimensions (simplified)
        self.effective_dim = {
            'fixed': design_info['X'].shape[1],
            'spatial': len(design_info['penalty_matrices'][0].data) if design_info['penalty_matrices'] else 0
        }

        # Calculate genotype effective dimension for fixed genotypes
        if design_info.get('n_geno_fixed') is not None:
            # For fixed genotypes, ED_geno equals number of genotype parameters
            # which is n_geno - 1 (drop_first=True in dummy coding) plus intercept contribution
            # Simplification: ED_geno ≈ n_geno - 1
            self._ED_geno = design_info['n_geno_fixed'] - 1
        else:
            self._ED_geno = None
    
    def _decompose_fitted_components(self, design_info, beta, u):
        """Decompose fitted values into fixed, spatial, and other random components."""
        X = design_info['X']
        Z = design_info['Z']

        # Convert LinearOperator to dense array if needed
        if isinstance(Z, LinearOperator) or isinstance(Z, ConcatenatedBlockOperator):
            n_obs = X.shape[0]
            n_random = Z.shape[1]
            Z_dense = np.zeros((n_obs, n_random))
            for j in range(n_random):
                e_j = np.zeros(n_random)
                e_j[j] = 1.0
                Z_dense[:, j] = Z @ e_j
            Z = Z_dense

        # Initialize component arrays
        self.spatial_effects = np.full(len(self.data), np.nan)
        self.random_effects = np.full(len(self.data), np.nan)
        self.fixed_effects = np.full(len(self.data), np.nan)

        # Initialize spatial centering adjustment
        spatial_mean_adjustment = 0.0

        if len(u) > 0 and Z.shape[1] > 0:
            # Decompose random effects by component
            penalty_matrices = design_info['penalty_matrices']
            
            if penalty_matrices:
                # First component is typically spatial (2D P-splines)
                spatial_size = penalty_matrices[0].shape[0]
                spatial_u = u[:spatial_size]
                spatial_Z = Z[:, :spatial_size]
                
                # Spatial effects component
                spatial_part = spatial_Z @ spatial_u
                # Center spatial effects to ensure identifiability (sum-to-zero constraint)
                spatial_mean_adjustment = np.mean(spatial_part)
                spatial_part_centered = spatial_part - spatial_mean_adjustment
                self.spatial_effects[self.valid_obs] = spatial_part_centered
                
                # Other random effects (if any)
                if len(u) > spatial_size:
                    other_u = u[spatial_size:]
                    other_Z = Z[:, spatial_size:]
                    other_part = other_Z @ other_u
                    self.random_effects[self.valid_obs] = other_part
                else:
                    self.random_effects[self.valid_obs] = 0.0
            else:
                # No spatial component, all are other random effects
                random_part = Z @ u
                self.random_effects[self.valid_obs] = random_part
                self.spatial_effects[self.valid_obs] = 0.0
        else:
            # No random effects
            self.spatial_effects[self.valid_obs] = 0.0
            self.random_effects[self.valid_obs] = 0.0
        
        # Fixed effects component (adjusted for spatial centering)
        fixed_part = X @ beta + spatial_mean_adjustment
        self.fixed_effects[self.valid_obs] = fixed_part
        
    def get_BLUEs(self):
        """
        Extract genotypic BLUEs (Best Linear Unbiased Estimates).
        
        BLUEs represent adjusted genotypic means, calculated as the average of 
        fitted values for each genotype. This matches the approach used by 
        R SpATS predict(model, which='genotype').
        
        Returns
        -------
        pd.Series
            BLUEs for each genotype
        """
        if not hasattr(self, 'fitted_values'):
            raise ValueError("Model must be fitted before extracting BLUEs")
        
        if self.genotype_as_random:
            raise ValueError("BLUEs are only available when genotype is treated as fixed effect")
        
        # Calculate BLUEs as fitted genotype means
        blues = {}
        
        for genotype in self.data[self.genotype].unique():
            # Find observations for this genotype
            geno_mask = (self.data[self.genotype] == genotype) & self.valid_obs
            
            if np.any(geno_mask):
                # BLUE = mean of fitted values for this genotype
                blues[genotype] = np.mean(self.fitted_values[geno_mask])
        
        return pd.Series(blues)
    
    
    def predict(self, newdata=None):
        """
        Make predictions from fitted model.

        Parameters
        ----------
        newdata : pd.DataFrame, optional
            New data for prediction. If None, returns fitted values.

        Returns
        -------
        np.ndarray
            Predicted values
        """
        if newdata is None:
            return self.fitted_values

        # For new data prediction, would need to reconstruct design matrices
        # This is a simplified implementation
        raise NotImplementedError("Prediction on new data not yet implemented")

    def get_heritability(self, mode: str = "generalized") -> float:
        """
        Calculate heritability from genotype effective dimension.

        Default heritability follows SpATS generalized H² = ED_geno / n_geno.
        For comparison with older results, set mode='classical' to compute
        ED_geno / (n_geno - 1).

        Parameters
        ----------
        mode : {"generalized", "classical"}, default="generalized"
            - "generalized": ED_geno / n_geno       (SpATS-style generalized H²)
            - "classical"  : ED_geno / (n_geno - 1) (legacy)

        Returns
        -------
        float
            Heritability estimate

        Raises
        ------
        ValueError
            If genotype is not treated as fixed effect or model is not fitted

        Examples
        --------
        >>> model = SpATS(response='yield', genotype='geno', spatial=('col','row'), data=data)
        >>> h2 = model.get_heritability()  # generalized (default)
        >>> h2_classical = model.get_heritability(mode='classical')  # legacy
        """
        if not hasattr(self, '_ED_geno') or self._ED_geno is None:
            raise ValueError("Heritability is only available when genotype is treated as fixed effect")

        if not hasattr(self, '_n_geno') or self._n_geno is None:
            raise ValueError("Model must be fitted before calculating heritability")

        from .utils import get_heritability
        return get_heritability(self._ED_geno, self._n_geno, mode=mode)

    @property
    def heritability(self) -> float:
        """
        Heritability estimate using generalized method (H² = ED_geno / n_geno).

        For classical heritability (ED_geno / (n_geno - 1)), use:
        model.get_heritability(mode='classical')

        Returns
        -------
        float
            Generalized heritability estimate

        Raises
        ------
        ValueError
            If genotype is not treated as fixed effect or model is not fitted
        """
        return self.get_heritability(mode="generalized")
    
    def summary(self, which="dimensions"):
        """
        Print model summary.

        Parameters
        ----------
        which : str
            Type of summary: "dimensions", "variances", or "all"
        """
        print(f"SpATS Model Summary")
        print("=" * 50)
        print(f"Response: {self.response}")
        print(f"Observations: {self.n_obs}")
        print(f"Deviance: {self.deviance:.4f}")
        print(f"Iterations: {self.n_iterations}")

        if which in ["dimensions", "all"]:
            print("\nModel Dimensions:")
            for component, dim in self.effective_dim.items():
                print(f"  {component}: {dim}")

        if which in ["variances", "all"]:
            print("\nVariance Components:")
            print(f"  Dispersion (psi): {self.psi:.6f}")
            for component, var in self.var_comp.items():
                print(f"  {component}: {var:.6f}")

    def summary_ed(self):
        """
        Print effective dimension (ED) summary for all model components.

        Effective dimensions quantify the "amount of smoothing" or complexity
        consumed by each random effect. For spatial smooths in PS-ANOVA:
        - row_smooth: ED for row-wise spatial trend
        - col_smooth: ED for column-wise spatial trend
        - interaction_smooth: ED for 2D spatial interaction (non-separable pattern)

        Higher ED indicates less smoothing (more model flexibility), while
        lower ED indicates more aggressive smoothing (simpler surface).

        Note: Current implementation uses nominal dimensions as approximations.
        For exact CHOLMOD-based EDs, use the REML optimizer path.

        Examples
        --------
        >>> model = SpATS(response='yield', genotype='geno', spatial=('col','row'), data=data)
        >>> model.summary_ed()
        """
        print("SpATS Effective Dimension Summary")
        print("=" * 60)
        print(f"Response: {self.response}")
        print(f"Observations: {self.n_obs}")
        print()

        if hasattr(self, 'effective_dim') and self.effective_dim:
            print("Effective Dimensions:")
            print("-" * 60)

            # Fixed effects
            if 'fixed' in self.effective_dim:
                print(f"  Fixed effects:                    {self.effective_dim['fixed']:>8.2f}")

            # Spatial components (from block_info if available)
            if hasattr(self, '_design_info') and 'block_info' in self._design_info:
                blocks = self._design_info['block_info']
                for block in blocks:
                    # Use block size as nominal dimension approximation
                    ed_approx = block.size
                    print(f"  {block.name:30s}  {ed_approx:>8.2f} (nominal)")
            elif 'spatial' in self.effective_dim:
                print(f"  Spatial effects:                  {self.effective_dim['spatial']:>8.2f}")

            # Genotype
            if hasattr(self, '_ED_geno') and self._ED_geno is not None:
                print(f"  Genotype:                         {self._ED_geno:>8.2f}")

            print("-" * 60)

            # Total model ED (approximate)
            total_ed = sum(self.effective_dim.values())
            if hasattr(self, '_ED_geno') and self._ED_geno is not None:
                total_ed += self._ED_geno
            print(f"  Total model ED (approx):          {total_ed:>8.2f}")

            # Residual ED estimate
            ed_resid = self.n_obs - total_ed
            print(f"  Residual ED (approx):             {ed_resid:>8.2f}")
            print()
        else:
            print("No effective dimension information available.")
            print("Model may not have been fitted yet.")
            print()

        print("Note: EDs shown are nominal dimensions (parameter counts).")
        print("For exact EDs accounting for smoothing penalties, use REML optimizer.")
    
    def plot(self, all_in_one: bool = True, figsize: Tuple[int, int] = (15, 10), 
             show: bool = True, spa_trend: str = 'raw'):
        """
        Plot SpATS model results with 6 diagnostic panels (matching R SpATS behavior).
        
        Creates the following plots:
        1. Raw data
        2. Fitted data  
        3. Residuals
        4. Spatial trend
        5. Genotypic predictions (BLUPs/BLUEs)
        6. Histogram of genotype coefficients
        
        Parameters
        ----------
        all_in_one : bool, default=True
            Whether to show all plots in one figure
        figsize : tuple, default=(15, 10)
            Figure size
        show : bool, default=True
            Whether to display the plot window
        spa_trend : str, default='raw'
            Format for spatial trend: 'raw' or 'percentage'
            
        Returns
        -------
        plt.Figure
            Matplotlib figure object
        """
        fig = plotting.plot_spats_full(self, all_in_one=all_in_one, figsize=figsize, 
                                       spa_trend=spa_trend)
        if show:
            import matplotlib.pyplot as plt
            plt.show()
        return fig
    
    def plot_spatial(self, figsize: Tuple[int, int] = (10, 6), show: bool = True):
        """Plot spatial trend only."""
        fig = plotting._plot_spatial_trend(self, figsize)
        if show:
            import matplotlib.pyplot as plt
            plt.show()
        return fig
    
    def plot_residuals(self, figsize: Tuple[int, int] = (10, 6), show: bool = True):
        """Plot residuals vs fitted values."""
        fig = plotting._plot_residuals(self, figsize)
        if show:
            import matplotlib.pyplot as plt
            plt.show()
        return fig
    
    def plot_fitted(self, figsize: Tuple[int, int] = (10, 6), show: bool = True):
        """Plot fitted vs observed values."""
        fig = plotting._plot_fitted_vs_observed(self, figsize)
        if show:
            import matplotlib.pyplot as plt
            plt.show()
        return fig
    
    def __repr__(self):
        """String representation of SpATS model."""
        return (f"SpATS(response='{self.response}', genotype='{self.genotype}', "
                f"n_obs={self.n_obs}, deviance={self.deviance:.4f})")