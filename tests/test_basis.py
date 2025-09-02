"""
Test cases for basis construction functions.
"""

import pytest
import numpy as np
import pandas as pd
from scipy import sparse

import sys
import os
# Add parent directory to path to find pyspats package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspats.basis import (
    bspline_basis, construct_knots, penalty_matrix,
    construct_2d_pspline, construct_design_matrix
)


class TestBSplineBasis:
    """Test B-spline basis construction."""
    
    def test_basic_bspline_basis(self):
        """Test basic B-spline basis construction."""
        x = np.linspace(0, 10, 50)
        knots = construct_knots(x, nseg=5, degree=3)
        basis = bspline_basis(x, knots, degree=3)
        
        assert basis.shape[0] == len(x)
        assert basis.shape[1] == len(knots) - 3 - 1  # n_basis = n_knots - degree - 1
        
        # Check non-negativity
        assert np.all(basis >= 0)
        
        # Check partition of unity (approximately)
        row_sums = np.sum(basis, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-10)
        
    def test_different_degrees(self):
        """Test B-spline basis with different degrees."""
        x = np.linspace(0, 5, 30)
        
        for degree in [1, 2, 3]:
            knots = construct_knots(x, nseg=4, degree=degree)
            basis = bspline_basis(x, knots, degree=degree)
            
            assert basis.shape[1] == len(knots) - degree - 1
            assert np.all(basis >= 0)
            
    def test_knot_construction(self):
        """Test knot sequence construction."""
        x = np.array([1, 2, 3, 4, 5])
        knots = construct_knots(x, nseg=3, degree=3)
        
        # Check knot properties
        assert len(knots) == 3 + 1 + 2 * 3  # nseg + 1 + 2 * degree
        assert knots[0] < x.min()  # Extended knots
        assert knots[-1] > x.max()
        assert np.all(np.diff(knots) >= 0)  # Non-decreasing
        
    def test_penalty_matrix_construction(self):
        """Test penalty matrix construction."""
        n_basis = 10
        
        # Test different orders
        for order in [1, 2, 3]:
            P = penalty_matrix(n_basis, order)
            
            assert P.shape == (n_basis, n_basis)
            assert sparse.issparse(P)
            
            # Should be positive semi-definite
            P_dense = P.toarray()
            eigenvals = np.linalg.eigvals(P_dense)
            assert np.all(eigenvals >= -1e-10)  # Non-negative (allowing numerical error)
            
        # Order 0 should be identity
        P0 = penalty_matrix(n_basis, order=0)
        I = sparse.eye(n_basis)
        np.testing.assert_array_equal(P0.toarray(), I.toarray())


class Test2DPSpline:
    """Test 2D P-spline construction."""
    
    def test_2d_pspline_construction(self):
        """Test 2D P-spline basis and penalty construction."""
        # Create grid data
        x = np.repeat(np.arange(1, 6), 4)  # [1,1,1,1,2,2,2,2,...]
        y = np.tile(np.arange(1, 5), 5)   # [1,2,3,4,1,2,3,4,...]
        
        basis_2d, penalties = construct_2d_pspline(
            x, y, nseg=(3, 3), degree=3, penalty_order=2
        )
        
        assert basis_2d.shape[0] == len(x)
        assert len(penalties) == 2  # x and y direction penalties
        
        # Check penalty matrices
        for P in penalties:
            assert sparse.issparse(P)
            assert P.shape[0] == P.shape[1]
            assert P.shape[0] == basis_2d.shape[1]
            
    def test_different_nseg(self):
        """Test with different number of segments."""
        x = np.random.uniform(0, 10, 50)
        y = np.random.uniform(0, 8, 50)
        
        for nseg in [(5, 5), (3, 7), (10, 4)]:
            basis_2d, penalties = construct_2d_pspline(
                x, y, nseg=nseg, degree=3
            )
            
            assert basis_2d.shape[0] == len(x)
            assert len(penalties) == 2
            
    def test_tensor_product_properties(self):
        """Test tensor product basis properties."""
        # Simple regular grid
        x = np.array([1, 1, 2, 2, 3, 3])
        y = np.array([1, 2, 1, 2, 1, 2])
        
        basis_2d, _ = construct_2d_pspline(x, y, nseg=(2, 2), degree=1)
        
        # For B-splines: n_basis = n_knots - degree - 1
        # n_knots = nseg + 1 + 2*degree = 2 + 1 + 2*1 = 5
        # n_basis = 5 - 1 - 1 = 3
        expected_n_basis = 3 * 3  # 3 basis functions in each dimension
        assert basis_2d.shape[1] == expected_n_basis
        
    def test_penalty_directions(self):
        """Test penalty matrices for different directions."""
        x = np.linspace(0, 5, 20)
        y = np.linspace(0, 3, 20)
        
        _, penalties = construct_2d_pspline(x, y, nseg=(4, 3))
        
        P_x, P_y = penalties
        
        # Both should be positive semi-definite
        for P in [P_x, P_y]:
            eigs = np.linalg.eigvals(P.toarray())
            assert np.all(eigs >= -1e-10)


class TestDesignMatrixConstruction:
    """Test design matrix construction."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.data = pd.DataFrame({
            'genotype': pd.Categorical(['A', 'B', 'C', 'A', 'B'] * 10),
            'treatment': pd.Categorical(['T1', 'T2'] * 25),
            'block': pd.Categorical(['B1', 'B2', 'B3'] * 17)[:50],
            'col': np.tile(np.arange(1, 11), 5),
            'row': np.repeat(np.arange(1, 6), 10),
            'continuous_var': np.random.normal(0, 1, 50)
        })
        
    def test_fixed_genotype_design(self):
        """Test design matrix with fixed genotype effects."""
        result = construct_design_matrix(
            genotype='genotype',
            spatial_coords=('col', 'row'),
            fixed_vars=['treatment'],
            random_vars=['block'],
            data=self.data,
            genotype_as_random=False
        )
        
        X = result['X']
        Z = result['Z']
        
        # Check dimensions
        assert X.shape[0] == len(self.data)
        assert Z.shape[0] == len(self.data)
        
        # X should include intercept, genotype (2 levels after dropping first), treatment (1 level after dropping first)
        # Z should include spatial basis and block effects
        assert X.shape[1] >= 3  # At least intercept + some fixed effects
        assert Z.shape[1] > 0   # Should have random effects
        
    def test_random_genotype_design(self):
        """Test design matrix with random genotype effects."""
        result = construct_design_matrix(
            genotype='genotype',
            spatial_coords=('col', 'row'),
            fixed_vars=['treatment'],
            random_vars=['block'],
            data=self.data,
            genotype_as_random=True
        )
        
        X = result['X']
        Z = result['Z']
        
        # With random genotype, X should have fewer columns
        assert X.shape[1] >= 2  # Intercept + treatment
        assert Z.shape[1] > 0   # Genotype + spatial + block
        
    def test_continuous_fixed_effects(self):
        """Test continuous fixed effects."""
        result = construct_design_matrix(
            genotype='genotype',
            spatial_coords=('col', 'row'),
            fixed_vars=['continuous_var'],
            random_vars=[],
            data=self.data
        )
        
        X = result['X']
        
        # Should handle continuous variable
        assert X.shape[1] >= 3  # intercept + genotype + continuous
        
    def test_penalty_matrices(self):
        """Test penalty matrix construction."""
        result = construct_design_matrix(
            genotype='genotype',
            spatial_coords=('col', 'row'),
            fixed_vars=[],
            random_vars=['block'],
            data=self.data
        )
        
        penalties = result['penalties']
        
        # Should have spatial penalties (2) + block penalty (1)
        assert len(penalties) >= 3
        
        # All should be sparse matrices
        for P in penalties:
            assert sparse.issparse(P)
            
    def test_empty_effects(self):
        """Test with minimal effects."""
        result = construct_design_matrix(
            genotype='genotype',
            spatial_coords=('col', 'row'),
            fixed_vars=[],
            random_vars=[],
            data=self.data
        )
        
        X = result['X']
        Z = result['Z']
        
        # Should still work with minimal specification
        assert X.shape[1] >= 1  # At least intercept
        assert Z.shape[1] > 0   # At least spatial


class TestEdgeCases:
    """Test edge cases in basis construction."""
    
    def test_single_point(self):
        """Test with single spatial point."""
        x = np.array([5.0])
        y = np.array([3.0])
        
        # Should handle gracefully or raise informative error
        try:
            basis_2d, penalties = construct_2d_pspline(x, y)
            assert basis_2d.shape[0] == 1
        except (ValueError, np.linalg.LinAlgError):
            # Expected for degenerate cases
            pass
            
    def test_collinear_points(self):
        """Test with collinear points."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 1, 1, 1, 1])  # All same y
        
        basis_2d, penalties = construct_2d_pspline(x, y, nseg=(2, 2))
        
        # Should still construct basis
        assert basis_2d.shape[0] == 5
        
    def test_very_small_nseg(self):
        """Test with very small number of segments."""
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        
        basis_2d, penalties = construct_2d_pspline(x, y, nseg=(1, 1))
        
        # Should work with minimal segments
        assert basis_2d.shape[0] == 10
        
    def test_large_degree(self):
        """Test with large spline degree."""
        x = np.linspace(0, 10, 20)
        knots = construct_knots(x, nseg=10, degree=5)
        
        # Should construct knots appropriately
        assert len(knots) == 10 + 1 + 2 * 5


if __name__ == '__main__':
    pytest.main([__file__])