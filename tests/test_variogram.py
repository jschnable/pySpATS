"""
Test cases for variogram analysis.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock

import sys
import os
# Add parent directory to path to find pyspats package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspats.variogram import (
    variogram, fit_variogram_model, directional_variogram,
    Variogram, _spherical_model, _exponential_model, _gaussian_model
)
from pyspats.datasets import generate_field_trial_data
from pyspats.core import SpATS


class TestVariogramModels:
    """Test theoretical variogram models."""
    
    def test_spherical_model(self):
        """Test spherical variogram model."""
        h = np.linspace(0, 10, 50)
        nugget, sill, range_param = 2.0, 5.0, 3.0
        
        gamma = _spherical_model(h, nugget, sill, range_param)
        
        # Check properties
        assert gamma[0] == nugget  # At h=0
        assert np.all(gamma >= nugget)  # Always >= nugget
        
        # At large distances, should approach nugget + sill
        large_h_idx = h > range_param
        if np.any(large_h_idx):
            np.testing.assert_allclose(gamma[large_h_idx], nugget + sill, rtol=1e-10)
            
    def test_exponential_model(self):
        """Test exponential variogram model."""
        h = np.linspace(0, 10, 50)
        nugget, sill, range_param = 1.0, 4.0, 2.0
        
        gamma = _exponential_model(h, nugget, sill, range_param)
        
        # Check properties
        assert gamma[0] == nugget
        assert np.all(gamma >= nugget)
        
        # Should approach nugget + sill asymptotically
        assert gamma[-1] < nugget + sill
        assert gamma[-1] > nugget + 0.9 * sill  # Should be close
        
    def test_gaussian_model(self):
        """Test Gaussian variogram model."""
        h = np.linspace(0, 10, 50)
        nugget, sill, range_param = 0.5, 3.0, 2.5
        
        gamma = _gaussian_model(h, nugget, sill, range_param)
        
        # Check properties
        assert gamma[0] == nugget
        assert np.all(gamma >= nugget)
        
        # Gaussian model has very smooth transition
        assert gamma[-1] < nugget + sill


class TestVariogram:
    """Test Variogram class and empirical variogram computation."""
    
    def setup_method(self):
        """Set up test data and fitted model."""
        # Generate data with known spatial structure
        np.random.seed(42)
        self.data = generate_field_trial_data(
            n_rows=10, n_cols=8, 
            spatial_variance=50.0,
            seed=42
        )
        
        # Fit SpATS model
        self.model = SpATS(
            response='response',
            genotype='genotype',
            spatial=('col', 'row'),
            data=self.data
        )
        
    def test_variogram_computation(self):
        """Test basic variogram computation."""
        var_obj = variogram(self.model, n_bins=10)
        
        assert isinstance(var_obj, Variogram)
        assert len(var_obj.distances) > 0
        assert len(var_obj.gamma) == len(var_obj.distances)
        assert len(var_obj.n_pairs) == len(var_obj.distances)
        
        # Check properties
        assert np.all(var_obj.distances >= 0)
        assert np.all(var_obj.gamma >= 0)
        assert np.all(var_obj.n_pairs > 0)
        
    def test_variogram_max_distance(self):
        """Test variogram with maximum distance constraint."""
        max_dist = 3.0
        var_obj = variogram(self.model, max_dist=max_dist)
        
        assert np.all(var_obj.distances <= max_dist)
        
    def test_variogram_cutoff(self):
        """Test variogram with cutoff parameter."""
        cutoff = 2.5
        var_obj = variogram(self.model, cutoff=cutoff)
        
        assert np.all(var_obj.distances <= cutoff)
        
    def test_variogram_bins(self):
        """Test variogram with different number of bins."""
        for n_bins in [5, 10, 20]:
            var_obj = variogram(self.model, n_bins=n_bins)
            assert len(var_obj.distances) <= n_bins  # May be fewer due to empty bins
            
    def test_variogram_with_missing_data(self):
        """Test variogram computation with missing residuals."""
        # Introduce some NaN residuals
        self.model.residuals[0:3] = np.nan
        
        var_obj = variogram(self.model)
        
        # Should still work
        assert len(var_obj.distances) > 0
        
    def test_variogram_properties(self):
        """Test that variogram has expected properties."""
        var_obj = variogram(self.model)
        
        # Generally, semivariance should increase with distance (not always true)
        # But at least first distance should be relatively small
        assert var_obj.gamma[0] >= 0
        
        # Should have reasonable number of pairs at short distances
        assert var_obj.n_pairs[0] > 0


class TestVariogramModelFitting:
    """Test fitting theoretical models to empirical variograms."""
    
    def setup_method(self):
        """Set up test variogram."""
        # Create synthetic variogram data
        distances = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
        
        # True parameters
        true_nugget, true_sill, true_range = 1.0, 3.0, 2.0
        
        # Generate true values with some noise
        true_gamma = _spherical_model(distances, true_nugget, true_sill, true_range)
        noise = np.random.normal(0, 0.1, len(distances))
        gamma = true_gamma + noise
        gamma = np.maximum(gamma, 0)  # Ensure non-negative
        
        n_pairs = np.array([100, 95, 80, 70, 60, 45, 30, 20])  # Decreasing with distance
        
        self.var_obj = Variogram(distances, gamma, n_pairs)
        
    def test_spherical_model_fitting(self):
        """Test fitting spherical model."""
        result = fit_variogram_model(self.var_obj, model='spherical')
        
        assert result['model'] == 'spherical'
        assert 'nugget' in result
        assert 'sill' in result
        assert 'range' in result
        assert 'r_squared' in result
        
        # Parameters should be positive
        assert result['nugget'] >= 0
        assert result['sill'] >= 0
        assert result['range'] > 0
        
    def test_exponential_model_fitting(self):
        """Test fitting exponential model."""
        result = fit_variogram_model(self.var_obj, model='exponential')
        
        assert result['model'] == 'exponential'
        assert result['nugget'] >= 0
        assert result['sill'] >= 0
        assert result['range'] > 0
        
    def test_gaussian_model_fitting(self):
        """Test fitting Gaussian model."""
        result = fit_variogram_model(self.var_obj, model='gaussian')
        
        assert result['model'] == 'gaussian'
        assert result['nugget'] >= 0
        assert result['sill'] >= 0
        assert result['range'] > 0
        
    def test_invalid_model(self):
        """Test error for invalid model type."""
        with pytest.raises(ValueError, match="Unknown variogram model"):
            fit_variogram_model(self.var_obj, model='invalid')
            
    def test_fitted_values_stored(self):
        """Test that fitted values are stored in variogram object."""
        fit_variogram_model(self.var_obj, model='spherical')
        
        assert hasattr(self.var_obj, 'fitted_model')
        assert hasattr(self.var_obj, 'fitted_params')
        assert hasattr(self.var_obj, 'fitted_gamma')
        
        assert self.var_obj.fitted_model == 'spherical'
        assert len(self.var_obj.fitted_gamma) == len(self.var_obj.distances)


class TestDirectionalVariogram:
    """Test directional variogram analysis."""
    
    def setup_method(self):
        """Set up test data with directional structure."""
        np.random.seed(123)
        
        # Create data with preferential variation in one direction
        n_rows, n_cols = 12, 10
        rows = np.repeat(np.arange(1, n_rows + 1), n_cols)
        cols = np.tile(np.arange(1, n_cols + 1), n_rows)
        
        # Add stronger variation in row direction
        row_effect = 2 * np.sin(2 * np.pi * (rows - 1) / n_rows)
        col_effect = 0.5 * np.sin(2 * np.pi * (cols - 1) / n_cols)
        
        response = 100 + row_effect + col_effect + np.random.normal(0, 1, len(rows))
        
        self.data = pd.DataFrame({
            'response': response,
            'genotype': pd.Categorical(np.random.choice(['A', 'B', 'C'], len(rows))),
            'row': rows,
            'col': cols
        })
        
        self.model = SpATS(
            response='response',
            genotype='genotype',
            spatial=('col', 'row'),
            data=self.data
        )
        
    def test_directional_variogram_computation(self):
        """Test directional variogram computation."""
        # Test different directions
        for direction in [0, 45, 90, 135]:
            var_dir = directional_variogram(self.model, direction=direction)
            
            assert isinstance(var_dir, Variogram)
            # May be empty for directions with no data
            assert len(var_dir.gamma) == len(var_dir.distances)
            assert len(var_dir.n_pairs) == len(var_dir.distances)
            
    def test_directional_tolerance(self):
        """Test directional variogram with different tolerances."""
        direction = 90  # North
        
        for tolerance in [15, 30, 45]:
            var_dir = directional_variogram(
                self.model, direction=direction, tolerance=tolerance
            )
            
            # Should return valid variogram object (may be empty)
            assert isinstance(var_dir, Variogram)
            assert len(var_dir.gamma) == len(var_dir.distances)
            
    def test_directional_angle_wrapping(self):
        """Test angle wrapping for directional variogram."""
        # These should be equivalent due to angle wrapping
        var_0 = directional_variogram(self.model, direction=0)
        var_360 = directional_variogram(self.model, direction=360)
        
        # Should have similar results (may not be identical due to numerical precision)
        assert len(var_0.distances) == len(var_360.distances)


class TestVariogramEdgeCases:
    """Test edge cases in variogram analysis."""
    
    def test_insufficient_data(self):
        """Test variogram with very little data."""
        # Create minimal dataset
        data = pd.DataFrame({
            'response': [1, 2, 3],
            'genotype': pd.Categorical(['A', 'A', 'B']),
            'row': [1, 2, 3],
            'col': [1, 1, 1]
        })
        
        model = SpATS(
            response='response',
            genotype='genotype',
            spatial=('col', 'row'),
            data=data
        )
        
        # Should raise error or handle gracefully
        with pytest.raises(ValueError):
            variogram(model)
            
    def test_no_spatial_variation(self):
        """Test variogram with no spatial variation."""
        # All points at same location
        data = pd.DataFrame({
            'response': np.random.normal(100, 10, 20),
            'genotype': pd.Categorical(np.random.choice(['A', 'B'], 20)),
            'row': np.ones(20),
            'col': np.ones(20)
        })
        
        model = SpATS(
            response='response',
            genotype='genotype',
            spatial=('col', 'row'),
            data=data
        )
        
        # Should handle degenerate case
        try:
            var_obj = variogram(model)
            # If it succeeds, distances should be zero or very small
            assert np.all(var_obj.distances < 1e-10)
        except ValueError:
            # Also acceptable to raise error
            pass
            
    def test_perfect_model_fit(self):
        """Test variogram fitting with perfect synthetic data."""
        # Create perfect spherical variogram
        distances = np.linspace(0.1, 5, 20)
        nugget, sill, range_param = 1.0, 4.0, 2.0
        gamma = _spherical_model(distances, nugget, sill, range_param)
        n_pairs = np.ones(len(distances), dtype=int) * 100
        
        var_obj = Variogram(distances, gamma, n_pairs)
        result = fit_variogram_model(var_obj, model='spherical')
        
        # Should recover parameters very accurately
        assert abs(result['nugget'] - nugget) < 0.01
        assert abs(result['sill'] - sill) < 0.01
        assert abs(result['range'] - range_param) < 0.01
        assert result['r_squared'] > 0.99
        
    def test_bad_fitting_data(self):
        """Test variogram fitting with problematic data."""
        # Create problematic variogram (decreasing)
        distances = np.array([1, 2, 3, 4, 5])
        gamma = np.array([5, 4, 3, 2, 1])  # Decreasing (non-physical)
        n_pairs = np.array([10, 8, 6, 4, 2])
        
        var_obj = Variogram(distances, gamma, n_pairs)
        result = fit_variogram_model(var_obj, model='spherical')
        
        # Should not crash, but may have poor fit
        assert 'nugget' in result
        assert 'sill' in result
        assert 'range' in result


if __name__ == '__main__':
    pytest.main([__file__])