"""
Test cases for core SpATS functionality.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

import sys
import os
# Add parent directory to path to find pyspats package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspats.core import SpATS
from pyspats.control import SpATSControl
from pyspats.families import gaussian, poisson, binomial
from pyspats.datasets import create_toy_example, generate_field_trial_data


class TestSpATS:
    """Test cases for SpATS class."""
    
    def setup_method(self):
        """Set up test data."""
        self.data = create_toy_example()
        
    def test_basic_initialization(self):
        """Test basic SpATS initialization."""
        model = SpATS(
            response='response',
            genotype='genotype',
            spatial=('col', 'row'),
            data=self.data
        )
        
        assert model.response == 'response'
        assert model.genotype == 'genotype'
        assert model.spatial == ('col', 'row')
        assert not model.genotype_as_random
        
    def test_input_validation(self):
        """Test input validation."""
        # Missing column
        with pytest.raises(ValueError, match="Missing columns"):
            SpATS(
                response='missing_col',
                genotype='genotype',
                spatial=('col', 'row'),
                data=self.data
            )
        
        # Non-DataFrame data
        with pytest.raises(ValueError, match="data must be a pandas DataFrame"):
            SpATS(
                response='response',
                genotype='genotype',
                spatial=('col', 'row'),
                data="not a dataframe"
            )
    
    def test_basic_fitting(self):
        """Test basic model fitting."""
        model = SpATS(
            response='response',
            genotype='genotype',
            spatial=('col', 'row'),
            data=self.data
        )
        
        # Check that model fitted
        assert hasattr(model, 'fitted_values')
        assert hasattr(model, 'residuals')
        assert hasattr(model, 'coefficients')
        assert hasattr(model, 'deviance')
        
        # Check dimensions
        assert len(model.fitted_values) == len(self.data)
        assert len(model.residuals) == len(self.data)
        
    def test_fixed_and_random_effects(self):
        """Test model with fixed and random effects."""
        model = SpATS(
            response='response',
            genotype='genotype',
            spatial=('col', 'row'),
            fixed=['treatment'],
            random=['block'],
            data=self.data
        )
        
        assert 'treatment' in model.fixed
        assert 'block' in model.random
        
    def test_genotype_as_random(self):
        """Test genotype as random effect."""
        model = SpATS(
            response='response',
            genotype='genotype',
            spatial=('col', 'row'),
            genotype_as_random=True,
            data=self.data
        )
        
        assert model.genotype_as_random
        
    def test_custom_family(self):
        """Test custom distribution families."""
        # Gaussian (default)
        model_gauss = SpATS(
            response='response',
            genotype='genotype',
            spatial=('col', 'row'),
            family=gaussian(),
            data=self.data
        )
        assert model_gauss.family.family == 'gaussian'
        
        # For other families, would need appropriate data
        # This tests the interface
        model_pois = SpATS(
            response='response',
            genotype='genotype',
            spatial=('col', 'row'),
            family=poisson(),
            data=self.data
        )
        assert model_pois.family.family == 'poisson'
        
    def test_custom_control(self):
        """Test custom control parameters."""
        control = SpATSControl(tolerance=1e-6, monitoring=True)
        
        model = SpATS(
            response='response',
            genotype='genotype',
            spatial=('col', 'row'),
            control=control,
            data=self.data
        )
        
        assert model.control.tolerance == 1e-6
        assert model.control.monitoring
        
    def test_weights_and_offset(self):
        """Test weights and offset."""
        n_obs = len(self.data)
        weights = np.random.uniform(0.5, 1.5, n_obs)
        offset = np.random.normal(0, 0.1, n_obs)
        
        model = SpATS(
            response='response',
            genotype='genotype',
            spatial=('col', 'row'),
            weights=weights,
            offset=offset,
            data=self.data
        )
        
        assert len(model.weights) == n_obs
        assert len(model.offset) == n_obs
        
    def test_missing_data_handling(self):
        """Test handling of missing data."""
        data_missing = self.data.copy()
        
        # Introduce missing values in response
        data_missing.loc[:2, 'response'] = np.nan
        
        # Introduce missing values in predictors
        data_missing.loc[3:5, 'col'] = np.nan
        
        model = SpATS(
            response='response',
            genotype='genotype',
            spatial=('col', 'row'),
            data=data_missing
        )
        
        # Should have fewer valid observations
        assert model.n_obs < len(data_missing)
        
    def test_predict_method(self):
        """Test prediction method."""
        model = SpATS(
            response='response',
            genotype='genotype',
            spatial=('col', 'row'),
            data=self.data
        )
        
        # Predict on same data (fitted values)
        predictions = model.predict()
        assert len(predictions) == len(self.data)
        np.testing.assert_array_equal(predictions, model.fitted_values)
        
    def test_summary_methods(self):
        """Test summary methods."""
        model = SpATS(
            response='response',
            genotype='genotype',
            spatial=('col', 'row'),
            data=self.data
        )
        
        # These should not raise errors
        model.summary()
        model.summary('dimensions')
        model.summary('variances')
        model.summary('all')
        
    def test_string_representation(self):
        """Test string representation."""
        model = SpATS(
            response='response',
            genotype='genotype',
            spatial=('col', 'row'),
            data=self.data
        )
        
        repr_str = repr(model)
        assert 'SpATS' in repr_str
        assert 'response' in repr_str
        assert 'genotype' in repr_str


class TestSpATSIntegration:
    """Integration tests for SpATS functionality."""
    
    def test_wheat_data_analysis(self):
        """Test analysis with wheat-like data."""
        from spats.datasets import load_wheatdata
        
        # This might be slow, so use smaller version
        data = generate_field_trial_data(
            n_rows=10, n_cols=8, n_genotypes=20, seed=42
        )
        
        model = SpATS(
            response='response',
            genotype='genotype',
            spatial=('col', 'row'),
            fixed=['treatment'],
            random=['block'],
            data=data
        )
        
        # Check model fitting worked
        assert model.deviance > 0
        assert model.n_iterations > 0
        assert len(model.var_comp) > 0
        
    def test_convergence_properties(self):
        """Test model convergence properties."""
        data = generate_field_trial_data(seed=123)
        
        # Very strict tolerance
        control = SpATSControl(tolerance=1e-8, max_iter=500)
        
        model = SpATS(
            response='response',
            genotype='genotype',
            spatial=('col', 'row'),
            control=control,
            data=data
        )
        
        # Should converge
        assert model.n_iterations < 500
        
    def test_large_dataset_performance(self):
        """Test performance with larger dataset."""
        # Create moderate-sized dataset
        data = generate_field_trial_data(
            n_rows=20, n_cols=15, n_genotypes=50, seed=456
        )
        
        import time
        start_time = time.time()
        
        model = SpATS(
            response='response',
            genotype='genotype',
            spatial=('col', 'row'),
            data=data
        )
        
        fitting_time = time.time() - start_time
        
        # Should fit in reasonable time (adjust as needed)
        assert fitting_time < 30  # seconds
        assert model.deviance > 0


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_single_genotype(self):
        """Test with single genotype."""
        data = generate_field_trial_data(n_genotypes=1, seed=789)
        
        # Should work but genotype effect will be absorbed into intercept
        model = SpATS(
            response='response',
            genotype='genotype',
            spatial=('col', 'row'),
            data=data
        )
        
        assert model.deviance > 0
        
    def test_no_spatial_variation(self):
        """Test with no spatial variation."""
        data = generate_field_trial_data(spatial_variance=0, seed=101112)
        
        model = SpATS(
            response='response',
            genotype='genotype',
            spatial=('col', 'row'),
            data=data
        )
        
        # Should still fit
        assert model.deviance > 0
        
    def test_high_missing_rate(self):
        """Test with high missing data rate."""
        data = generate_field_trial_data(missing_rate=0.3, seed=131415)
        
        model = SpATS(
            response='response',
            genotype='genotype',
            spatial=('col', 'row'),
            data=data
        )
        
        # Should handle missing data
        assert model.n_obs < len(data)
        # With high missing data and regularization, deviance might be very small
        assert model.deviance >= 0
        
    def test_extreme_values(self):
        """Test with extreme values."""
        data = create_toy_example()
        
        # Add some extreme values
        data.loc[0, 'response'] = 1000
        data.loc[1, 'response'] = -100
        
        model = SpATS(
            response='response',
            genotype='genotype',
            spatial=('col', 'row'),
            data=data
        )
        
        # Should handle extreme values
        assert model.deviance > 0


if __name__ == '__main__':
    pytest.main([__file__])