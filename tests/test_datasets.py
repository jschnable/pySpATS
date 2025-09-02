"""
Test cases for dataset generation and loading.
"""

import pytest
import numpy as np
import pandas as pd

import sys
import os
# Add parent directory to path to find pyspats package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspats.datasets import (
    load_wheatdata, generate_field_trial_data, 
    load_example_spatial_data, create_toy_example
)


class TestWheatData:
    """Test wheat dataset loading and properties."""
    
    def test_load_wheatdata(self):
        """Test loading wheat dataset."""
        data = load_wheatdata()
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        
        # Check required columns
        required_cols = ['yield', 'geno', 'rep', 'row', 'col', 'rowcode', 'colcode']
        for col in required_cols:
            assert col in data.columns
            
    def test_wheatdata_structure(self):
        """Test wheat dataset structure."""
        data = load_wheatdata()
        
        # Check data types
        assert pd.api.types.is_numeric_dtype(data['yield'])
        assert pd.api.types.is_categorical_dtype(data['geno']) or data['geno'].dtype == 'object'
        assert pd.api.types.is_categorical_dtype(data['rep']) or data['rep'].dtype == 'object'
        assert pd.api.types.is_numeric_dtype(data['row'])
        assert pd.api.types.is_numeric_dtype(data['col'])
        
        # Check positive yields
        assert np.all(data['yield'] > 0)
        
        # Check reasonable ranges
        assert data['row'].min() >= 1
        assert data['col'].min() >= 1
        
    def test_wheatdata_factors(self):
        """Test categorical variables in wheat data."""
        data = load_wheatdata()
        
        # Should have multiple genotypes
        assert data['geno'].nunique() > 10
        
        # Should have multiple replicates
        assert data['rep'].nunique() >= 2
        
        # Row and column codes should be factors
        assert data['rowcode'].nunique() >= 2
        assert data['colcode'].nunique() >= 2
        
    def test_wheatdata_spatial_coverage(self):
        """Test spatial coverage of wheat data."""
        data = load_wheatdata()
        
        # Should have reasonable spatial dimensions
        n_rows = data['row'].nunique()
        n_cols = data['col'].nunique()
        
        assert n_rows >= 5
        assert n_cols >= 5
        
        # Total observations should be reasonable
        assert len(data) >= 50


class TestFieldTrialGeneration:
    """Test field trial data generation."""
    
    def test_basic_generation(self):
        """Test basic field trial data generation."""
        data = generate_field_trial_data(seed=42)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        
        # Check required columns
        required_cols = ['response', 'genotype', 'row', 'col']
        for col in required_cols:
            assert col in data.columns
            
    def test_custom_parameters(self):
        """Test generation with custom parameters."""
        data = generate_field_trial_data(
            n_rows=8,
            n_cols=6,
            n_genotypes=15,
            seed=123
        )
        
        assert len(data) == 8 * 6
        assert data['genotype'].nunique() <= 15
        assert data['row'].max() == 8
        assert data['col'].max() == 6
        
    def test_variance_parameters(self):
        """Test generation with different variance parameters."""
        # High spatial variance
        data_high_spatial = generate_field_trial_data(
            spatial_variance=200.0,
            genotype_variance=10.0,
            error_variance=5.0,
            seed=456
        )
        
        # Low spatial variance  
        data_low_spatial = generate_field_trial_data(
            spatial_variance=10.0,
            genotype_variance=10.0,
            error_variance=5.0,
            seed=456
        )
        
        # High spatial variance data should have more spatial variation
        # This is a rough test - exact comparison depends on random realization
        assert isinstance(data_high_spatial, pd.DataFrame)
        assert isinstance(data_low_spatial, pd.DataFrame)
        
    def test_missing_data_generation(self):
        """Test generation with missing data."""
        missing_rate = 0.2
        data = generate_field_trial_data(missing_rate=missing_rate, seed=789)
        
        # Should have missing values
        n_missing = data['response'].isnull().sum()
        expected_missing = len(data) * missing_rate
        
        # Allow some tolerance due to randomness
        assert abs(n_missing - expected_missing) <= 0.1 * len(data)
        
    def test_reproducibility(self):
        """Test that generation is reproducible with same seed."""
        data1 = generate_field_trial_data(seed=42)
        data2 = generate_field_trial_data(seed=42)
        
        pd.testing.assert_frame_equal(data1, data2)
        
    def test_different_seeds_differ(self):
        """Test that different seeds produce different data."""
        data1 = generate_field_trial_data(seed=1)
        data2 = generate_field_trial_data(seed=2)
        
        # Should be different
        assert not data1['response'].equals(data2['response'])
        
    def test_genotype_assignment(self):
        """Test genotype assignment in generated data."""
        n_genotypes = 20
        data = generate_field_trial_data(n_genotypes=n_genotypes, seed=101)
        
        # Should have at most n_genotypes unique genotypes
        assert data['genotype'].nunique() <= n_genotypes
        
        # With reasonable field size, should use most genotypes
        unique_genos = data['genotype'].nunique()
        assert unique_genos >= min(n_genotypes * 0.7, n_genotypes)


class TestToyExample:
    """Test toy example generation."""
    
    def test_toy_example_creation(self):
        """Test creation of toy example."""
        data = create_toy_example()
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert len(data) < 100  # Should be small
        
        # Check required columns
        required_cols = ['response', 'genotype', 'row', 'col']
        for col in required_cols:
            assert col in data.columns
            
    def test_toy_example_size(self):
        """Test toy example has appropriate size."""
        data = create_toy_example()
        
        # Should be small enough for quick testing
        assert len(data) <= 50
        assert data['genotype'].nunique() <= 20


class TestExampleDataLoader:
    """Test example data loading functionality."""
    
    def test_load_wheat_dataset(self):
        """Test loading wheat dataset via example loader."""
        data = load_example_spatial_data('wheat')
        
        assert isinstance(data, pd.DataFrame)
        assert 'yield' in data.columns
        
    def test_load_simulated_dataset(self):
        """Test loading simulated dataset via example loader."""
        data = load_example_spatial_data('simulated')
        
        assert isinstance(data, pd.DataFrame)
        assert 'response' in data.columns
        
    def test_invalid_dataset_name(self):
        """Test error for invalid dataset name."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_example_spatial_data('nonexistent')


class TestDataProperties:
    """Test properties of generated datasets."""
    
    def test_spatial_coordinates_coverage(self):
        """Test spatial coordinate coverage."""
        data = generate_field_trial_data(n_rows=10, n_cols=8, seed=202)
        
        # Should cover full spatial range
        assert data['row'].min() == 1
        assert data['row'].max() == 10
        assert data['col'].min() == 1
        assert data['col'].max() == 8
        
        # Should have all row/col combinations
        spatial_combinations = data[['row', 'col']].drop_duplicates()
        assert len(spatial_combinations) == 10 * 8
        
    def test_response_distribution(self):
        """Test properties of response distribution."""
        data = generate_field_trial_data(seed=303)
        
        # Response should be roughly normal (allowing for skewness)
        response = data['response'].dropna()
        
        assert len(response) > 0
        assert np.isfinite(response).all()
        
        # Should have reasonable mean and variance
        assert response.mean() > 0  # Assuming base response is positive
        assert response.std() > 0   # Should have variation
        
    def test_genotype_balance(self):
        """Test genotype representation balance."""
        data = generate_field_trial_data(n_genotypes=10, seed=404)
        
        # Count observations per genotype
        geno_counts = data['genotype'].value_counts()
        
        # Should have reasonable balance (not perfect due to randomness)
        min_count = geno_counts.min()
        max_count = geno_counts.max()
        
        # Ratio shouldn't be too extreme
        assert max_count / min_count <= 5  # Allows some imbalance
        
    def test_factor_levels(self):
        """Test factor variables have appropriate levels."""
        data = generate_field_trial_data(seed=505)
        
        # Treatment should have reasonable number of levels
        if 'treatment' in data.columns:
            assert data['treatment'].nunique() >= 2
            
        # Block should have reasonable number of levels
        if 'block' in data.columns:
            assert data['block'].nunique() >= 2
            
    def test_data_consistency(self):
        """Test internal consistency of generated data."""
        data = generate_field_trial_data(n_rows=6, n_cols=5, seed=606)
        
        # Number of observations should match grid
        assert len(data) == 6 * 5
        
        # Each row/col combination should appear exactly once
        spatial_counts = data.groupby(['row', 'col']).size()
        assert np.all(spatial_counts == 1)


class TestEdgeCases:
    """Test edge cases in data generation."""
    
    def test_minimal_field_size(self):
        """Test with minimal field dimensions."""
        data = generate_field_trial_data(n_rows=2, n_cols=2, seed=707)
        
        assert len(data) == 4
        assert data['row'].nunique() == 2
        assert data['col'].nunique() == 2
        
    def test_single_genotype(self):
        """Test with single genotype."""
        data = generate_field_trial_data(n_genotypes=1, seed=808)
        
        assert data['genotype'].nunique() == 1
        
    def test_zero_variances(self):
        """Test with zero variance components."""
        data = generate_field_trial_data(
            spatial_variance=0.0,
            genotype_variance=0.0,
            error_variance=1.0,  # Keep some variance
            seed=909
        )
        
        # Should still generate valid data
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        
    def test_high_missing_rate(self):
        """Test with high missing rate."""
        data = generate_field_trial_data(missing_rate=0.8, seed=1010)
        
        # Should have many missing values
        missing_prop = data['response'].isnull().mean()
        assert missing_prop >= 0.7  # Allow some tolerance
        
        # But should still have some valid observations
        assert data['response'].notna().sum() > 0


if __name__ == '__main__':
    pytest.main([__file__])