"""
Test cases for heritability calculation.
"""

import pytest
import sys
import os

# Add parent directory to path to find pyspats package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspats.utils import get_heritability


def test_get_heritability_generalized_vs_classical():
    """Test that generalized and classical heritability formulas work correctly."""
    EDg = 50.0
    nG = 100

    h2_gen = get_heritability(EDg, nG, mode="generalized")
    h2_cls = get_heritability(EDg, nG, mode="classical")

    # Check formulas
    assert abs(h2_gen - (EDg / nG)) < 1e-12
    assert abs(h2_cls - (EDg / (nG - 1))) < 1e-12

    # Sanity: generalized is slightly smaller than classical at same ED and n
    assert h2_gen < h2_cls


def test_get_heritability_edge_cases():
    """Test edge cases for heritability calculation."""
    # n_geno = 2 (minimum)
    h2 = get_heritability(1.0, 2, mode="generalized")
    assert abs(h2 - 0.5) < 1e-12

    h2_cls = get_heritability(1.0, 2, mode="classical")
    assert abs(h2_cls - 1.0) < 1e-12

    # n_geno = 1 should raise error
    with pytest.raises(ValueError, match="n_geno must be > 1"):
        get_heritability(1.0, 1, mode="generalized")


def test_get_heritability_invalid_mode():
    """Test that invalid mode raises error."""
    with pytest.raises(ValueError, match="mode must be 'generalized' or 'classical'"):
        get_heritability(50.0, 100, mode="invalid")


def test_get_heritability_values():
    """Test specific heritability values."""
    # Test case 1: ED_geno = 80, n_geno = 100
    h2_gen = get_heritability(80.0, 100, mode="generalized")
    assert abs(h2_gen - 0.8) < 1e-12

    h2_cls = get_heritability(80.0, 100, mode="classical")
    expected_cls = 80.0 / 99.0
    assert abs(h2_cls - expected_cls) < 1e-12

    # Test case 2: ED_geno = 30, n_geno = 50
    h2_gen = get_heritability(30.0, 50, mode="generalized")
    assert abs(h2_gen - 0.6) < 1e-12

    h2_cls = get_heritability(30.0, 50, mode="classical")
    expected_cls = 30.0 / 49.0
    assert abs(h2_cls - expected_cls) < 1e-12


def test_get_heritability_default_mode():
    """Test that default mode is 'generalized'."""
    EDg = 50.0
    nG = 100

    h2_default = get_heritability(EDg, nG)
    h2_explicit = get_heritability(EDg, nG, mode="generalized")

    assert h2_default == h2_explicit
    assert abs(h2_default - 0.5) < 1e-12
