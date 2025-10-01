"""
Unit tests for exact ED computation via CHOLMOD selected inverse.
"""

import pytest
import numpy as np
import scipy.sparse as sp
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspats.ed_selected_inverse import (
    BlockInfo,
    ed_components_from_selected_inverse,
    is_cholmod_available,
    factorize_C_cholmod,
    _diag_selected_inverse
)

# Skip all tests if CHOLMOD not available
pytestmark = pytest.mark.skipif(
    not is_cholmod_available(),
    reason="CHOLMOD not available (requires scikit-sparse with SuiteSparse)"
)


class TestBlockInfo:
    """Test BlockInfo helper class."""

    def test_basic_properties(self):
        """Test basic BlockInfo properties."""
        block = BlockInfo("test", 5, 10, is_random=True)
        assert block.name == "test"
        assert block.start == 5
        assert block.stop == 10
        assert block.size == 5
        assert block.is_random is True
        assert block.slice == slice(5, 10)

    def test_repr(self):
        """Test string representation."""
        block = BlockInfo("geno", 0, 100, is_random=False)
        repr_str = repr(block)
        assert "geno" in repr_str
        assert "0:100" in repr_str


class TestCHOLMODFactorization:
    """Test CHOLMOD factorization utilities."""

    def test_simple_factorization(self):
        """Test factorization of simple SPD matrix."""
        # Create simple SPD matrix
        n = 5
        C = sp.eye(n, format='csc') * 2.0
        C = C + sp.eye(n, k=1, format='csc') * 0.5
        C = C + sp.eye(n, k=-1, format='csc') * 0.5

        factor, P, invP = factorize_C_cholmod(C)

        # Check permutation properties
        assert len(P) == n
        assert len(invP) == n
        assert np.all(invP[P] == np.arange(n))
        assert np.all(P[invP] == np.arange(n))

    def test_diagonal_selected_inverse(self):
        """Test diagonal extraction matches dense inverse."""
        # Small test matrix
        n = 6
        C_dense = np.eye(n) * 3.0
        C_dense += np.eye(n, k=1) * 0.5
        C_dense += np.eye(n, k=-1) * 0.5
        C = sp.csc_matrix(C_dense)

        # Compute exact inverse diagonal from dense
        C_inv_dense = np.linalg.inv(C_dense)
        diag_inv_exact = np.diag(C_inv_dense)

        # Compute via selected inverse
        factor, P, invP = factorize_C_cholmod(C)
        idx = np.arange(n)
        diag_inv_sparse = _diag_selected_inverse(factor, idx, P, invP)

        # Compare
        np.testing.assert_allclose(diag_inv_sparse, diag_inv_exact, rtol=1e-9, atol=1e-12)


class TestEDComputation:
    """Test exact ED computation."""

    def test_simple_two_block_identity_G(self):
        """Test ED computation for 2 random blocks with G_k = I."""
        # Build coefficient matrix
        # Structure: [fixed(3) | random1(3) | random2(4)]
        n = 10
        C_dense = np.eye(n) * 2.0
        C_dense += np.eye(n, k=1) * 0.3
        C_dense += np.eye(n, k=-1) * 0.3
        C = sp.csc_matrix(C_dense)

        # Define blocks
        blocks = [
            BlockInfo('fixed', 0, 3, is_random=False),
            BlockInfo('block1', 3, 6, is_random=True),
            BlockInfo('block2', 6, 10, is_random=True)
        ]

        # Compute ED via selected inverse
        ed_sparse = ed_components_from_selected_inverse(C, blocks, G_blocks=None)

        # Compute ED via dense inverse
        C_inv_dense = np.linalg.inv(C_dense)
        ed_dense = {}
        for b in blocks:
            if not b.is_random:
                continue
            idx = np.arange(b.start, b.stop)
            diag_inv_block = np.diag(C_inv_dense)[idx]
            m_k = b.size
            # G_k = I, so tr(G^{-1} C^{-1}) = tr(C^{-1}) = sum(diag)
            tr_term = np.sum(diag_inv_block)
            ed_dense[b.name] = m_k - tr_term

        # Compare
        for name in ['block1', 'block2']:
            np.testing.assert_allclose(
                ed_sparse[name],
                ed_dense[name],
                rtol=1e-9,
                atol=1e-12,
                err_msg=f"ED mismatch for {name}"
            )

    def test_scalar_G_blocks(self):
        """Test ED with scalar G_k (G_k = sigma^2 I)."""
        # Build coefficient matrix
        n = 8
        C_dense = np.eye(n) * 3.0
        C_dense += np.eye(n, k=1) * 0.4
        C_dense += np.eye(n, k=-1) * 0.4
        C = sp.csc_matrix(C_dense)

        # Define blocks
        blocks = [
            BlockInfo('fixed', 0, 2, is_random=False),
            BlockInfo('geno', 2, 5, is_random=True),
            BlockInfo('spatial', 5, 8, is_random=True)
        ]

        # G_k = sigma_k^2 I with different sigmas
        G_blocks = {
            'geno': 2.0,      # sigma^2 = 2
            'spatial': 0.5    # sigma^2 = 0.5
        }

        # Compute ED via selected inverse
        ed_sparse = ed_components_from_selected_inverse(C, blocks, G_blocks)

        # Compute ED via dense inverse
        C_inv_dense = np.linalg.inv(C_dense)
        ed_dense = {}
        for b in blocks:
            if not b.is_random:
                continue
            idx = np.arange(b.start, b.stop)
            diag_inv_block = np.diag(C_inv_dense)[idx]
            m_k = b.size
            sigma_sq = G_blocks[b.name]
            tr_term = np.sum(diag_inv_block) / sigma_sq
            ed_dense[b.name] = m_k - tr_term

        # Compare
        for name in ['geno', 'spatial']:
            np.testing.assert_allclose(
                ed_sparse[name],
                ed_dense[name],
                rtol=1e-9,
                atol=1e-12,
                err_msg=f"ED mismatch for {name}"
            )

    def test_diagonal_G_blocks(self):
        """Test ED with diagonal G_k (non-constant variances)."""
        # Build coefficient matrix
        n = 7
        C_dense = np.eye(n) * 2.5
        C_dense += np.eye(n, k=1) * 0.2
        C_dense += np.eye(n, k=-1) * 0.2
        C = sp.csc_matrix(C_dense)

        # Define blocks
        blocks = [
            BlockInfo('fixed', 0, 2, is_random=False),
            BlockInfo('random1', 2, 5, is_random=True),
            BlockInfo('random2', 5, 7, is_random=True)
        ]

        # Diagonal G_k (different variance for each coefficient)
        G_blocks = {
            'random1': np.array([1.0, 2.0, 1.5]),  # length 3
            'random2': np.array([0.8, 1.2])        # length 2
        }

        # Compute ED via selected inverse
        ed_sparse = ed_components_from_selected_inverse(C, blocks, G_blocks)

        # Compute ED via dense inverse
        C_inv_dense = np.linalg.inv(C_dense)
        ed_dense = {}
        for b in blocks:
            if not b.is_random:
                continue
            idx = np.arange(b.start, b.stop)
            diag_inv_block = np.diag(C_inv_dense)[idx]
            m_k = b.size
            G_diag = G_blocks[b.name]
            # tr(G^{-1} C^{-1}) = sum(C^{-1}_{ii} / G_{ii})
            tr_term = np.sum(diag_inv_block / G_diag)
            ed_dense[b.name] = m_k - tr_term

        # Compare
        for name in ['random1', 'random2']:
            np.testing.assert_allclose(
                ed_sparse[name],
                ed_dense[name],
                rtol=1e-9,
                atol=1e-12,
                err_msg=f"ED mismatch for {name}"
            )

    def test_only_random_blocks_returned(self):
        """Test that only random blocks appear in ED output."""
        n = 6
        C = sp.eye(n, format='csc') * 2.0

        blocks = [
            BlockInfo('fixed1', 0, 2, is_random=False),
            BlockInfo('random1', 2, 4, is_random=True),
            BlockInfo('fixed2', 4, 5, is_random=False),
            BlockInfo('random2', 5, 6, is_random=True)
        ]

        ed = ed_components_from_selected_inverse(C, blocks)

        # Should only have random blocks
        assert set(ed.keys()) == {'random1', 'random2'}
        assert 'fixed1' not in ed
        assert 'fixed2' not in ed

    def test_error_on_wrong_G_shape(self):
        """Test error when G_k has wrong shape."""
        n = 6
        C = sp.eye(n, format='csc') * 2.0

        blocks = [
            BlockInfo('random1', 0, 3, is_random=True),
            BlockInfo('random2', 3, 6, is_random=True)
        ]

        # Wrong shape for block1 (should be length 3, not 2)
        G_blocks = {
            'random1': np.array([1.0, 2.0])  # Wrong! Should be length 3
        }

        with pytest.raises(ValueError, match="must be scalar or 1D array of length 3"):
            ed_components_from_selected_inverse(C, blocks, G_blocks)

    def test_large_realistic_matrix(self):
        """Test on larger, more realistic coefficient matrix."""
        # Simulate realistic mixed model coefficient matrix
        np.random.seed(42)
        n = 50

        # Build sparse SPD matrix
        # Add diagonal dominance
        C = sp.eye(n, format='csc') * 5.0

        # Add some off-diagonal structure
        for k in [1, 2]:
            C = C + sp.eye(n, k=k, format='csc') * 0.5
            C = C + sp.eye(n, k=-k, format='csc') * 0.5

        # Ensure SPD by adding more to diagonal
        C = C + sp.eye(n, format='csc') * 2.0

        # Define blocks: fixed(10) + 2 random blocks
        blocks = [
            BlockInfo('fixed', 0, 10, is_random=False),
            BlockInfo('genotype', 10, 30, is_random=True),
            BlockInfo('spatial', 30, 50, is_random=True)
        ]

        G_blocks = {
            'genotype': 1.5,
            'spatial': 0.8
        }

        # Compute via sparse
        ed_sparse = ed_components_from_selected_inverse(C, blocks, G_blocks)

        # Verify via dense (for blocks, use subsets to keep it tractable)
        C_dense = C.toarray()
        C_inv_dense = np.linalg.inv(C_dense)

        for b in blocks:
            if not b.is_random:
                continue
            idx = np.arange(b.start, b.stop)
            diag_inv_block = np.diag(C_inv_dense)[idx]
            m_k = b.size
            sigma_sq = G_blocks[b.name]
            tr_term = np.sum(diag_inv_block) / sigma_sq
            ed_dense_k = m_k - tr_term

            np.testing.assert_allclose(
                ed_sparse[b.name],
                ed_dense_k,
                rtol=1e-8,
                atol=1e-10,
                err_msg=f"ED mismatch for {b.name} in large matrix"
            )


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_element_block(self):
        """Test block with single element."""
        n = 5
        C = sp.eye(n, format='csc') * 2.0

        blocks = [
            BlockInfo('fixed', 0, 2, is_random=False),
            BlockInfo('single', 2, 3, is_random=True),
            BlockInfo('normal', 3, 5, is_random=True)
        ]

        ed = ed_components_from_selected_inverse(C, blocks)

        # Should work fine
        assert 'single' in ed
        assert 'normal' in ed
        # For identity-like matrix with C=2I, inv(C)=0.5I
        # ED = 1 - tr(0.5) = 1 - 0.5 = 0.5
        assert ed['single'] == pytest.approx(0.5, abs=1e-10)

    def test_no_random_blocks(self):
        """Test when there are no random blocks."""
        n = 5
        C = sp.eye(n, format='csc') * 2.0

        blocks = [
            BlockInfo('fixed1', 0, 3, is_random=False),
            BlockInfo('fixed2', 3, 5, is_random=False)
        ]

        ed = ed_components_from_selected_inverse(C, blocks)

        # Should return empty dict
        assert ed == {}


def test_cholmod_availability():
    """Test CHOLMOD availability check."""
    # This should not raise, just return bool
    available = is_cholmod_available()
    assert isinstance(available, bool)
    # In test environment with skipif, this should be True
    assert available is True
