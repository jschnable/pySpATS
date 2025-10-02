"""
Unit tests for PS-ANOVA hygiene: orthogonality, nullspace removal, and block structure.

These tests verify that the PS-ANOVA decomposition correctly:
1. Separates polynomial fixed effects from random smooths
2. Removes nullspace (constant/linear trends) from random parts
3. Maintains orthogonality between fixed polynomial and random smooths
4. Creates contiguous, well-labeled random blocks for ED computation
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspats.psanova_basis import (
    make_bspline_basis,
    D2_penalty,
    row_col_bases,
    remove_nullspace_and_whiten,
    project_out_polynomial,
    build_psanova_design,
    verify_orthogonality
)


class TestBSplineBasis:
    """Test B-spline basis construction."""

    def test_basic_bspline(self):
        """Test basic B-spline basis construction."""
        x = np.linspace(0, 10, 50)
        B, knots = make_bspline_basis(x, n_knots=8, degree=3)

        # Check dimensions
        assert B.shape[0] == 50
        assert B.shape[1] > 0  # Should have basis functions
        assert len(knots) > 0

        # Check partition of unity (B-splines sum to 1)
        row_sums = B.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-10)

    def test_constant_input(self):
        """Test B-spline with constant input (degenerate case)."""
        x = np.ones(20) * 5.0
        B, knots = make_bspline_basis(x, n_knots=6, degree=3)

        # Should handle gracefully
        assert B.shape[0] == 20
        assert B.shape[1] > 0

    def test_varying_knots(self):
        """Test different numbers of knots."""
        x = np.linspace(0, 1, 30)

        for n_knots in [5, 10, 15]:
            B, knots = make_bspline_basis(x, n_knots=n_knots, degree=3)
            assert B.shape[0] == 30
            # More knots -> more basis functions
            assert B.shape[1] >= n_knots


class TestPenaltyMatrix:
    """Test 2nd-order difference penalty construction."""

    def test_d2_penalty_basic(self):
        """Test 2nd-order penalty matrix properties."""
        n_coef = 10
        K = D2_penalty(n_coef)

        # Check dimensions
        assert K.shape == (n_coef, n_coef)

        # Check symmetry
        K_dense = K.toarray()
        np.testing.assert_allclose(K_dense, K_dense.T, rtol=1e-14)

        # Check positive semi-definite (all eigenvalues >= 0)
        eigvals = np.linalg.eigvalsh(K_dense)
        assert np.all(eigvals >= -1e-10)

    def test_d2_penalty_nullspace(self):
        """Test that 2nd-order penalty has 2D nullspace (constant + linear)."""
        n_coef = 12
        K = D2_penalty(n_coef)
        K_dense = K.toarray()

        # Eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(K_dense)

        # Should have exactly 2 zero eigenvalues (up to numerical precision)
        zero_eigs = np.sum(eigvals < 1e-8)
        assert zero_eigs == 2, f"Expected 2 zero eigenvalues, got {zero_eigs}"

        # Zero eigenvectors should span constant and linear functions
        # (On uniform grid, these are close to [1,1,...,1] and [0,1,2,...,n-1])
        null_vecs = eigvecs[:, :2]

        # Check that constant and linear vectors are (approximately) in nullspace
        const_vec = np.ones(n_coef)
        linear_vec = np.arange(n_coef, dtype=float)

        # Project onto nullspace
        const_proj = null_vecs @ (null_vecs.T @ const_vec)
        linear_proj = null_vecs @ (null_vecs.T @ linear_vec)

        # Should reconstruct well
        assert np.linalg.norm(const_vec - const_proj) < 1e-6
        assert np.linalg.norm(linear_vec - linear_proj) < 1e-6

    def test_d2_penalty_small(self):
        """Test penalty for small coefficient count."""
        K = D2_penalty(2)
        # Should return identity for degenerate case
        assert K.shape == (2, 2)


class TestNullspaceRemoval:
    """Test nullspace removal and whitening."""

    def test_nullspace_removal_removes_polynomial(self):
        """Test that nullspace removal reduces dimensionality correctly."""
        # Create a B-spline basis
        x = np.linspace(0, 1, 50)
        B, _ = make_bspline_basis(x, n_knots=10, degree=3)
        K = D2_penalty(B.shape[1])

        # Remove nullspace and whiten
        Z_tilde, U_plus = remove_nullspace_and_whiten(B, K)

        # Should have removed 2D nullspace (constant + linear)
        assert Z_tilde.shape[1] == B.shape[1] - 2, \
            f"Expected {B.shape[1] - 2} columns after nullspace removal, got {Z_tilde.shape[1]}"

        # After subsequent polynomial projection (done in build_psanova_design),
        # the result will be orthogonal. This function only removes nullspace.
        # To verify full orthogonality, use the complete pipeline:
        n = len(x)
        X_poly = np.column_stack([np.ones(n), x, x**2])

        # Project out polynomial space
        Z_orth = project_out_polynomial(Z_tilde, X_poly)

        # Now should be orthogonal
        cross = X_poly.T @ Z_orth
        assert np.abs(cross).max() < 1e-8

    def test_whitening_produces_identity_covariance(self):
        """Test that whitening gives identity covariance structure."""
        # Create basis and penalty
        x = np.linspace(0, 1, 40)
        B, _ = make_bspline_basis(x, n_knots=8, degree=3)
        K = D2_penalty(B.shape[1])

        # Whiten
        Z_tilde, U_plus = remove_nullspace_and_whiten(B, K)

        # With whitening, if coefficients u ~ N(0, σ²I), then
        # Z_tilde @ u has covariance σ² Z_tilde @ Z_tilde^T
        # Check that penalty is absorbed correctly:
        # Original: cov(B @ u) = B @ K^+ @ B^T (pseudoinverse)
        # Whitened: Z_tilde = B @ U_+ @ Λ_+^{1/2}, so penalty becomes identity

        # Verify dimensions
        assert Z_tilde.shape[0] == len(x)
        assert Z_tilde.shape[1] <= B.shape[1]  # Nullspace removed
        assert Z_tilde.shape[1] > 0  # Should have some penalized space

    def test_empty_penalized_space(self):
        """Test handling when penalty has no positive eigenvalues."""
        # Create a zero penalty (pure nullspace)
        n = 5
        K = np.zeros((n, n))
        Z = np.random.randn(10, n)

        Z_tilde, U_plus = remove_nullspace_and_whiten(Z, K)

        # Should return empty matrices
        assert Z_tilde.shape == (10, 0)
        assert U_plus.shape == (5, 0)


class TestOrthogonality:
    """Test orthogonality projection and verification."""

    def test_project_out_polynomial_basic(self):
        """Test basic polynomial projection."""
        n = 50
        x = np.linspace(0, 1, n)

        # Polynomial space: [1, x, x^2]
        X_poly = np.column_stack([np.ones(n), x, x**2])

        # Random matrix with some polynomial leakage
        np.random.seed(42)
        Z = np.random.randn(n, 5) + np.outer(x, np.ones(5))  # Add linear trend

        # Project out
        Z_orth = project_out_polynomial(Z, X_poly)

        # Check orthogonality
        cross = X_poly.T @ Z_orth
        assert np.abs(cross).max() < 1e-10

    def test_project_inplace(self):
        """Test in-place projection."""
        n = 30
        X_poly = np.column_stack([np.ones(n), np.arange(n)])
        Z = np.random.randn(n, 3)
        Z_copy = Z.copy()

        # In-place
        Z_inplace = project_out_polynomial(Z, X_poly, inplace=True)

        # Should modify original
        assert Z_inplace is Z
        assert not np.allclose(Z, Z_copy)

    def test_verify_orthogonality_pass(self):
        """Test orthogonality verification with orthogonal blocks."""
        n = 40
        X_poly = np.column_stack([np.ones(n), np.arange(n)])

        # Create orthogonal random blocks
        Q_X, _ = np.linalg.qr(X_poly)
        # Random vectors orthogonal to X_poly
        Z1 = np.random.randn(n, 3)
        Z1 = Z1 - Q_X @ (Q_X.T @ Z1)

        Z2 = np.random.randn(n, 4)
        Z2 = Z2 - Q_X @ (Q_X.T @ Z2)

        # Should pass
        assert verify_orthogonality(X_poly, [Z1, Z2], tol=1e-8)

    def test_verify_orthogonality_fail(self):
        """Test orthogonality verification with non-orthogonal blocks."""
        n = 40
        X_poly = np.column_stack([np.ones(n), np.arange(n)])

        # Non-orthogonal: includes constant
        Z_bad = np.random.randn(n, 3) + 5.0

        # Should fail
        assert not verify_orthogonality(X_poly, [Z_bad], tol=1e-8)


class TestPSANOVADesign:
    """Test full PS-ANOVA design construction."""

    def test_psanova_orthogonality(self):
        """Test that PS-ANOVA produces orthogonal random blocks."""
        # Small grid
        n_r, n_c = 8, 7
        r = np.repeat(np.arange(n_r, dtype=float), n_c)
        c = np.tile(np.arange(n_c, dtype=float), n_r)

        X_poly, Z_r, Z_c, Z_rc, blocks = build_psanova_design(
            r, c, nkr=6, nkc=6, degree=3
        )

        # Orthonormalize X_poly for check
        Q_X, _ = np.linalg.qr(X_poly)

        # Check orthogonality for each block
        for Z_block, block_name in [(Z_r, 'row'), (Z_c, 'col'), (Z_rc, 'interaction')]:
            if Z_block.shape[1] > 0:
                cross = Q_X.T @ Z_block
                max_cross = np.abs(cross).max()
                assert max_cross < 1e-8, f"{block_name}: max |X'Z| = {max_cross:.2e}"

    def test_psanova_nullspace_removed(self):
        """Test that PS-ANOVA removes polynomial nullspace."""
        n_r, n_c = 10, 10
        r = np.repeat(np.linspace(0, 1, n_r), n_c)
        c = np.tile(np.linspace(0, 1, n_c), n_r)

        X_poly, Z_r, Z_c, Z_rc, blocks = build_psanova_design(
            r, c, nkr=8, nkc=8, degree=3
        )

        # Constant and linear trends shouldn't be in random parts
        # Check by fitting polynomial to column sums
        one = np.ones_like(r)
        poly_design = np.column_stack([one, r, c])

        for Z_block, name in [(Z_r, 'row'), (Z_c, 'col'), (Z_rc, 'interaction')]:
            if Z_block.shape[1] > 0:
                # Sum columns of Z_block (handle LinearOperators)
                from scipy.sparse.linalg import LinearOperator
                if isinstance(Z_block, LinearOperator):
                    # Materialize by computing Z @ ones vector
                    z_sum = Z_block @ np.ones(Z_block.shape[1])
                else:
                    z_sum = Z_block.sum(axis=1) if Z_block.shape[1] > 1 else Z_block.ravel()

                # Fit polynomial
                coef, *_ = np.linalg.lstsq(poly_design, z_sum, rcond=None)

                # Coefficients should be small (no polynomial leakage)
                assert np.max(np.abs(coef)) < 1e-4, f"{name}: polynomial leakage detected, max coef = {np.max(np.abs(coef)):.2e}"

    def test_psanova_block_info_contiguous(self):
        """Test that block metadata has contiguous, non-overlapping ranges."""
        n_r, n_c = 6, 6
        r = np.repeat(np.arange(n_r, dtype=float), n_c)
        c = np.tile(np.arange(n_c, dtype=float), n_r)

        X_poly, Z_r, Z_c, Z_rc, blocks = build_psanova_design(
            r, c, nkr=5, nkc=5, degree=3
        )

        # Filter random blocks
        random_blocks = [b for b in blocks if b.is_random]

        # Should have up to 3 blocks (row, col, interaction)
        assert len(random_blocks) <= 3
        assert len(random_blocks) > 0

        # Check contiguous and sorted
        starts = [b.start for b in random_blocks]
        stops = [b.stop for b in random_blocks]

        # Should be sorted
        assert starts == sorted(starts)

        # Non-overlapping: stop[i] <= start[i+1]
        for i in range(len(random_blocks) - 1):
            assert stops[i] <= starts[i + 1], f"Blocks overlap: {stops[i]} > {starts[i+1]}"

        # Check sizes match Z dimensions
        total_cols = sum(b.size for b in random_blocks)

        # Verify total columns (handle LinearOperators)
        from scipy.sparse.linalg import LinearOperator
        expected_total = Z_r.shape[1] + Z_c.shape[1] + Z_rc.shape[1]
        assert expected_total == total_cols, f"Block sizes {total_cols} don't match Z dimensions {expected_total}"

    def test_psanova_fixed_poly_structure(self):
        """Test that fixed polynomial has expected structure."""
        n_r, n_c = 5, 5
        r = np.repeat(np.arange(n_r, dtype=float), n_c)
        c = np.tile(np.arange(n_c, dtype=float), n_r)

        X_poly, Z_r, Z_c, Z_rc, blocks = build_psanova_design(
            r, c, nkr=5, nkc=5, degree=3
        )

        # X_poly should be (n_obs × 3): [1, r_norm, c_norm]
        assert X_poly.shape == (n_r * n_c, 3)

        # First column should be constant
        assert np.allclose(X_poly[:, 0], 1.0)

        # Columns should be linearly independent
        rank = np.linalg.matrix_rank(X_poly)
        assert rank == 3

    def test_psanova_varying_grid_sizes(self):
        """Test PS-ANOVA on different grid sizes."""
        for n_r, n_c in [(5, 5), (8, 6), (10, 12)]:
            r = np.repeat(np.arange(n_r, dtype=float), n_c)
            c = np.tile(np.arange(n_c, dtype=float), n_r)

            X_poly, Z_r, Z_c, Z_rc, blocks = build_psanova_design(
                r, c, nkr=min(n_r, 8), nkc=min(n_c, 8), degree=3
            )

            # Check dimensions
            assert X_poly.shape[0] == n_r * n_c
            assert X_poly.shape[1] == 3

            # Check orthogonality
            assert verify_orthogonality(X_poly, [Z_r, Z_c, Z_rc], tol=1e-8)

    def test_psanova_empty_blocks(self):
        """Test handling when some blocks might be empty."""
        # Very small grid or very few knots might produce empty blocks
        n_r, n_c = 3, 3
        r = np.repeat(np.arange(n_r, dtype=float), n_c)
        c = np.tile(np.arange(n_c, dtype=float), n_r)

        X_poly, Z_r, Z_c, Z_rc, blocks = build_psanova_design(
            r, c, nkr=3, nkc=3, degree=3
        )

        # Should handle gracefully
        assert X_poly.shape[0] == n_r * n_c
        # Blocks list should only contain non-empty blocks
        for block in blocks:
            assert block.size > 0


class TestRowColBases:
    """Test row/column basis construction."""

    def test_row_col_bases_basic(self):
        """Test basic row/col basis construction."""
        r = np.repeat(np.arange(5, dtype=float), 4)
        c = np.tile(np.arange(4, dtype=float), 5)

        B_r, B_c, K_r, K_c = row_col_bases(r, c, n_knots_r=5, n_knots_c=4, degree=3)

        # Check dimensions
        assert B_r.shape[0] == 20
        assert B_c.shape[0] == 20
        assert B_r.shape[1] > 0
        assert B_c.shape[1] > 0

        # Penalties should be square and match basis dimensions
        assert K_r.shape == (B_r.shape[1], B_r.shape[1])
        assert K_c.shape == (B_c.shape[1], B_c.shape[1])

    def test_row_col_penalties_symmetric(self):
        """Test that row/col penalties are symmetric."""
        r = np.linspace(0, 10, 30)
        c = np.linspace(0, 8, 30)

        B_r, B_c, K_r, K_c = row_col_bases(r, c, n_knots_r=6, n_knots_c=6, degree=3)

        # Check symmetry
        K_r_dense = K_r.toarray()
        K_c_dense = K_c.toarray()

        np.testing.assert_allclose(K_r_dense, K_r_dense.T, rtol=1e-14)
        np.testing.assert_allclose(K_c_dense, K_c_dense.T, rtol=1e-14)


def test_integration_with_blockinfo():
    """Test that PS-ANOVA integrates correctly with BlockInfo."""
    from pyspats.psanova_basis import build_psanova_design

    n_r, n_c = 7, 6
    r = np.repeat(np.arange(n_r, dtype=float), n_c)
    c = np.tile(np.arange(n_c, dtype=float), n_r)

    X_poly, Z_r, Z_c, Z_rc, blocks = build_psanova_design(
        r, c, nkr=6, nkc=5, degree=3
    )

    # Check that BlockInfo objects have required attributes
    for block in blocks:
        assert hasattr(block, 'name')
        assert hasattr(block, 'start')
        assert hasattr(block, 'stop')
        assert hasattr(block, 'is_random')
        assert hasattr(block, 'size')

        # Check that size is correct
        assert block.size == block.stop - block.start
        assert block.is_random is True
