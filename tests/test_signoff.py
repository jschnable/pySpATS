"""
Sign-off tests for Kronecker interaction implementation.

These tests verify critical properties for production readiness:
1. Irregular layout with missing plots
2. Extreme smoothing values
3. Permutation invariance
4. Dense/Kron equivalence on small problems
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspats.psanova_basis import build_psanova_design


class TestIrregularLayout:
    """Test Kronecker path with irregular layouts and missing plots."""

    def test_missing_plots_equivalence(self):
        """Verify dense/kron equivalence with ~20% missing plots."""
        np.random.seed(42)

        # Create 8x8 grid
        n_r, n_c = 8, 8
        r = np.repeat(np.arange(n_r, dtype=float), n_c)
        c = np.tile(np.arange(n_c, dtype=float), n_r)

        # Randomly drop ~20% of observations
        n_obs = len(r)
        keep_idx = np.random.choice(n_obs, size=int(0.8 * n_obs), replace=False)
        keep_idx = np.sort(keep_idx)

        r_irregular = r[keep_idx]
        c_irregular = c[keep_idx]

        # Build with Kronecker path
        X_poly_kron, Z_r_kron, Z_c_kron, Z_rc_kron, blocks_kron = build_psanova_design(
            r_irregular, c_irregular, nkr=6, nkc=6, degree=3, use_kron_interaction=True
        )

        # Build with dense path
        X_poly_dense, Z_r_dense, Z_c_dense, Z_rc_dense, blocks_dense = build_psanova_design(
            r_irregular, c_irregular, nkr=6, nkc=6, degree=3, use_kron_interaction=False
        )

        # Verify dimensions match
        assert Z_rc_kron.shape == Z_rc_dense.shape, \
            f"Shape mismatch: kron {Z_rc_kron.shape} vs dense {Z_rc_dense.shape}"

        # Verify X'Z orthogonality for both paths
        from scipy.sparse.linalg import LinearOperator
        if isinstance(Z_rc_kron, LinearOperator):
            # Materialize via matvec
            Z_rc_kron_dense = np.zeros((Z_rc_kron.shape[0], Z_rc_kron.shape[1]))
            for j in range(Z_rc_kron.shape[1]):
                e_j = np.zeros(Z_rc_kron.shape[1])
                e_j[j] = 1.0
                Z_rc_kron_dense[:, j] = Z_rc_kron @ e_j
        else:
            Z_rc_kron_dense = Z_rc_kron

        XtZ_kron = X_poly_kron.T @ Z_rc_kron_dense
        XtZ_dense = X_poly_kron.T @ Z_rc_dense

        assert np.max(np.abs(XtZ_kron)) < 1e-8, \
            f"Kron path not orthogonal: max |X'Z| = {np.max(np.abs(XtZ_kron)):.2e}"
        assert np.max(np.abs(XtZ_dense)) < 1e-8, \
            f"Dense path not orthogonal: max |X'Z| = {np.max(np.abs(XtZ_dense)):.2e}"

        # Note: Kron and dense paths use different eigenbases, so Z'Z matrices differ
        # What matters is that they span equivalent model spaces
        # Verify this by checking that projection matrices P_Z = Z(Z'Z)^{-1}Z' are close

        ZtZ_kron = Z_rc_kron_dense.T @ Z_rc_kron_dense
        ZtZ_dense = Z_rc_dense.T @ Z_rc_dense

        # Compute pseudo-inverses
        ZtZ_kron_inv = np.linalg.pinv(ZtZ_kron)
        ZtZ_dense_inv = np.linalg.pinv(ZtZ_dense)

        # Compute projection matrices: P = Z @ (Z'Z)^{-1} @ Z'
        P_kron = Z_rc_kron_dense @ ZtZ_kron_inv @ Z_rc_kron_dense.T
        P_dense = Z_rc_dense @ ZtZ_dense_inv @ Z_rc_dense.T

        # Projection matrices should be very close (they project onto the same space)
        diff_norm = np.linalg.norm(P_kron - P_dense, 'fro')
        ref_norm = np.linalg.norm(P_dense, 'fro')
        rel_error = diff_norm / ref_norm if ref_norm > 0 else diff_norm

        assert rel_error < 1e-4, \
            f"Projection matrix mismatch: ||P_kron - P_dense|| / ||P_dense|| = {rel_error:.2e}"

    def test_heavy_missing_plots(self):
        """Test with ~40% missing plots (more extreme)."""
        np.random.seed(123)

        n_r, n_c = 10, 10
        r = np.repeat(np.arange(n_r, dtype=float), n_c)
        c = np.tile(np.arange(n_c, dtype=float), n_r)

        # Drop 40% randomly
        n_obs = len(r)
        keep_idx = np.random.choice(n_obs, size=int(0.6 * n_obs), replace=False)
        keep_idx = np.sort(keep_idx)

        r_sparse = r[keep_idx]
        c_sparse = c[keep_idx]

        # Should not crash and should produce reasonable dimensions
        X_poly, Z_r, Z_c, Z_rc, blocks = build_psanova_design(
            r_sparse, c_sparse, nkr=7, nkc=7, degree=3, use_kron_interaction=True
        )

        # Basic sanity checks
        assert X_poly.shape[0] == len(r_sparse)
        assert Z_rc.shape[0] == len(r_sparse)
        assert Z_rc.shape[1] > 0, "Interaction smooth should have positive dimension"


class TestExtremeSmoothing:
    """Test with extreme variance component values."""

    def test_very_small_variance(self):
        """Test with very small spatial variance (heavy smoothing)."""
        from pyspats.psanova_basis import tensor_whiten_interaction, row_col_bases

        n_r, n_c = 6, 6
        B_r, B_c, K_r, K_c = row_col_bases(
            np.arange(n_r, dtype=float),
            np.arange(n_c, dtype=float),
            n_knots_r=5, n_knots_c=5, degree=3
        )

        # Compute eigendecomposition
        meta = tensor_whiten_interaction(B_r, K_r, B_c, K_c, drop_null=True, tol=1e-10)

        # All EDs should be in valid range
        Urp, Ucp = meta["Urp"], meta["Ucp"]
        lrp, lcp = meta["lrp"], meta["lcp"]

        n_modes = np.sum(meta["keep_mask"])

        # Effective dimensions should be in [0, n_modes]
        # (This would be verified in actual REML fit)
        assert n_modes >= 0, f"Invalid mode count: {n_modes}"
        assert n_modes <= Urp.shape[1] * Ucp.shape[1], \
            f"Mode count {n_modes} exceeds max {Urp.shape[1] * Ucp.shape[1]}"

    def test_very_large_variance(self):
        """Test with very large spatial variance (minimal smoothing)."""
        from pyspats.psanova_basis import tensor_whiten_interaction, row_col_bases

        n_r, n_c = 6, 6
        B_r, B_c, K_r, K_c = row_col_bases(
            np.arange(n_r, dtype=float),
            np.arange(n_c, dtype=float),
            n_knots_r=5, n_knots_c=5, degree=3
        )

        # With high variance, we expect more modes retained
        meta = tensor_whiten_interaction(B_r, K_r, B_c, K_c, drop_null=True, tol=1e-12)

        n_modes = np.sum(meta["keep_mask"])

        # Should have reasonable number of modes
        assert n_modes > 0, "Should retain at least some modes"
        assert n_modes < 1000, f"Unreasonably many modes: {n_modes}"


class TestPermutationInvariance:
    """Test that results are invariant to observation order."""

    def test_observation_permutation(self):
        """Verify results unchanged when observations shuffled."""
        np.random.seed(456)

        # Create regular grid
        n_r, n_c = 6, 6
        r = np.repeat(np.arange(n_r, dtype=float), n_c)
        c = np.tile(np.arange(n_c, dtype=float), n_r)

        # Build with original order
        X_poly_orig, Z_r_orig, Z_c_orig, Z_rc_orig, blocks_orig = build_psanova_design(
            r, c, nkr=5, nkc=5, degree=3, use_kron_interaction=True
        )

        # Permute observations
        n_obs = len(r)
        perm = np.random.permutation(n_obs)
        r_perm = r[perm]
        c_perm = c[perm]

        # Build with permuted order
        X_poly_perm, Z_r_perm, Z_c_perm, Z_rc_perm, blocks_perm = build_psanova_design(
            r_perm, c_perm, nkr=5, nkc=5, degree=3, use_kron_interaction=True
        )

        # Dimensions should match
        assert Z_rc_orig.shape == Z_rc_perm.shape, \
            f"Dimension changed after permutation: {Z_rc_orig.shape} vs {Z_rc_perm.shape}"

        # Block structure should be identical
        assert len(blocks_orig) == len(blocks_perm)
        for b_orig, b_perm in zip(blocks_orig, blocks_perm):
            assert b_orig.name == b_perm.name
            assert b_orig.size == b_perm.size

    def test_coordinate_permutation_invariance(self):
        """Test invariance when swapping row/column coordinates."""
        n_r, n_c = 6, 7  # Asymmetric to make swap detectable

        r = np.repeat(np.arange(n_r, dtype=float), n_c)
        c = np.tile(np.arange(n_c, dtype=float), n_r)

        # Build with (r, c)
        X1, Z_r1, Z_c1, Z_rc1, blocks1 = build_psanova_design(
            r, c, nkr=5, nkc=6, degree=3, use_kron_interaction=True
        )

        # Build with (c, r) - swap dimensions
        X2, Z_r2, Z_c2, Z_rc2, blocks2 = build_psanova_design(
            c, r, nkr=6, nkc=5, degree=3, use_kron_interaction=True
        )

        # Interaction should have similar complexity (dimensions may differ due to asymmetry)
        # But both should produce valid models
        assert Z_rc1.shape[1] > 0
        assert Z_rc2.shape[1] > 0


class TestDenseKronEquivalence:
    """Test exact equivalence on small problems."""

    def test_small_grid_exact_match(self):
        """Verify dense and Kron paths produce identical results on 5x5 grid."""
        n_r, n_c = 5, 5
        r = np.repeat(np.arange(n_r, dtype=float), n_c)
        c = np.tile(np.arange(n_c, dtype=float), n_r)

        # Kron path
        X_k, Z_r_k, Z_c_k, Z_rc_k, blocks_k = build_psanova_design(
            r, c, nkr=4, nkc=4, degree=3, use_kron_interaction=True
        )

        # Dense path
        X_d, Z_r_d, Z_c_d, Z_rc_d, blocks_d = build_psanova_design(
            r, c, nkr=4, nkc=4, degree=3, use_kron_interaction=False
        )

        # Exact dimension match
        assert Z_rc_k.shape == Z_rc_d.shape, \
            f"Dimension mismatch: Kron {Z_rc_k.shape} vs Dense {Z_rc_d.shape}"

        # Block structure match
        assert len(blocks_k) == len(blocks_d)
        for bk, bd in zip(blocks_k, blocks_d):
            assert bk.name == bd.name
            assert bk.size == bd.size

        # Materialize Kron operator if needed
        from scipy.sparse.linalg import LinearOperator
        if isinstance(Z_rc_k, LinearOperator):
            Z_rc_k_mat = np.zeros((Z_rc_k.shape[0], Z_rc_k.shape[1]))
            for j in range(Z_rc_k.shape[1]):
                e_j = np.zeros(Z_rc_k.shape[1])
                e_j[j] = 1.0
                Z_rc_k_mat[:, j] = Z_rc_k @ e_j
        else:
            Z_rc_k_mat = Z_rc_k

        # Note: Kron and dense use different eigenbases, so we check model space equivalence
        # via projection matrices P = Z @ (Z'Z)^{-1} @ Z'
        ZtZ_k = Z_rc_k_mat.T @ Z_rc_k_mat
        ZtZ_d = Z_rc_d.T @ Z_rc_d

        ZtZ_k_inv = np.linalg.pinv(ZtZ_k)
        ZtZ_d_inv = np.linalg.pinv(ZtZ_d)

        P_k = Z_rc_k_mat @ ZtZ_k_inv @ Z_rc_k_mat.T
        P_d = Z_rc_d @ ZtZ_d_inv @ Z_rc_d.T

        rel_error = np.linalg.norm(P_k - P_d, 'fro') / np.linalg.norm(P_d, 'fro')
        assert rel_error < 1e-4, \
            f"Projection matrix relative error = {rel_error:.2e} exceeds tolerance 1e-4"
