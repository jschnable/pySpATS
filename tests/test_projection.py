"""
Tests for polynomial projection wrapper (LeftProjectedLO).
"""

import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator

from pyspats.spatial.projection import LeftProjectedLO
from pyspats.psanova_basis import build_psanova_design


class TestLeftProjectedLO:
    """Test the LeftProjectedLO polynomial projection wrapper."""

    def test_projection_wrapper_basic(self):
        """Test that LeftProjectedLO correctly projects out polynomial space."""
        n = 100
        m = 50

        # Create simple base operator (identity-like)
        def mv(x):
            return np.tile(x.mean(), n)  # Broadcasts to constant (in polynomial space)

        def rmv(y):
            return np.full(m, y.sum() / n)

        base = LinearOperator((n, m), matvec=mv, rmatvec=rmv)

        # Polynomial space: [1, x, x^2]
        x = np.linspace(0, 1, n)
        X_poly = np.column_stack([np.ones(n), x, x**2])
        Qx, _ = np.linalg.qr(X_poly, mode='reduced')

        # Wrap with projection
        proj_op = LeftProjectedLO(base, Qx)

        # Test matvec: should project out polynomial components
        v = np.random.randn(m)
        y = proj_op @ v

        # Check orthogonality: X_poly^T @ y should be ~0
        orthog = np.abs(X_poly.T @ y)
        assert np.max(orthog) < 1e-10, f"Not orthogonal: max |X'y| = {np.max(orthog)}"

    def test_projection_wrapper_rmatvec(self):
        """Test that rmatvec correctly projects on the left."""
        n = 100
        m = 50

        # Create random base operator
        A = np.random.randn(n, m) * 0.1
        base = LinearOperator((n, m), matvec=lambda x: A @ x, rmatvec=lambda y: A.T @ y)

        # Polynomial space
        x = np.linspace(0, 1, n)
        X_poly = np.column_stack([np.ones(n), x])
        Qx, _ = np.linalg.qr(X_poly, mode='reduced')

        # Wrap with projection
        proj_op = LeftProjectedLO(base, Qx)

        # Test rmatvec
        w = np.random.randn(n)
        result = proj_op.rmatvec(w)

        # Should be equivalent to base.T @ (P_perp @ w)
        w_perp = w - Qx @ (Qx.T @ w)
        expected = A.T @ w_perp

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_projection_preserves_orthogonal_components(self):
        """Test that components already orthogonal to polynomial space are preserved."""
        n = 100
        m = 50

        # Create base operator that returns orthogonal vectors
        x = np.linspace(0, 1, n)
        X_poly = np.column_stack([np.ones(n), x])
        Qx, _ = np.linalg.qr(X_poly, mode='reduced')

        # Create a vector orthogonal to polynomial space
        v_orth = np.sin(2 * np.pi * x)
        v_orth -= Qx @ (Qx.T @ v_orth)  # Ensure orthogonality

        def mv(x):
            return v_orth * x[0]

        def rmv(y):
            return np.array([v_orth @ y])

        base = LinearOperator((n, 1), matvec=mv, rmatvec=rmv)
        proj_op = LeftProjectedLO(base, Qx)

        # Test that orthogonal component is preserved
        coef = np.array([2.5])
        y_base = base @ coef
        y_proj = proj_op @ coef

        # Should be very close (projection of already-orthogonal vector)
        np.testing.assert_allclose(y_proj, y_base, rtol=1e-10)


class TestKroneckerWithProjection:
    """Test Kronecker interaction with polynomial projection."""

    def test_kron_interaction_orthogonality(self):
        """Test that Kronecker interaction with projection is orthogonal to X_poly."""
        # Small grid for testing
        r = np.repeat(np.arange(10), 10)
        c = np.tile(np.arange(10), 10)

        # Build with Kronecker path
        X_poly, Z_r, Z_c, Z_rc, blocks = build_psanova_design(
            r, c, nkr=8, nkc=8, use_kron_interaction=True
        )

        # Z_rc should be a LinearOperator (wrapped with projection)
        assert isinstance(Z_rc, LinearOperator), "Expected LinearOperator for Kronecker path"

        # Test orthogonality: X_poly^T @ Z_rc should be ~0
        # Sample a few random coefficient vectors
        n_rc = Z_rc.shape[1]
        for _ in range(5):
            alpha = np.random.randn(n_rc)
            z = Z_rc @ alpha

            # Check orthogonality
            orthog = np.abs(X_poly.T @ z)
            max_orthog = np.max(orthog)

            assert max_orthog < 1e-8, (
                f"Kronecker interaction not orthogonal to X_poly: "
                f"max |X'Z| = {max_orthog:.3e}"
            )

    def test_kron_vs_dense_orthogonality_parity(self):
        """Test that Kronecker and dense paths both achieve orthogonality."""
        r = np.repeat(np.arange(8), 8)
        c = np.tile(np.arange(8), 8)

        # Kronecker path
        X_poly_k, Z_r_k, Z_c_k, Z_rc_k, blocks_k = build_psanova_design(
            r, c, nkr=6, nkc=6, use_kron_interaction=True
        )

        # Dense path
        X_poly_d, Z_r_d, Z_c_d, Z_rc_d, blocks_d = build_psanova_design(
            r, c, nkr=6, nkc=6, use_kron_interaction=False
        )

        # Check row-smooth orthogonality (should be same for both)
        for Z_r in [Z_r_k, Z_r_d]:
            orthog_r = np.abs(X_poly_k.T @ Z_r)
            assert np.max(orthog_r) < 1e-8

        # Check col-smooth orthogonality (should be same for both)
        for Z_c in [Z_c_k, Z_c_d]:
            orthog_c = np.abs(X_poly_k.T @ Z_c)
            assert np.max(orthog_c) < 1e-8

        # Check interaction orthogonality for Kronecker path
        n_rc_k = Z_rc_k.shape[1]
        alpha_k = np.random.randn(n_rc_k)
        z_k = Z_rc_k @ alpha_k
        orthog_k = np.abs(X_poly_k.T @ z_k)
        max_orthog_k = np.max(orthog_k)

        # Check interaction orthogonality for dense path
        orthog_d = np.abs(X_poly_d.T @ Z_rc_d)
        max_orthog_d = np.max(orthog_d)

        # Both should be orthogonal
        assert max_orthog_k < 1e-8, f"Kronecker not orthogonal: {max_orthog_k:.3e}"
        assert max_orthog_d < 1e-8, f"Dense not orthogonal: {max_orthog_d:.3e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
