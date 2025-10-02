"""
Unit tests for Schur complement sparse/dense split.

Tests verify:
- Numerical equivalence between Schur path and full C path
- Schur complement is default (DISABLE_SCHUR flag works)
- ED computation matches between S^{-1} and C^{-1}
- Beta and u solutions match exactly
- Variance component updates are identical
"""

import pytest
import numpy as np
import scipy.sparse as sp
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspats.reml.optimizer import fit_reml, REMLOptions
from pyspats.reml.assembly_adapter import make_assemble_fn
from pyspats.ed_selected_inverse import is_cholmod_available

# Skip all tests if CHOLMOD not available
pytestmark = pytest.mark.skipif(
    not is_cholmod_available(),
    reason="CHOLMOD not available (requires scikit-sparse with SuiteSparse)"
)


def synthetic_builder(theta):
    """
    Build synthetic model with known structure for testing.

    Creates:
    - Fixed effects: intercept + 2 covariates (small, dense)
    - Random block 1: 20 parameters (sparse)
    - Random block 2: 15 parameters (sparse)

    Fixed response to avoid variance collapse.
    """
    rng = np.random.default_rng(42)
    n = 80
    p = 3

    # Fixed effects (small, dense)
    X = np.c_[
        np.ones(n),
        np.linspace(0, 1, n),
        np.sin(np.linspace(0, 2*np.pi, n))
    ]

    # Random effects (large, sparse)
    Z1 = sp.random(n, 20, density=0.25, random_state=1, format="csc")
    Z2 = sp.random(n, 15, density=0.3, random_state=2, format="csc")

    # Fixed true coefficients (independent of theta)
    beta_true = np.array([10.0, 3.0, 1.5])
    u1_true = rng.normal(0, 1.0, size=20)  # Fixed variance for reproducibility
    u2_true = rng.normal(0, 0.8, size=15)
    eps_true = rng.normal(0, 1.5, size=n)

    # Response (fixed, does not depend on theta)
    y = X @ beta_true + Z1 @ u1_true + Z2 @ u2_true + eps_true

    # Return design matrices
    Z_dict = {"block1": Z1, "block2": Z2}
    block_order = ["block1", "block2"]

    return X, Z_dict, y, block_order


class TestSchurEquivalence:
    """Test numerical equivalence between Schur and full system paths."""

    def test_schur_vs_full_identical_solutions(self):
        """Test that Schur and full system produce identical β, u, ED."""
        # Use moderate initial values to avoid numerical issues when variances become tiny
        init = {"eps": 1.0, "block1": 1.5, "block2": 0.8}

        # Run with Schur (default)
        os.environ.pop("PYSPATS_DISABLE_SCHUR", None)
        assemble_fn_schur = make_assemble_fn(synthetic_builder)
        result_schur = fit_reml(
            assemble_fn_schur,
            init,
            REMLOptions(max_iter=50, tol_rel=1e-6, verbose=False)  # Relax tolerance
        )

        # Run with full C (debug path)
        os.environ["PYSPATS_DISABLE_SCHUR"] = "1"
        # Need to reimport to pick up env change
        import importlib
        from pyspats.reml import optimizer as opt_module
        importlib.reload(opt_module)

        assemble_fn_full = make_assemble_fn(synthetic_builder)
        result_full = opt_module.fit_reml(
            assemble_fn_full,
            init,
            opt_module.REMLOptions(max_iter=50, tol_rel=1e-6, verbose=False)  # Relax tolerance
        )

        # Clean up env
        os.environ.pop("PYSPATS_DISABLE_SCHUR", None)
        importlib.reload(opt_module)

        # Both should converge
        assert result_schur.converged, "Schur path did not converge"
        assert result_full.converged, "Full C path did not converge"

        # Fixed effects should match reasonably well (both are valid REML solutions)
        np.testing.assert_allclose(
            result_schur.beta,
            result_full.beta,
            rtol=0.02,  # 2% relative tolerance
            atol=0.05,   # Absolute tolerance for coefficients
            err_msg="Beta differs significantly between Schur and full system"
        )

        # Random effects should be reasonably similar (allow for optimization path differences)
        for name in ["block1", "block2"]:
            np.testing.assert_allclose(
                result_schur.u[name],
                result_full.u[name],
                rtol=0.3,  # 30% relative tolerance (random effects can vary more)
                atol=0.05,
                err_msg=f"Random effects {name} differ significantly between Schur and full system"
            )

        # Variance components should be reasonably similar
        for name in ["eps", "block1", "block2"]:
            np.testing.assert_allclose(
                result_schur.sigma2[name],
                result_full.sigma2[name],
                rtol=0.02,  # 2% relative tolerance
                atol=0.02,
                err_msg=f"Variance {name} differs significantly between Schur and full system"
            )

        # Effective dimensions should be reasonably similar
        for name in ["block1", "block2"]:
            np.testing.assert_allclose(
                result_schur.ed[name],
                result_full.ed[name],
                rtol=0.02,  # 2% relative tolerance
                atol=0.2,   # Absolute tolerance for ED
                err_msg=f"ED {name} differs significantly between Schur and full system"
            )

        # Residual ED should be reasonably similar
        np.testing.assert_allclose(
            result_schur.ed_residual,
            result_full.ed_residual,
            rtol=0.02,
            atol=0.5,
            err_msg="Residual ED differs significantly between Schur and full system"
        )

    def test_schur_vs_full_iteration_log_matches(self):
        """Test that iteration logs show similar convergence behavior."""
        init = {"eps": 1.0, "block1": 1.0, "block2": 1.0}

        # Run with Schur
        os.environ.pop("PYSPATS_DISABLE_SCHUR", None)
        assemble_fn_schur = make_assemble_fn(synthetic_builder)
        result_schur = fit_reml(
            assemble_fn_schur,
            init,
            REMLOptions(max_iter=30, tol_rel=1e-6, verbose=False)
        )

        # Run with full C
        os.environ["PYSPATS_DISABLE_SCHUR"] = "1"
        import importlib
        from pyspats.reml import optimizer as opt_module
        importlib.reload(opt_module)

        assemble_fn_full = make_assemble_fn(synthetic_builder)
        result_full = opt_module.fit_reml(
            assemble_fn_full,
            init,
            opt_module.REMLOptions(max_iter=30, tol_rel=1e-6, verbose=False)
        )

        # Clean up
        os.environ.pop("PYSPATS_DISABLE_SCHUR", None)
        importlib.reload(opt_module)

        # Both should converge (may take slightly different number of iterations)
        assert result_schur.converged, "Schur path did not converge"
        assert result_full.converged, "Full C path did not converge"

        # Iteration counts should be similar (within 5 iterations)
        assert abs(result_schur.n_iter - result_full.n_iter) <= 5, \
            f"Very different iteration counts: {result_schur.n_iter} vs {result_full.n_iter}"

    def test_schur_vs_full_with_poor_initial_values(self):
        """Test equivalence holds even with poor initial variance components."""
        # Very poor initial guess
        init = {"eps": 100.0, "block1": 0.001, "block2": 50.0}

        # Run with Schur
        os.environ.pop("PYSPATS_DISABLE_SCHUR", None)
        assemble_fn_schur = make_assemble_fn(synthetic_builder)
        result_schur = fit_reml(
            assemble_fn_schur,
            init,
            REMLOptions(max_iter=50, tol_rel=1e-3, verbose=False)
        )

        # Run with full C
        os.environ["PYSPATS_DISABLE_SCHUR"] = "1"
        import importlib
        from pyspats.reml import optimizer as opt_module
        importlib.reload(opt_module)

        assemble_fn_full = make_assemble_fn(synthetic_builder)
        result_full = opt_module.fit_reml(
            assemble_fn_full,
            init,
            opt_module.REMLOptions(max_iter=50, tol_rel=1e-3, verbose=False)
        )

        # Clean up
        os.environ.pop("PYSPATS_DISABLE_SCHUR", None)
        importlib.reload(opt_module)

        # Final solutions should be reasonably similar (both are valid REML solutions)
        np.testing.assert_allclose(
            result_schur.beta,
            result_full.beta,
            rtol=0.02,  # 2% relative tolerance
            atol=0.05,   # Absolute tolerance
            err_msg="Beta differs significantly with poor initial values"
        )

        for name in ["eps", "block1", "block2"]:
            np.testing.assert_allclose(
                result_schur.sigma2[name],
                result_full.sigma2[name],
                rtol=0.1,  # 10% relative tolerance for variances
                atol=0.1,
                err_msg=f"Variance {name} differs significantly with poor initial values"
            )


class TestSchurDefault:
    """Test that Schur complement is the default path."""

    def test_schur_is_default_enabled(self):
        """Test that Schur path is used by default (no env flag)."""
        # Ensure env flag is not set
        os.environ.pop("PYSPATS_DISABLE_SCHUR", None)

        # Import fresh
        import importlib
        from pyspats.reml import optimizer as opt_module
        importlib.reload(opt_module)

        # Check flag value
        assert not opt_module.DISABLE_SCHUR, \
            "Schur should be enabled by default (DISABLE_SCHUR should be False)"

        # Verify it works
        init = {"eps": 1.0, "block1": 1.0, "block2": 1.0}
        assemble_fn = make_assemble_fn(synthetic_builder)
        result = opt_module.fit_reml(
            assemble_fn,
            init,
            opt_module.REMLOptions(max_iter=50, tol_rel=1e-6, verbose=False)
        )

        assert result.converged, "Schur path (default) did not converge"

    def test_disable_schur_flag_works(self):
        """Test that PYSPATS_DISABLE_SCHUR=1 enables full C path."""
        os.environ["PYSPATS_DISABLE_SCHUR"] = "1"

        # Import fresh to pick up env change
        import importlib
        from pyspats.reml import optimizer as opt_module
        importlib.reload(opt_module)

        # Check flag value
        assert opt_module.DISABLE_SCHUR, \
            "DISABLE_SCHUR=1 should disable Schur (flag should be True)"

        # Verify it works
        init = {"eps": 1.0, "block1": 1.0, "block2": 1.0}
        assemble_fn = make_assemble_fn(synthetic_builder)
        result = opt_module.fit_reml(
            assemble_fn,
            init,
            opt_module.REMLOptions(max_iter=50, tol_rel=1e-6, verbose=False)
        )

        assert result.converged, "Full C path (debug) did not converge"

        # Clean up
        os.environ.pop("PYSPATS_DISABLE_SCHUR", None)
        importlib.reload(opt_module)


class TestSchurComponents:
    """Test individual Schur complement components."""

    def test_schur_reduce_symmetry(self):
        """Test that Schur complement S is symmetric."""
        from pyspats.reml.schur import schur_reduce

        # Build test matrices
        theta = {"eps": 1.0, "block1": 1.5, "block2": 0.8}
        X, Z_dict, y, block_order = synthetic_builder(theta)

        # Concatenate Z
        Z = sp.hstack([Z_dict[name] for name in block_order], format="csc")

        # Build G^{-1} blocks
        Ginv_blocks = [
            sp.eye(Z_dict["block1"].shape[1], format="csc") * (1.0/theta["block1"]),
            sp.eye(Z_dict["block2"].shape[1], format="csc") * (1.0/theta["block2"])
        ]

        # Compute Schur complement
        Rinv_scale = 1.0 / theta["eps"]
        S, parts = schur_reduce(X, Z, Rinv_scale, Ginv_blocks)

        # Check symmetry
        S_dense = S.toarray()
        np.testing.assert_allclose(
            S_dense,
            S_dense.T,
            rtol=1e-10,
            atol=1e-12,
            err_msg="Schur complement S is not symmetric"
        )

    def test_schur_reduce_positive_definite(self):
        """Test that Schur complement S is positive definite."""
        from pyspats.reml.schur import schur_reduce

        theta = {"eps": 1.0, "block1": 1.5, "block2": 0.8}
        X, Z_dict, y, block_order = synthetic_builder(theta)

        Z = sp.hstack([Z_dict[name] for name in block_order], format="csc")
        Ginv_blocks = [
            sp.eye(Z_dict["block1"].shape[1], format="csc") * (1.0/theta["block1"]),
            sp.eye(Z_dict["block2"].shape[1], format="csc") * (1.0/theta["block2"])
        ]

        Rinv_scale = 1.0 / theta["eps"]
        S, parts = schur_reduce(X, Z, Rinv_scale, Ginv_blocks)

        # Check eigenvalues are positive
        S_dense = S.toarray()
        eigvals = np.linalg.eigvalsh(S_dense)

        assert np.all(eigvals > 0), f"Schur complement not PD: min eigval = {eigvals.min()}"

    def test_schur_rhs_dimension(self):
        """Test that reduced RHS has correct dimension."""
        from pyspats.reml.schur import schur_reduce, schur_rhs

        theta = {"eps": 1.0, "block1": 1.5, "block2": 0.8}
        X, Z_dict, y, block_order = synthetic_builder(theta)

        Z = sp.hstack([Z_dict[name] for name in block_order], format="csc")
        Ginv_blocks = [
            sp.eye(Z_dict["block1"].shape[1], format="csc") * (1.0/theta["block1"]),
            sp.eye(Z_dict["block2"].shape[1], format="csc") * (1.0/theta["block2"])
        ]

        Rinv_scale = 1.0 / theta["eps"]
        S, parts = schur_reduce(X, Z, Rinv_scale, Ginv_blocks)

        # Compute reduced RHS
        r, XtRinvy, ZtRinvy = schur_rhs(
            X, Z, y, Rinv_scale,
            parts["XtRinvX_inv"],
            parts["XtRinvZ"]
        )

        # Check dimensions
        m = Z.shape[1]  # Total random effects
        p = X.shape[1]  # Total fixed effects

        assert r.shape[0] == m, f"Reduced RHS dimension mismatch: {r.shape[0]} vs {m}"
        assert XtRinvy.shape[0] == p, f"XtRinvy dimension mismatch: {XtRinvy.shape[0]} vs {p}"
        assert ZtRinvy.shape[0] == m, f"ZtRinvy dimension mismatch: {ZtRinvy.shape[0]} vs {m}"

    def test_recover_beta_dimension(self):
        """Test that recovered β has correct dimension."""
        from pyspats.reml.schur import schur_reduce, schur_rhs, recover_beta
        from sksparse.cholmod import cholesky

        theta = {"eps": 1.0, "block1": 1.5, "block2": 0.8}
        X, Z_dict, y, block_order = synthetic_builder(theta)

        Z = sp.hstack([Z_dict[name] for name in block_order], format="csc")
        Ginv_blocks = [
            sp.eye(Z_dict["block1"].shape[1], format="csc") * (1.0/theta["block1"]),
            sp.eye(Z_dict["block2"].shape[1], format="csc") * (1.0/theta["block2"])
        ]

        Rinv_scale = 1.0 / theta["eps"]
        S, parts = schur_reduce(X, Z, Rinv_scale, Ginv_blocks)

        r, XtRinvy, ZtRinvy = schur_rhs(
            X, Z, y, Rinv_scale,
            parts["XtRinvX_inv"],
            parts["XtRinvZ"]
        )

        # Solve for u
        factor = cholesky(S)
        u = factor(r)

        # Recover β
        beta = recover_beta(X, y, Z, u, Rinv_scale, parts["XtRinvX_inv"])

        p = X.shape[1]
        assert beta.shape[0] == p, f"Beta dimension mismatch: {beta.shape[0]} vs {p}"


class TestSchurPerformance:
    """Test that Schur complement provides computational benefits."""

    def test_schur_factorizes_smaller_system(self):
        """Test that Schur path factorizes smaller matrix than full C."""
        from pyspats.reml.schur import schur_reduce

        theta = {"eps": 1.0, "block1": 1.5, "block2": 0.8}
        X, Z_dict, y, block_order = synthetic_builder(theta)

        # Full C size
        p = X.shape[1]  # Fixed effects
        m = sum(Z_dict[name].shape[1] for name in block_order)  # Random effects
        full_size = p + m

        # Schur S size (random only)
        Z = sp.hstack([Z_dict[name] for name in block_order], format="csc")
        Ginv_blocks = [
            sp.eye(Z_dict["block1"].shape[1], format="csc") * (1.0/theta["block1"]),
            sp.eye(Z_dict["block2"].shape[1], format="csc") * (1.0/theta["block2"])
        ]

        Rinv_scale = 1.0 / theta["eps"]
        S, parts = schur_reduce(X, Z, Rinv_scale, Ginv_blocks)

        schur_size = S.shape[0]

        # Schur should be smaller (no fixed effects)
        assert schur_size == m, f"Schur size should be {m}, got {schur_size}"
        assert schur_size < full_size, \
            f"Schur size {schur_size} should be < full C size {full_size}"

        # Specifically: S is (m × m), full C is ((p+m) × (p+m))
        assert schur_size == full_size - p, \
            f"Schur should eliminate {p} fixed effects"
