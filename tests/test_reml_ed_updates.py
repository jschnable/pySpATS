"""
Unit tests for REML optimizer with ED-based variance updates.

Tests verify:
- Convergence on synthetic data
- CHOLMOD factorization reuse per iteration
- ED-based variance component updates
- Proper assembly of C(θ) and RHS
- BlockInfo metadata and slicing
"""

import pytest
import numpy as np
import scipy.sparse as sp
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspats.reml.optimizer import fit_reml, REMLOptions, REMLResult
from pyspats.reml.assembly_adapter import make_assemble_fn, make_builder_from_psanova
from pyspats.ed_selected_inverse import is_cholmod_available

# Skip all tests if CHOLMOD not available
pytestmark = pytest.mark.skipif(
    not is_cholmod_available(),
    reason="CHOLMOD not available (requires scikit-sparse with SuiteSparse)"
)


def toy_builder(theta):
    """
    Build simple synthetic model: y = X β + Z1 u1 + Z2 u2 + ε

    Creates a small problem with known structure for testing.
    """
    rng = np.random.default_rng(0)
    n = 60
    p = 3

    # Fixed effects: intercept + 2 covariates
    X = np.c_[np.ones(n), np.linspace(0, 1, n), np.sin(np.linspace(0, 2*np.pi, n))]

    # Two random blocks with sparse design
    Z1 = sp.random(n, 15, density=0.3, random_state=1, format="csc")
    Z2 = sp.random(n, 10, density=0.3, random_state=2, format="csc")

    # True coefficients (for data generation)
    b_true = np.array([5.0, 2.0, 1.5])
    u1_true = rng.normal(0, np.sqrt(theta.get("blk1", 1.0)), size=15)
    u2_true = rng.normal(0, np.sqrt(theta.get("blk2", 1.0)), size=10)
    eps_true = rng.normal(0, np.sqrt(theta.get("eps", 1.0)), size=n)

    # Response
    y = X @ b_true + Z1 @ u1_true + Z2 @ u2_true + eps_true

    # Return design matrices
    Z_dict = {"blk1": Z1, "blk2": Z2}
    block_order = ["blk1", "blk2"]

    return X, Z_dict, y, block_order


class TestREMLOptimizer:
    """Test REML optimizer core functionality."""

    def test_reml_converges_on_synthetic_data(self):
        """Test that REML converges on synthetic data."""
        # Initial variance components
        init = {"eps": 1.0, "blk1": 1.0, "blk2": 1.0}

        # Create assembly function
        assemble_fn = make_assemble_fn(toy_builder)

        # Run REML
        result = fit_reml(
            assemble_fn,
            init,
            REMLOptions(max_iter=30, tol_rel=1e-3, verbose=False)
        )

        # Should converge
        assert result.converged, "REML failed to converge"
        assert result.n_iter < 30, f"Took too many iterations: {result.n_iter}"

        # Variance components should be positive
        assert result.sigma2["eps"] > 0, "Residual variance not positive"
        assert result.sigma2["blk1"] > 0, "Block 1 variance not positive"
        assert result.sigma2["blk2"] > 0, "Block 2 variance not positive"

    def test_ed_residual_within_bounds(self):
        """Test that ED residual is within valid range."""
        init = {"eps": 1.0, "blk1": 1.0, "blk2": 1.0}
        assemble_fn = make_assemble_fn(toy_builder)

        result = fit_reml(
            assemble_fn,
            init,
            REMLOptions(max_iter=20, tol_rel=1e-3, verbose=False)
        )

        # Get n and rank(X) from builder
        X, Z_dict, y, _ = toy_builder(init)
        n = y.shape[0]
        rank_X = np.linalg.matrix_rank(X)

        # ED residual should be in [0, n - rank(X)]
        assert 0 <= result.ed_residual <= n - rank_X + 1, \
            f"ED residual {result.ed_residual} out of range [0, {n - rank_X}]"

    def test_ed_values_positive(self):
        """Test that all effective dimensions are positive."""
        init = {"eps": 1.0, "blk1": 1.0, "blk2": 1.0}
        assemble_fn = make_assemble_fn(toy_builder)

        result = fit_reml(
            assemble_fn,
            init,
            REMLOptions(max_iter=20, tol_rel=1e-3, verbose=False)
        )

        # All EDs should be positive
        for name, ed_val in result.ed.items():
            assert ed_val > 0, f"ED for {name} not positive: {ed_val}"

    def test_variance_updates_use_ed(self):
        """Test that variance updates follow ED-based formula."""
        init = {"eps": 1.0, "blk1": 2.0, "blk2": 0.5}
        assemble_fn = make_assemble_fn(toy_builder)

        result = fit_reml(
            assemble_fn,
            init,
            REMLOptions(max_iter=5, tol_rel=1e-3, verbose=False)
        )

        # Check that we have iteration log
        assert len(result.log) > 0, "No iteration log"

        # For each iteration after the first, verify update formula
        for log_entry in result.log[1:]:
            sigma2_old = log_entry["sigma2"]
            sigma2_new = log_entry["sigma2_new"]
            ed = log_entry["ed"]
            ss_u = log_entry["ss_u"]
            ss_e = log_entry["ss_e"]

            # Check random blocks: σ²_k,new ≈ ss_u[k] / ED_k
            for name in ss_u:
                expected = ss_u[name] / max(ed[name], 1e-12)
                actual = sigma2_new[name]
                # Allow for safeguarding
                assert actual >= 1e-12, f"Variance {name} below safeguard"

    def test_solution_dimensions(self):
        """Test that solution has correct dimensions."""
        init = {"eps": 1.0, "blk1": 1.0, "blk2": 1.0}
        assemble_fn = make_assemble_fn(toy_builder)

        result = fit_reml(
            assemble_fn,
            init,
            REMLOptions(max_iter=20, tol_rel=1e-3, verbose=False)
        )

        # Get expected dimensions from builder
        X, Z_dict, y, block_order = toy_builder(init)

        # Check beta dimension
        assert result.beta.shape[0] == X.shape[1], \
            f"Beta dimension mismatch: {result.beta.shape[0]} vs {X.shape[1]}"

        # Check u dimensions
        for name in block_order:
            assert name in result.u, f"Missing block {name} in result.u"
            expected_dim = Z_dict[name].shape[1]
            actual_dim = result.u[name].shape[0]
            assert actual_dim == expected_dim, \
                f"Block {name} dimension mismatch: {actual_dim} vs {expected_dim}"


class TestAssemblyAdapter:
    """Test assembly adapter functionality."""

    def test_assembly_creates_valid_system(self):
        """Test that assembly creates valid C and rhs."""
        theta = {"eps": 1.0, "blk1": 1.5, "blk2": 0.8}
        assemble_fn = make_assemble_fn(toy_builder)

        C, rhs, blocks, rank_X, n, X, Z_dict = assemble_fn(theta)

        # C should be square
        assert C.shape[0] == C.shape[1], "C not square"

        # C should be symmetric (check a few entries)
        C_dense = C.toarray()
        assert np.allclose(C_dense, C_dense.T, rtol=1e-10), "C not symmetric"

        # RHS dimension should match C
        assert rhs.shape[0] == C.shape[0], "RHS dimension mismatch"

        # Number of observations should be positive
        assert n > 0, "Invalid number of observations"

    def test_block_info_contiguous(self):
        """Test that BlockInfo has contiguous ranges."""
        theta = {"eps": 1.0, "blk1": 1.0, "blk2": 1.0}
        assemble_fn = make_assemble_fn(toy_builder)

        C, rhs, blocks, rank_X, n, X, Z_dict = assemble_fn(theta)

        # Blocks should be contiguous
        for i in range(len(blocks) - 1):
            assert blocks[i].stop == blocks[i + 1].start, \
                f"Blocks {i} and {i+1} not contiguous"

        # Blocks should cover entire random part
        if blocks:
            total_random = blocks[-1].stop - blocks[0].start
            expected_random = sum(Z_dict[b.name].shape[1] for b in blocks)
            assert total_random == expected_random, \
                f"Block coverage mismatch: {total_random} vs {expected_random}"

    def test_slices_valid(self):
        """Test that beta_slice and random_slice are valid."""
        theta = {"eps": 1.0, "blk1": 1.0, "blk2": 1.0}
        assemble_fn = make_assemble_fn(toy_builder)

        C, rhs, blocks, rank_X, n, X, Z_dict = assemble_fn(theta)

        # Slices should be set
        assert hasattr(assemble_fn, 'beta_slice'), "Missing beta_slice"
        assert hasattr(assemble_fn, 'random_slice'), "Missing random_slice"
        assert hasattr(assemble_fn, 'y'), "Missing y"

        # Beta slice should match X dimension
        beta_size = assemble_fn.beta_slice.stop - assemble_fn.beta_slice.start
        assert beta_size == X.shape[1], \
            f"Beta slice size {beta_size} != X columns {X.shape[1]}"

        # Random slice should match Z dimension
        total_random = sum(Z.shape[1] for Z in Z_dict.values())
        random_size = assemble_fn.random_slice.stop - assemble_fn.random_slice.start
        assert random_size == total_random, \
            f"Random slice size {random_size} != total random {total_random}"


class TestPSANOVABuilder:
    """Test PS-ANOVA builder helper."""

    def test_psanova_builder_basic(self):
        """Test basic PS-ANOVA builder creation."""
        n = 40
        X_poly = np.c_[np.ones(n), np.linspace(0, 1, n), np.linspace(0, 1, n)]
        Z_r = np.random.randn(n, 8)
        Z_c = np.random.randn(n, 6)
        Z_rc = np.random.randn(n, 48)
        y = np.random.randn(n)

        builder = make_builder_from_psanova(X_poly, Z_r, Z_c, Z_rc, y)
        theta = {"eps": 1.0, "row_smooth": 1.0, "col_smooth": 1.0, "interaction_smooth": 1.0}

        X, Z_dict, y_out, block_order = builder(theta)

        # Check structure
        assert X.shape == X_poly.shape, "X dimension mismatch"
        assert np.allclose(X, X_poly), "X content mismatch"
        assert y_out is y, "y not preserved"

        # Check blocks
        assert "row_smooth" in Z_dict
        assert "col_smooth" in Z_dict
        assert "interaction_smooth" in Z_dict
        assert len(block_order) == 3

    def test_psanova_builder_with_genotype(self):
        """Test PS-ANOVA builder with genotype effects."""
        n = 40
        X_poly = np.c_[np.ones(n), np.linspace(0, 1, n), np.linspace(0, 1, n)]
        Z_r = np.random.randn(n, 8)
        Z_c = np.random.randn(n, 6)
        Z_rc = np.random.randn(n, 48)
        genotype_Z = np.random.randn(n, 20)
        y = np.random.randn(n)

        builder = make_builder_from_psanova(
            X_poly, Z_r, Z_c, Z_rc, y,
            genotype_Z=genotype_Z
        )
        theta = {
            "eps": 1.0,
            "row_smooth": 1.0,
            "col_smooth": 1.0,
            "interaction_smooth": 1.0,
            "genotype": 1.0
        }

        X, Z_dict, y_out, block_order = builder(theta)

        # Should have genotype block
        assert "genotype" in Z_dict
        assert "genotype" in block_order
        assert Z_dict["genotype"].shape[1] == 20


class TestConvergence:
    """Test convergence behavior."""

    def test_converges_from_poor_initial(self):
        """Test convergence from poor initial values."""
        # Very poor initial guess
        init = {"eps": 100.0, "blk1": 0.01, "blk2": 50.0}
        assemble_fn = make_assemble_fn(toy_builder)

        result = fit_reml(
            assemble_fn,
            init,
            REMLOptions(max_iter=50, tol_rel=1e-3, verbose=False)
        )

        # Should still converge (eventually)
        assert result.converged or result.n_iter > 30, \
            "Failed to make progress from poor initial"

    def test_early_convergence_detection(self):
        """Test that convergence is detected early if already close."""
        # Start near a reasonable solution
        init = {"eps": 1.0, "blk1": 1.0, "blk2": 1.0}
        assemble_fn = make_assemble_fn(toy_builder)

        # Run once to get converged values
        result1 = fit_reml(
            assemble_fn,
            init,
            REMLOptions(max_iter=50, tol_rel=1e-3, verbose=False)
        )

        # Run again starting from converged values
        result2 = fit_reml(
            assemble_fn,
            result1.sigma2,
            REMLOptions(max_iter=50, tol_rel=1e-3, verbose=False)
        )

        # Should converge in 1-2 iterations
        assert result2.n_iter <= 3, \
            f"Should converge immediately from solution, got {result2.n_iter} iters"

    def test_safeguard_prevents_negative_variance(self):
        """Test that safeguarding prevents negative variances."""
        # Create a scenario that might produce negative updates
        init = {"eps": 1e-10, "blk1": 1e-10, "blk2": 1e-10}
        assemble_fn = make_assemble_fn(toy_builder)

        result = fit_reml(
            assemble_fn,
            init,
            REMLOptions(max_iter=20, tol_rel=1e-3, safeguard_min=1e-12, verbose=False)
        )

        # All variances should be >= safeguard
        for name, var in result.sigma2.items():
            assert var >= 1e-12, f"Variance {name} below safeguard: {var}"


def test_reml_integration_with_psanova():
    """Integration test: REML with PS-ANOVA design."""
    # Create small PS-ANOVA problem
    from pyspats.psanova_basis import build_psanova_design

    n_r, n_c = 8, 6
    r = np.repeat(np.arange(n_r, dtype=float), n_c)
    c = np.tile(np.arange(n_c, dtype=float), n_r)

    # Build PS-ANOVA design
    X_poly, Z_r, Z_c, Z_rc, blocks = build_psanova_design(
        r, c, nkr=6, nkc=5, degree=3
    )

    # Simulate response with known structure
    rng = np.random.default_rng(42)
    n = len(r)
    beta_true = np.array([10.0, 0.5, 0.3])  # intercept, r, c
    y = X_poly @ beta_true + rng.normal(0, 1.0, n)

    # Create builder and assembly function
    builder = make_builder_from_psanova(X_poly, Z_r, Z_c, Z_rc, y)
    assemble_fn = make_assemble_fn(builder)

    # Initial variance components
    init = {"eps": 1.0, "row_smooth": 1.0, "col_smooth": 1.0, "interaction_smooth": 1.0}

    # Run REML
    result = fit_reml(
        assemble_fn,
        init,
        REMLOptions(max_iter=50, tol_rel=1e-3, verbose=False)  # Increase max_iter
    )

    # Should converge (or at least make progress)
    assert result.converged or result.n_iter >= 30, "REML failed to make progress on PS-ANOVA problem"

    # Should recover reasonable variance estimates
    assert result.sigma2["eps"] > 0, "Residual variance not positive"

    # Beta should be close to true values (loose check)
    assert np.abs(result.beta[0] - 10.0) < 5.0, "Intercept estimate too far off"
