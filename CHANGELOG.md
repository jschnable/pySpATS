# Changelog

All notable changes to pySpATS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-10-02

### Added

- **Memory-efficient Kronecker interaction smooth** (`pyspats/spatial/`)
  - Lazy LinearOperator implementation avoids materializing full `(n × p_r·p_c)` interaction matrix
  - Memory footprint: O(n·p_r + n·p_c) instead of O(n·p_r·p_c) for large field trials
  - Enabled by default via `use_kron_interaction=True`

- **Gram-matrix-based Z'Z and X'Z computation** (`spatial/gram_kron.py`)
  - Eliminates O(n) observation loops using small dense Gram matrices
  - Computes Z'Z via Kronecker algebra: O(p_r² · p_c²) instead of O(n · p_rc)
  - Numerical safeguards: symmetrization, re-orthogonalization, minimal jitter for PSD

- **Polynomial projection for PS-ANOVA orthogonality** (`spatial/projection.py`)
  - `LeftProjectedLO` applies P_⊥ = I - Q_x Q_x^T ensuring X_poly^T @ Z_rc ≈ 0
  - Works on both matvec and rmatvec without materializing full matrix

- **Exact mode selection** for interaction nullspace
  - Uses Kronecker sum criterion: λ_rc[i,j] = λ_r[i] + λ_c[j] > tol
  - Relative tolerance: tol_effective = tol * max(λ_r, λ_c) for scale robustness
  - Matches dense path dimensions exactly

- **REML optimizer with CHOLMOD-based exact EDs** (`pyspats/reml/`)
  - Factorization reuse: one CHOLMOD factorization per iteration
  - Exact effective dimensions via Takahashi selected inverse
  - Closed-form variance updates: σ²_k = (u_k' u_k) / ED_k
  - Schur complement sparse/dense split for efficiency

- **Generalized heritability (default)**
  - H² = ED_geno / n_geno (SpATS-style, matches R package)
  - Classical H² = ED_geno / (n_geno - 1) available via `get_heritability(mode='classical')`

- **Enhanced diagnostics**
  - `summary_ed()`: Effective dimension summary with percentage breakdown
  - Shows each component's contribution to total degrees of freedom
  - Residual ED percentage for model parsimony assessment

- **Comprehensive test suite**
  - 176 tests passing (13 Kronecker, 5 projection, 5 mode selection, 7 sign-off)
  - Tests for irregular layouts, missing plots, extreme smoothing, permutation invariance
  - Validates model space equivalence between Kronecker and dense paths

### Changed

- **PS-ANOVA decomposition** now enforces strict hygiene:
  - Random smooths orthogonal to polynomial space (X_poly^T @ Z_k ≈ 0)
  - Nullspace-free: constant/linear trends removed from random effects
  - Identity covariance: G_k = σ_k² I after whitening

- **Block metadata structure**: All random effects now have contiguous `BlockInfo` for exact ED computation

### Fixed

- Heritability calculation now uses exact effective dimensions from CHOLMOD when available
- Improved numerical stability in polynomial projection with re-orthogonalization
- Relative tolerance for nullspace detection avoids scale-dependent issues

### Documentation

- Updated CLAUDE.md with Kronecker implementation architecture
- Updated README.md with heritability documentation and usage examples
- Added comprehensive API documentation for spatial subpackage

## [0.1.0] - 2025-01-15

### Added

- Initial Python implementation of SpATS
- PS-ANOVA spatial decomposition with 2D P-splines
- BLUEs extraction for genotypes
- Basic heritability estimation
- Diagnostic plotting functions
- Validation against R SpATS package

---

**Note**: Version 0.2.0 represents a major enhancement focused on computational efficiency, numerical accuracy, and production readiness while maintaining full backward compatibility.
