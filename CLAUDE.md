# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

pySpATS is a Python implementation of the R SpATS package for Spatial Analysis of Field Trials with Splines. It provides spatial analysis of agricultural field trials using two-dimensional Penalized splines (P-splines) to correct for spatial heterogeneity and extract genotypic BLUEs (Best Linear Unbiased Estimates).

**Original Reference**: Rodriguez-Alvarez et al. (2018). "Correcting for spatial heterogeneity in plant breeding experiments with P-splines." Spatial Statistics, 23, 52-71.

**Key Validation Note**: This package seeks to replicate R SpATS results, but has not been extensively validated across all use cases. Users should validate results against the original R package.

## Development Commands

### Installation
```bash
# Install in development mode
pip install -e .

# Install with dependencies
pip install -r requirements.txt
```

### Testing
```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_core.py

# Run specific test
python -m pytest tests/test_core.py::TestSpATS::test_basic_fitting

# Run with verbose output
python -m pytest -v

# Run validation scripts
python tests/simple_validation.py
python tests/comprehensive_validation.py
```

### Examples
```bash
# Run comprehensive sorghum example (generates diagnostic plots)
python examples/pyspats_sorghum_example.py
```

## Architecture

### Core Model Flow

The SpATS model fitting follows this sequence:

1. **Input Validation** (`core.py:_validate_inputs`): Checks for missing columns, validates spatial coordinates, ensures categorical variables are properly typed
2. **Data Preparation** (`core.py:_prepare_data`): Handles missing values, converts categorical variables, removes problematic factors with insufficient levels
3. **Design Matrix Construction** (`basis.py:construct_design_matrix`): Builds fixed effects (X) and random effects (Z) design matrices
4. **Spatial Basis Construction** (`basis.py:construct_2d_pspline`): Creates 2D P-spline basis using tensor products of 1D B-splines
5. **Model Fitting** (`solver.py:SAP_solver`): Uses SAP (Separation of Anisotropic Penalties) algorithm to estimate coefficients and variance components
6. **BLUE Extraction** (`core.py:get_BLUEs`): Computes genotype Best Linear Unbiased Estimates
7. **Heritability Calculation** (`core.py:heritability`): Estimates broad-sense heritability from variance components

### Key Components

**`pyspats/core.py`**: Main SpATS class
- Orchestrates the entire model fitting process
- Handles fixed/random effects specification
- Computes fitted values, residuals, deviance
- Provides methods for extracting BLUEs and heritability
- Integration point for plotting functions

**`pyspats/basis.py`**: B-spline and design matrix construction
- `bspline_basis()`: Constructs 1D B-spline basis functions using scipy.interpolate.BSpline
- `construct_2d_pspline()`: Creates 2D tensor product basis for spatial surface
- `construct_design_matrix()`: Builds design matrices for fixed and random effects
- Penalty matrix construction for P-spline smoothing

**`pyspats/solver.py`**: SAP algorithm implementation
- `SAP_solver.solve()`: Iteratively estimates coefficients and variance components
- Efficient sparse matrix operations using scipy.sparse
- Handles separation of anisotropic penalties for spatial effects
- Convergence based on relative change in variance components

**`pyspats/families.py`**: Distribution families and link functions
- `gaussian()`: Normal distribution with identity link (default)
- `poisson()`: Poisson distribution with log link
- `binomial()`: Binomial distribution with logit link
- Follows GLM framework similar to R's family objects

**`pyspats/plotting.py`**: Visualization functions
- `plot_spats()`: 6-panel comprehensive diagnostic plot
- `plot_spatial()`: Spatial trend visualization
- `plot_residuals()`: Residual diagnostic plots
- `plot_variogram()`: Spatial correlation analysis

**`pyspats/variogram.py`**: Spatial correlation analysis
- Empirical variogram calculation
- Distance-based correlation assessment
- Integration with plotting for spatial diagnostics

**`pyspats/control.py`**: Algorithm control parameters
- SpATSControl class manages convergence criteria, iteration limits, smoothing parameters

**`pyspats/utils.py`**: Utility functions
- `SAP()`: Interface to SAP algorithm
- `PSANOVA()`: P-spline ANOVA functionality
- `interpret_formula()`: Formula parsing for model specification
- `get_heritability()`: Heritability calculation from effective dimensions (generalized and classical modes)

### Data Flow

```
Input DataFrame
    ↓
SpATS.__init__() validates inputs
    ↓
_prepare_data() cleans and structures data
    ↓
construct_design_matrix() creates X (fixed) and Z (random) matrices
    ↓
construct_2d_pspline() creates spatial basis
    ↓
SAP_solver.solve() iteratively estimates parameters
    ↓
Fitted model with coefficients, variance components
    ↓
get_BLUEs() extracts genotype estimates
    ↓
plot() generates diagnostics
```

## Important Implementation Details

### Genotype Handling
- Genotypes can be treated as **fixed effects** (default, `genotype_as_random=False`) or **random effects** (`genotype_as_random=True`)
- Fixed genotype treatment is typical for extracting BLUEs for genomic selection
- The genotype factor must be categorical (automatically converted if needed)

### Spatial Coordinates
- Specified as tuple: `spatial=('col', 'row')` for column and row coordinates
- These create the 2D spatial surface using tensor product P-splines
- Default number of segments (`nseg`) controls smoothness of spatial trend

### Factor Level Validation
- The package automatically detects and removes fixed/random effects with insufficient levels (< 2 levels)
- This prevents singular design matrices and provides informative warnings
- Located in `core.py:_prepare_data()`

### Missing Data
- Rows with missing values in response, spatial coordinates, or required factors are automatically removed
- Warnings are issued indicating how many rows were removed

### Variance Components
- Stored in `model.var_comp` dictionary with keys for each random effect
- Spatial variance is under key 'f(col):f(row)'
- Residual variance under key 'residual'

### Heritability Calculation
- **Default (generalized)**: H² = ED_geno / n_geno (SpATS-style, matches R SpATS)
- **Classical**: H² = ED_geno / (n_geno - 1) (legacy, available via `mode='classical'`)
- Only available when genotype is treated as **fixed effect**
- Accessed via `model.heritability` property (generalized) or `model.get_heritability(mode='classical')` for classical
- Based on effective dimension (ED_geno) which represents the complexity/smoothness of the genotype effect

## Testing Strategy

The test suite includes:

1. **Unit tests** (`tests/test_*.py`): Test individual components (basis construction, design matrices, core functionality)
2. **Validation scripts** (`tests/*_validation.py`): Compare pySpATS results against R SpATS outputs
3. **Plot validation** (`tests/plot_validation.py`): Verify visualization functions work correctly
4. **Example data** (`pyspats/datasets.py`): Provides `create_toy_example()` and `generate_field_trial_data()` for testing

## Common Workflows

### Adding a New Feature
1. Implement core logic in appropriate module (`core.py`, `basis.py`, `solver.py`, etc.)
2. Add unit tests in corresponding `tests/test_*.py` file
3. Update `__init__.py` if exposing new public API
4. Add example usage to docstrings
5. Consider validation against R SpATS if applicable

### Debugging Model Fitting Issues
1. Check input data validation warnings in `_validate_inputs()` and `_prepare_data()`
2. Examine design matrices: `X.shape`, `Z.shape` should match expected dimensions
3. Review variance component estimates: Large values may indicate convergence issues
4. Use diagnostic plots: `model.plot()` to assess residual patterns
5. Compare with R SpATS if results seem unexpected

### Adding New Plotting Functions
- Add to `pyspats/plotting.py`
- Follow existing pattern: accept model object or data, return figure/axes
- Use matplotlib for consistency
- Add to SpATS class as method for convenience (delegates to plotting module)

## Dependencies

Core scientific stack:
- **numpy**: Array operations, linear algebra
- **pandas**: Data manipulation, DataFrame I/O
- **scipy**: Sparse matrices, B-splines, linear solvers, statistical distributions
- **matplotlib**: Plotting and visualization
- **scikit-learn**: Used for some utility functions

## Package Structure

```
pyspats/
  __init__.py          # Public API exports
  core.py              # Main SpATS class and fitting logic
  basis.py             # B-spline basis and design matrix construction
  solver.py            # SAP algorithm implementation
  families.py          # Distribution families (gaussian, poisson, binomial)
  control.py           # SpATSControl parameter class
  plotting.py          # Visualization functions
  variogram.py         # Variogram analysis
  utils.py             # Helper functions (SAP, PSANOVA, formula parsing)
  datasets.py          # Example data generators

tests/
  test_*.py            # Unit tests (pytest)
  *_validation.py      # Validation against R SpATS

examples/
  pyspats_sorghum_example.py  # Comprehensive real-world example
  sorghum_data.csv            # Example dataset
```

## Notes for AI Assistance

- This package prioritizes **statistical correctness** over performance - validate changes against R SpATS when modifying core algorithms
- The SAP algorithm in `solver.py` is mathematically complex - consult Rodriguez-Alvarez et al. (2015) paper before modifying
- Sparse matrix operations are critical for performance with large field trials - maintain sparsity when possible
- Error messages should be informative for agricultural researchers who may not be Python experts
- Plotting functions should produce publication-quality figures suitable for scientific papers
