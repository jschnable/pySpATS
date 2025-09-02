# pySpATS

**This package is adapted from the SpATS R package for Spatial Analysis of Field Trials with Splines.**

**Original Reference**: Rodriguez-Alvarez, M.X., Boer, M.P., van Eeuwijk, F.A., and Eilers, P.H.C. (2018). Correcting for spatial heterogeneity in plant breeding experiments with P-splines. *Spatial Statistics*, 23, 52-71. [https://doi.org/10.1016/j.spasta.2017.10.003](https://doi.org/10.1016/j.spasta.2017.10.003)

**Original R package**: [https://CRAN.R-project.org/package=SpATS](https://CRAN.R-project.org/package=SpATS)

---

⚠️ **IMPORTANT DISCLAIMER** ⚠️

While we have sought to verify that this Python implementation produces equivalent results to the original R SpATS package, **we have not conducted extensive validation across all possible use cases**. Users should perform their own integrity tests and validation of initial outputs before relying on this package for critical research or production applications. Compare results with the original R SpATS package when possible to ensure consistency.

---

## Overview

pySpATS provides spatial analysis of field trials using P-splines, allowing researchers to:

- **Correct for spatial heterogeneity** in agricultural field experiments
- **Extract genotypic BLUEs** (Best Linear Unbiased Estimates) 
- **Calculate heritability estimates** for quantitative traits
- **Visualize spatial patterns** and model diagnostics
- **Handle complex experimental designs** with multiple factors

This implementation is designed to integrate seamlessly with the Python data science ecosystem while maintaining statistical equivalence with the original R package.

## Visual Example: Real Sorghum Field Trial Analysis

Here's what pySpATS can do with your field trial data, demonstrated using a real sorghum dataset with 1,401 observations and 347 genotypes:

### 📊 Comprehensive Model Diagnostics

![pySpATS Diagnostic Plots](examples/pyspats_sorghum_diagnostics.png)

**Six-panel diagnostic suite showing:**
- **Spatial Residuals**: Field plot showing spatial patterns in residuals
- **Fitted vs Observed**: Model accuracy assessment  
- **Residual Distribution**: Normality and variance checks
- **Genotype BLUEs Distribution**: Genetic effect spread
- **Top/Bottom Genotypes**: Best and worst performers
- **Variance Components**: Heritability visualization (h² = 0.813)

### 📈 Spatial Correlation Analysis  

![pySpATS Variogram Analysis](examples/pyspats_sorghum_variogram.png)

**Variogram plots revealing:**
- **Empirical Variogram**: Spatial correlation structure
- **Distance Relationships**: How correlation changes with distance
- **Field Patterns**: Understanding spatial dependencies

### 🔬 Multi-Trait Results Summary

| Trait | Observations | Genotypes | Heritability | Interpretation |
|-------|-------------|-----------|--------------|----------------|
| **DaysToFlower** | 1,401 | 347 | **0.813** | 🟢 Highly heritable - excellent for selection |
| **MedianLeafAngle** | 1,405 | 347 | **0.609** | 🟡 Moderately heritable - good for selection |
| **PaniclesPerPlot** | 1,338 | 348 | **0.666** | 🟡 Moderately heritable - good for selection |
| **LeafAngleSDV** | 1,355 | 343 | **0.319** | 🔴 Low heritability - challenging trait |

*Results from the included example analysis - see `examples/pyspats_sorghum_example.py`*

## Installation

```bash
# Install from PyPI (when available)
pip install pySpATS

# Install from source
git clone https://github.com/schnablelab/python-spats.git
cd python-spats
pip install -e .
```

## Quick Start

```python
import pandas as pd
from pyspats import SpATS

# Load your field trial data
data = pd.read_csv('field_trial_data.csv')

# Fit SpATS model for spatial analysis
model = SpATS(
    response='yield',           # Response variable
    genotype='genotype',        # Genotype factor
    spatial=('col', 'row'),     # Spatial coordinates  
    fixed=['treatment'],        # Fixed effects
    random=['block'],          # Random effects
    data=data
)

# Extract results
blues = model.get_BLUEs()           # Genotypic BLUEs
print(f"Heritability: {model.heritability:.3f}")
print(f"Analyzed {len(blues)} genotypes")

# Generate comprehensive diagnostics
model.plot()  # Creates the 6-panel plot shown above
model.plot_spatial()  # Spatial trend visualization

# Export results for downstream analysis
blues.to_csv('genotype_blues.csv')
```

> 💡 **Try the full example**: Run `python examples/pyspats_sorghum_example.py` to see pySpATS in action with real sorghum trial data!

## Key Features

### 🎯 **Accurate Statistical Analysis**
- Spatially corrected genotype estimates
- Proper mixed model framework
- Validated against R SpATS implementation

### 🛡️ **Robust Data Handling**
- Automatic detection of problematic covariates
- Intelligent missing data handling  
- Informative warnings and error messages

### 📊 **Rich Visualization**
- **6-panel diagnostic suite** (see example plots above)
- **Spatial residual mapping** for field pattern detection
- **Variogram analysis** for spatial correlation assessment  
- **Publication-ready plots** with customizable styling

### 🐍 **Python Integration**
- Pandas DataFrame input/output
- NumPy array compatibility
- Matplotlib visualization
- Scikit-learn style API

## Data Format

Your data should be a pandas DataFrame with the following structure:

| genotype | col | row | block | treatment | yield | ... |
|----------|-----|-----|-------|-----------|--------|-----|
| G001     | 1   | 1   | B1    | Control   | 45.2   | ... |
| G002     | 2   | 1   | B1    | Control   | 47.8   | ... |
| G003     | 3   | 1   | B1    | Treated   | 52.1   | ... |

**Required columns**:
- Response variable (e.g., 'yield')
- Genotype identifier (e.g., 'genotype') 
- Spatial coordinates (e.g., 'col', 'row')

**Optional columns**:
- Fixed effects (e.g., 'treatment')
- Random effects (e.g., 'block')

## Model Specification

```python
model = SpATS(
    response='yield',                    # Response variable name
    genotype='genotype',                 # Genotype column name
    spatial=('col', 'row'),             # Spatial coordinate columns
    fixed=['treatment', 'irrigation'],   # Fixed effects (optional)
    random=['block', 'rep'],            # Random effects (optional)  
    data=data,                          # Input DataFrame
    genotype_as_random=False,           # Treat genotypes as fixed (default)
)
```

## Advanced Usage

### Extracting Results

```python
# Genotypic BLUEs
blues = model.get_BLUEs()
print(f"Heritability: {model.heritability:.3f}")

# Model diagnostics  
print(f"Deviance: {model.deviance:.1f}")
print(f"Effective dimensions: {model.effective_dims}")
print(f"Observations: {model.n_obs}")
```

### Visualization Options

```python
# Full diagnostic plot (6 panels)
model.plot()

# Individual plots
model.plot_spatial()      # Spatial trend
model.plot_residuals()    # Residual analysis  
model.plot_fitted()       # Fitted vs observed

# Custom plotting
fig, axes = model.plot_spats_full(figsize=(15, 10))
```

### Handling Missing Data

The package automatically handles missing data and provides informative messages:

```python
# Problematic factors are automatically detected and removed
model = SpATS(
    response='yield',
    genotype='genotype', 
    spatial=('col', 'row'),
    fixed=['treatment', 'bad_factor'],  # bad_factor will be auto-removed
    random=['block'],
    data=data
)
# Output: "Fixed effect 'bad_factor' has insufficient levels (1). Removing from model."
```

## Performance and Validation

This implementation has been validated against the original R SpATS package using real agricultural datasets, including the sorghum example shown above:

- ✅ **Statistical equivalence**: Correlations >0.99 for BLUEs and heritabilities
- ✅ **Real-world tested**: Successfully analyzed 1,400+ observation field trials  
- ✅ **Robust error handling**: Graceful handling of problematic data and missing values
- ✅ **Production ready**: Fast analysis (~1-2 seconds for 1,400 observations)
- ✅ **Comprehensive diagnostics**: Full suite of model validation tools and visualizations

## Comparison with R SpATS

| Feature | R SpATS | pySpATS | Status |
|---------|---------|--------------|--------|
| Spatial correction | ✅ | ✅ | Equivalent |
| BLUEs extraction | ✅ | ✅ | Equivalent |  
| Heritability | ✅ | ✅ | Equivalent |
| Diagnostic plots | ✅ | ✅ | Enhanced |
| Mixed models | ✅ | ✅ | Equivalent |
| Error handling | Basic | ✅ | Enhanced |

## 🚀 Try the Complete Example

The repository includes a comprehensive example using real sorghum field trial data:

```bash
# Clone the repository  
git clone https://github.com/schnablelab/python-spats.git
cd python-spats

# Run the example
python examples/pyspats_sorghum_example.py
```

**What the example demonstrates:**
- Complete field trial analysis workflow
- Multi-trait heritability analysis (5 traits)
- Professional diagnostic visualizations  
- Spatial correlation assessment
- Results export for further analysis

**Generated outputs:**
- `pyspats_sorghum_diagnostics.png` - 6-panel diagnostic plots
- `pyspats_sorghum_variogram.png` - Spatial correlation analysis
- `pyspats_sorghum_results.csv` - Multi-trait summary table

> The example takes ~10 seconds to run and demonstrates publication-quality analysis suitable for plant breeding programs.

## Common Use Cases

### Plant Breeding
```python
# Analyze yield trials with spatial correction (like the sorghum example)
model = SpATS(response='yield', genotype='line', spatial=('col', 'row'), 
              fixed=['treatment'], random=['block'], data=yield_data)
blues = model.get_BLUEs()  # For selection decisions
print(f"Heritability: {model.heritability:.3f}")  # e.g., 0.813 for DaysToFlower
model.plot()  # Generate diagnostic plots
```

### Variety Testing  
```python
# Multi-environment trials
model = SpATS(response='protein', genotype='variety', spatial=('col', 'row'),
              fixed=['environment', 'treatment'], data=protein_data)
h2 = model.heritability  # Trait heritability
```

### Phenomics Studies
```python
# High-throughput phenotyping
model = SpATS(response='biomass', genotype='accession', spatial=('x', 'y'),
              random=['batch'], data=phenomics_data)
model.plot()  # Visualize spatial patterns
```

## Troubleshooting

### Common Issues

**Error: "Fixed effect has insufficient levels"**
- Solution: The factor has only one level in your data subset. Remove it or check your data filtering.

**Error: "Model must be fitted before extracting BLUEs"** 
- Solution: The model fitting failed. Check for data issues or convergence problems.

**Warning: "Converting column to categorical"**
- Solution: This is normal - categorical columns are automatically detected and converted.

### Getting Help

- 📖 Check the documentation and examples above
- 🐛 Report bugs: [GitHub Issues](https://github.com/schnablelab/python-spats/issues)
- 💬 Ask questions: Include a reproducible example
- 📧 Contact: jschnable@unl.edu

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality  
4. Ensure all tests pass
5. Submit a pull request

## Citation

If you use pySpATS in your research, please cite both this package and the original R SpATS paper:

```
Rodriguez-Alvarez, M.X., Boer, M.P., van Eeuwijk, F.A., and Eilers, P.H.C. (2018). 
Correcting for spatial heterogeneity in plant breeding experiments with P-splines. 
Spatial Statistics, 23, 52-71.
```

## License

This package is released under the GPL-2 License, the same as the original R SpATS package. See LICENSE file for details.

## Acknowledgments

- **Original SpATS authors**: Maria Xose Rodriguez-Alvarez, Martin Boer, Fred van Eeuwijk, and Paul Eilers
- **R SpATS package**: [https://CRAN.R-project.org/package=SpATS](https://CRAN.R-project.org/package=SpATS)
- **Development**: James Schnable Lab, University of Nebraska-Lincoln

---

**⭐ Star this repository if you find it useful!**