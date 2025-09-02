# pySpATS Examples

This directory contains example scripts demonstrating the capabilities of the pySpATS package.

## Files

### Data Files
- `sorghum_data.csv` - Real sorghum field trial dataset with multiple traits
- `r_spats_summary_clean.csv` - R SpATS analysis results for comparison

### Example Scripts
- `pyspats_sorghum_example.py` - Comprehensive demonstration of pySpATS capabilities

## Running the Example

### Prerequisites
```bash
# Ensure required packages are installed
pip install numpy pandas matplotlib seaborn scipy scikit-learn
```

### Run the Example
```bash
cd examples
python3 pyspats_sorghum_example.py
```

## What the Example Demonstrates

The `pyspats_sorghum_example.py` script showcases:

1. **Data Loading and Exploration**
   - Loading field trial data
   - Examining data structure and quality
   - Identifying available traits

2. **SpATS Model Fitting**
   - Spatial analysis with 2D P-splines
   - Fixed effects (treatment) and random effects (block) 
   - Model convergence and diagnostics

3. **Genotype Analysis**
   - Extraction of genotype BLUEs (Best Linear Unbiased Estimates)
   - Heritability calculation
   - Variance component analysis

4. **Diagnostic Plotting**
   - Spatial residual patterns
   - Fitted vs observed values
   - Residual distributions
   - Genotype effect distributions
   - Variance component visualization

5. **Variogram Analysis**
   - Empirical variogram calculation
   - Spatial correlation assessment
   - Distance-based analysis

6. **Multi-trait Analysis**
   - Batch processing of multiple traits
   - Comparative heritability estimates
   - Summary statistics export

## Expected Outputs

The script generates the following files:

- `pyspats_sorghum_diagnostics.png` - Comprehensive diagnostic plots
- `pyspats_sorghum_variogram.png` - Variogram analysis plots  
- `pyspats_sorghum_results.csv` - Multi-trait analysis summary

## Sample Results

From the sorghum dataset analysis:

| Trait | Observations | Genotypes | Heritability | Mean |
|-------|-------------|-----------|--------------|------|
| DaysToFlower | 1401 | 347 | 0.813 | 65.5 |
| MedianLeafAngle | 1405 | 347 | 0.609 | 44.7 |
| PaniclesPerPlot | 1338 | 348 | 0.666 | 14.9 |
| LeafAngleSDV | 1355 | 343 | 0.319 | 6.1 |

## Interpretation Guide

### Heritability Values
- **High (h² > 0.7)**: Strong genetic control, good for selection
- **Moderate (0.4 < h² < 0.7)**: Moderate genetic control
- **Low (h² < 0.4)**: Environmental effects dominate

### Using Results for Plant Breeding
- **BLUEs**: Use for genomic prediction and selection
- **Spatial Effects**: Consider in trial design
- **Heritability**: Prioritize traits for selection programs

## Customization

To analyze your own data:

1. Replace `sorghum_data.csv` with your dataset
2. Ensure columns include:
   - Genotype identifiers
   - Spatial coordinates (row, column)
   - Treatment/block factors
   - Response traits
3. Modify column names in the script as needed
4. Adjust analysis parameters for your specific needs

## Performance Notes

- Analysis time scales with dataset size
- Expect ~1-2 seconds per trait for datasets of 1000-2000 observations
- Memory usage is generally modest for typical field trial sizes
- Diagnostic plots help assess model adequacy

## Support

For questions about:
- **pySpATS package**: Check the main README and documentation
- **Spatial analysis theory**: See original SpATS R package documentation
- **Field trial analysis**: Consult plant breeding and statistics resources