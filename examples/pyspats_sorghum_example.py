#!/usr/bin/env python3
"""
pySpATS Example: Spatial Analysis of Sorghum Field Trial Data

This script demonstrates the key capabilities of the pySpATS package using
real sorghum field trial data. It shows how to:

1. Load and prepare field trial data
2. Fit SpATS models for spatial analysis  
3. Extract genotype BLUEs (Best Linear Unbiased Estimates)
4. Calculate heritability estimates
5. Generate diagnostic plots
6. Perform variogram analysis
7. Compare multiple traits

Author: pySpATS Package
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
import os

# Add parent directory to path to find pyspats package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import pySpATS components
from pyspats import SpATS, SpATSControl, plot_spats, plot_variogram, variogram

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore', category=FutureWarning)

def main():
    """Main example demonstrating pySpATS capabilities."""
    
    print("=" * 80)
    print("pySpATS Example: Spatial Analysis of Sorghum Field Trial Data")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # 1. Load and Explore Data
    # -------------------------------------------------------------------------
    print("\n1. Loading sorghum field trial data...")
    
    # Load the sorghum dataset
    data_path = 'sorghum_data.csv'
    if not os.path.exists(data_path):
        data_path = 'examples/sorghum_data.csv'  # Try from parent directory
    
    data = pd.read_csv(data_path)
    print(f"   - Loaded {len(data)} observations")
    print(f"   - Field dimensions: {data['Column'].nunique()} columns × {data['Row'].nunique()} rows")
    print(f"   - Number of genotypes: {data['PINumber'].nunique()}")
    print(f"   - Number of treatments: {data['Treatment'].nunique()}")
    print(f"   - Number of blocks: {data['Block'].nunique()}")
    
    # Show available traits
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    trait_cols = [col for col in numeric_cols if col not in ['Column', 'Row', 'Block']]
    print(f"   - Available traits: {', '.join(trait_cols[:5])}..." if len(trait_cols) > 5 else f"   - Available traits: {', '.join(trait_cols)}")
    
    # -------------------------------------------------------------------------
    # 2. Select and Prepare Data for Analysis
    # -------------------------------------------------------------------------
    print("\n2. Preparing data for spatial analysis...")
    
    # Select a trait for demonstration (DaysToFlower)
    target_trait = 'DaysToFlower'
    if target_trait not in data.columns:
        # If DaysToFlower not available, use the first numeric trait
        target_trait = trait_cols[0]
    
    print(f"   - Analyzing trait: {target_trait}")
    
    # Prepare data for SpATS analysis
    trait_data = data.dropna(subset=[target_trait, 'PINumber']).copy()
    
    # Rename columns to standard names
    analysis_data = trait_data.rename(columns={
        'PINumber': 'genotype',
        'Column': 'col', 
        'Row': 'row',
        'Treatment': 'treatment',
        'Block': 'block',
        target_trait: 'response'
    })
    
    print(f"   - Clean dataset: {len(analysis_data)} observations")
    print(f"   - Genotypes in analysis: {analysis_data['genotype'].nunique()}")
    print(f"   - Response range: {analysis_data['response'].min():.1f} - {analysis_data['response'].max():.1f}")
    
    # -------------------------------------------------------------------------
    # 3. Fit SpATS Model
    # -------------------------------------------------------------------------
    print("\n3. Fitting SpATS model...")
    
    # Create SpATS model with spatial correction
    model = SpATS(
        response='response',
        genotype='genotype', 
        spatial=('col', 'row'),
        fixed=['treatment'],
        random=['block'],
        data=analysis_data
    )
    
    print(f"   ✅ Model fitted successfully!")
    print(f"   - Observations: {model.n_obs}")
    print(f"   - Converged: {hasattr(model, 'coefficients')}")
    print(f"   - Residual variance: {model.psi:.4f}")
    print(f"   - Model deviance: {model.deviance:.2f}")
    
    # -------------------------------------------------------------------------
    # 4. Extract Genotype BLUEs and Calculate Heritability
    # -------------------------------------------------------------------------
    print("\n4. Extracting genotype effects and calculating heritability...")
    
    # Get genotype BLUEs
    blues = model.get_BLUEs()
    print(f"   - Extracted BLUEs for {len(blues)} genotypes")
    print(f"   - BLUEs range: {blues.min():.2f} - {blues.max():.2f}")
    
    # Calculate heritability
    var_g = np.var(blues.values, ddof=1)  # Genotypic variance
    var_e = model.psi  # Error variance
    heritability = var_g / (var_g + var_e)
    
    print(f"   - Genotypic variance: {var_g:.4f}")
    print(f"   - Error variance: {var_e:.4f}")
    print(f"   - Heritability (h²): {heritability:.3f}")
    
    # -------------------------------------------------------------------------
    # 5. Generate Model Diagnostics
    # -------------------------------------------------------------------------
    print("\n5. Generating diagnostic plots...")
    
    # Create comprehensive diagnostic plots
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Spatial residuals
    plt.subplot(2, 3, 1)
    max_row = analysis_data['row'].max()
    min_row = analysis_data['row'].min()
    max_col = analysis_data['col'].max()
    min_col = analysis_data['col'].min()
    
    residuals_2d = np.full((int(max_row - min_row + 1), int(max_col - min_col + 1)), np.nan)
    for i, (idx, row) in enumerate(analysis_data.iterrows()):
        r_idx = int(row['row'] - min_row)
        c_idx = int(row['col'] - min_col) 
        if r_idx < residuals_2d.shape[0] and c_idx < residuals_2d.shape[1]:
            residuals_2d[r_idx, c_idx] = model.residuals[i]
    
    plt.imshow(residuals_2d, cmap='RdBu_r', aspect='auto')
    plt.colorbar(label='Residuals')
    plt.title('Spatial Pattern of Residuals')
    plt.xlabel('Column')
    plt.ylabel('Row')
    
    # Plot 2: Fitted vs Observed
    plt.subplot(2, 3, 2)
    plt.scatter(model.fitted_values, analysis_data['response'], alpha=0.6)
    plt.plot([analysis_data['response'].min(), analysis_data['response'].max()], 
             [analysis_data['response'].min(), analysis_data['response'].max()], 'r--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Observed Values')
    plt.title('Fitted vs Observed')
    
    # Plot 3: Residual distribution
    plt.subplot(2, 3, 3)
    plt.hist(model.residuals, bins=30, alpha=0.7, density=True)
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.title('Residual Distribution')
    
    # Plot 4: Genotype BLUEs distribution
    plt.subplot(2, 3, 4)
    plt.hist(blues.values, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel(f'{target_trait} BLUEs')
    plt.ylabel('Frequency')
    plt.title('Distribution of Genotype BLUEs')
    
    # Plot 5: Top and bottom genotypes
    plt.subplot(2, 3, 5)
    blues_sorted = blues.sort_values()
    top_bottom = pd.concat([blues_sorted.head(10), blues_sorted.tail(10)])
    colors = ['red'] * 10 + ['green'] * 10
    plt.barh(range(len(top_bottom)), top_bottom.values, color=colors, alpha=0.7)
    plt.yticks(range(len(top_bottom)), top_bottom.index, fontsize=8)
    plt.xlabel(f'{target_trait} BLUEs')
    plt.title('Top 10 (green) and Bottom 10 (red) Genotypes')
    
    # Plot 6: Heritability visualization
    plt.subplot(2, 3, 6)
    variance_components = [var_g, var_e]
    labels = ['Genotypic', 'Error']
    colors = ['skyblue', 'lightcoral']
    plt.pie(variance_components, labels=labels, colors=colors, autopct='%1.1f%%')
    plt.title(f'Variance Components\n(h² = {heritability:.3f})')
    
    plt.tight_layout()
    plt.savefig('pyspats_sorghum_diagnostics.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Diagnostic plots saved to 'pyspats_sorghum_diagnostics.png'")
    
    # -------------------------------------------------------------------------
    # 6. Variogram Analysis
    # -------------------------------------------------------------------------
    print("\n6. Performing variogram analysis...")
    
    # Calculate variogram
    vario_result = variogram(model)
    print(f"   - Calculated variogram with {len(vario_result.distances)} distance bins")
    
    # Plot variogram
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(vario_result.distances, vario_result.gamma, 'bo-', linewidth=2, markersize=6)
    plt.xlabel('Distance')
    plt.ylabel('Semivariance')
    plt.title('Empirical Variogram')
    plt.grid(True, alpha=0.3)
    
    # Plot number of pairs
    plt.subplot(1, 2, 2)
    plt.plot(vario_result.distances, vario_result.n_pairs, 'ro-', linewidth=2, markersize=6)
    plt.xlabel('Distance')
    plt.ylabel('Number of Pairs')
    plt.title('Number of Pairs per Distance Bin')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pyspats_sorghum_variogram.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Variogram plots saved to 'pyspats_sorghum_variogram.png'")
    
    # -------------------------------------------------------------------------
    # 7. Multi-trait Analysis Summary
    # -------------------------------------------------------------------------
    print("\n7. Multi-trait analysis summary...")
    
    # Analyze multiple traits if available
    analysis_results = []
    test_traits = [col for col in trait_cols[:5] if data[col].notna().sum() > 100]  # Traits with sufficient data
    
    print(f"   - Analyzing {len(test_traits)} traits with sufficient data...")
    
    for trait in test_traits:
        try:
            # Prepare data
            trait_data = data.dropna(subset=[trait, 'PINumber']).copy()
            if len(trait_data) < 50:  # Skip if insufficient data
                continue
                
            trait_analysis = trait_data.rename(columns={
                'PINumber': 'genotype',
                'Column': 'col',
                'Row': 'row', 
                'Treatment': 'treatment',
                'Block': 'block',
                trait: 'response'
            })
            
            # Fit model
            trait_model = SpATS(
                response='response',
                genotype='genotype',
                spatial=('col', 'row'),
                fixed=['treatment'],
                random=['block'],
                data=trait_analysis
            )
            
            # Calculate heritability
            trait_blues = trait_model.get_BLUEs()
            trait_var_g = np.var(trait_blues.values, ddof=1)
            trait_var_e = trait_model.psi
            trait_h2 = trait_var_g / (trait_var_g + trait_var_e)
            
            analysis_results.append({
                'Trait': trait,
                'N_Observations': trait_model.n_obs,
                'N_Genotypes': len(trait_blues),
                'Mean': trait_analysis['response'].mean(),
                'Std': trait_analysis['response'].std(),
                'Heritability': trait_h2,
                'Genotypic_Variance': trait_var_g,
                'Error_Variance': trait_var_e
            })
            
        except Exception as e:
            print(f"     - Warning: Failed to analyze {trait}: {str(e)[:50]}...")
            continue
    
    # Create summary table
    if analysis_results:
        results_df = pd.DataFrame(analysis_results)
        print(f"\n   📊 Multi-trait Analysis Summary:")
        print("   " + "="*70)
        for _, row in results_df.iterrows():
            print(f"   {row['Trait']:<20} | N={row['N_Observations']:<4} | h²={row['Heritability']:.3f} | Mean={row['Mean']:.1f}")
        
        # Save results
        results_df.to_csv('pyspats_sorghum_results.csv', index=False)
        print(f"\n   ✅ Results saved to 'pyspats_sorghum_results.csv'")
    
    # -------------------------------------------------------------------------
    # 8. Summary and Recommendations
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("="*80)
    print(f"✅ Successfully analyzed {target_trait} using pySpATS")
    print(f"✅ Model fitted with {model.n_obs} observations")  
    print(f"✅ Extracted BLUEs for {len(blues)} genotypes")
    print(f"✅ Estimated heritability: {heritability:.3f}")
    print(f"✅ Generated diagnostic plots and variogram analysis")
    if analysis_results:
        print(f"✅ Completed multi-trait summary for {len(analysis_results)} traits")
    
    print(f"\nFiles generated:")
    print(f"  - pyspats_sorghum_diagnostics.png")
    print(f"  - pyspats_sorghum_variogram.png") 
    if analysis_results:
        print(f"  - pyspats_sorghum_results.csv")
    
    print(f"\nInterpretation:")
    if heritability > 0.7:
        print(f"  🟢 High heritability ({heritability:.3f}) - trait is highly heritable")
    elif heritability > 0.4:
        print(f"  🟡 Moderate heritability ({heritability:.3f}) - moderate genetic control")  
    else:
        print(f"  🔴 Low heritability ({heritability:.3f}) - environmental effects dominate")
    
    print(f"\nRecommendations:")
    print(f"  • Use the extracted BLUEs for downstream genomic analysis")
    print(f"  • Consider spatial effects when designing future trials")  
    print(f"  • Review diagnostic plots for model adequacy")
    print(f"  • Compare heritabilities across traits for selection priorities")
    
    print("\n" + "="*80)
    print("🎯 pySpATS analysis completed successfully!")
    print("="*80)

if __name__ == "__main__":
    main()