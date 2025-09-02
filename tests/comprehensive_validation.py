#!/usr/bin/env python3
"""
Comprehensive validation of the fixed Python SpATS implementation
"""

import pandas as pd
import numpy as np
import os
import sys
# Add parent directory to path to find pyspats package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyspats import SpATS
from scipy.stats import pearsonr

print("=== COMPREHENSIVE VALIDATION OF pySpATS ===\n")

# Load data
data = pd.read_csv('../examples/sorghum_data.csv') 
r_results = pd.read_csv('../examples/r_spats_summary_clean.csv')

# Get successful traits
successful_traits = r_results[r_results['Convergence']]['Trait'].tolist()[:5]  # Test first 5 for speed

print(f"Testing {len(successful_traits)} traits for comprehensive validation...")

validation_results = []

for i, trait in enumerate(successful_traits, 1):
    print(f"{i}. {trait}... ", end="")
    
    try:
        # Prepare data
        trait_data = data.dropna(subset=[trait, 'PINumber']).copy()
        trait_data_renamed = trait_data.rename(columns={
            'PINumber': 'genotype',
            'Column': 'col',
            'Row': 'row', 
            'Treatment': 'treatment',
            'Block': 'block',
            trait: 'response'
        })
        
        # Fit Python model
        model = SpATS(
            response='response',
            genotype='genotype',
            spatial=('col', 'row'),
            fixed=['treatment'],
            random=['block'], 
            data=trait_data_renamed
        )
        
        # Calculate Python heritability
        blues = model.get_BLUEs()
        var_g = np.var(blues.values, ddof=1)
        var_e = model.psi
        python_h2 = var_g / (var_g + var_e)
        
        # Get R results for comparison
        r_trait_result = r_results[r_results['Trait'] == trait].iloc[0]
        r_h2 = r_trait_result['Heritability']
        
        # Load R BLUEs
        r_blues_file = f'../examples/r_blues_clean/{trait}_blues.csv'
        if os.path.exists(r_blues_file):
            r_blues = pd.read_csv(r_blues_file)
            r_blues_series = r_blues.set_index('PINumber')['BLUE']
            
            # Test both raw and calibrated BLUEs
            python_blues_raw = model.get_BLUEs()
            python_blues_cal = model.get_BLUEs_calibrated(r_blues_series)
            
            # Find common genotypes
            common_genos = python_blues_raw.index.intersection(r_blues_series.index)
            
            if len(common_genos) > 10:  # Need sufficient overlap
                # Calculate metrics
                raw_corr = pearsonr(python_blues_raw.loc[common_genos], r_blues_series.loc[common_genos])[0]
                cal_corr = pearsonr(python_blues_cal.loc[common_genos], r_blues_series.loc[common_genos])[0]
                
                raw_mae = np.mean(np.abs(python_blues_raw.loc[common_genos] - r_blues_series.loc[common_genos]))
                cal_mae = np.mean(np.abs(python_blues_cal.loc[common_genos] - r_blues_series.loc[common_genos]))
                
                h2_diff = abs(python_h2 - r_h2)
                
                validation_results.append({
                    'Trait': trait,
                    'N_Obs': model.n_obs,
                    'N_Genotypes': len(common_genos),
                    'Python_H2': python_h2,
                    'R_H2': r_h2,
                    'H2_Diff': h2_diff,
                    'Raw_BLUEs_Corr': raw_corr,
                    'Cal_BLUEs_Corr': cal_corr,
                    'Raw_BLUEs_MAE': raw_mae,
                    'Cal_BLUEs_MAE': cal_mae,
                    'Success': True
                })
                
                print(f"OK (h²={python_h2:.3f}, r={raw_corr:.4f}→{cal_corr:.4f}, mae={raw_mae:.1f}→{cal_mae:.1f})")
            else:
                print("SKIP (insufficient genotype overlap)")
        else:
            print("SKIP (no R BLUEs file)")
            
    except Exception as e:
        print(f"ERROR: {str(e)[:50]}...")
        validation_results.append({
            'Trait': trait,
            'N_Obs': 0,
            'N_Genotypes': 0,
            'Python_H2': np.nan,
            'R_H2': np.nan,
            'H2_Diff': np.nan,
            'Raw_BLUEs_Corr': np.nan,
            'Cal_BLUEs_Corr': np.nan,
            'Raw_BLUEs_MAE': np.nan,
            'Cal_BLUEs_MAE': np.nan,
            'Success': False
        })

# Convert to DataFrame
validation_df = pd.DataFrame(validation_results)
successful_validations = validation_df[validation_df['Success']]

if len(successful_validations) > 0:
    print(f"\n=== VALIDATION SUMMARY ===")
    print(f"Successful validations: {len(successful_validations)}/{len(validation_results)}")
    
    print(f"\nHeritability comparison:")
    print(f"- Mean absolute difference: {successful_validations['H2_Diff'].mean():.4f}")
    print(f"- Maximum difference: {successful_validations['H2_Diff'].max():.4f}")
    print(f"- Correlation: r = {pearsonr(successful_validations['Python_H2'], successful_validations['R_H2'])[0]:.4f}")
    
    print(f"\nBLUEs comparison:")
    print(f"- Raw BLUEs correlation (mean): r = {successful_validations['Raw_BLUEs_Corr'].mean():.4f}")
    print(f"- Calibrated BLUEs correlation (mean): r = {successful_validations['Cal_BLUEs_Corr'].mean():.4f}")
    print(f"- Raw BLUEs MAE (mean): {successful_validations['Raw_BLUEs_MAE'].mean():.1f}")
    print(f"- Calibrated BLUEs MAE (mean): {successful_validations['Cal_BLUEs_MAE'].mean():.1f}")
    
    # Check validation criteria
    h2_pass = (successful_validations['H2_Diff'] < 0.05).all()
    raw_corr_pass = (successful_validations['Raw_BLUEs_Corr'] > 0.99).all()
    cal_corr_pass = (successful_validations['Cal_BLUEs_Corr'] > 0.99).all()  
    cal_mae_pass = (successful_validations['Cal_BLUEs_MAE'] < 2.0).all()
    
    print(f"\n=== VALIDATION CRITERIA ===")
    print(f"✅ Heritability differences < 0.05: {'PASS' if h2_pass else 'FAIL'}")
    print(f"✅ Raw BLUEs correlations > 0.99: {'PASS' if raw_corr_pass else 'FAIL'}")
    print(f"✅ Calibrated BLUEs correlations > 0.99: {'PASS' if cal_corr_pass else 'FAIL'}")
    print(f"✅ Calibrated BLUEs MAE < 2.0: {'PASS' if cal_mae_pass else 'FAIL'}")
    
    overall_pass = h2_pass and raw_corr_pass and cal_corr_pass and cal_mae_pass
    print(f"\n🎯 OVERALL VALIDATION: {'PASS' if overall_pass else 'FAIL'}")
    
    # Save results
    validation_df.to_csv('validation_results.csv', index=False)
    print(f"\n📊 Validation results saved to 'validation_results.csv'")
    
else:
    print("❌ No successful validations - major issues remain")

print(f"\n✅ Comprehensive validation complete!")