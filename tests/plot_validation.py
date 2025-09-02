#!/usr/bin/env python3
"""
Test pySpATS plotting functionality
"""

import pandas as pd
import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
# Add parent directory to path to find pyspats package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=== pySpATS PLOTTING VALIDATION ===\n")

try:
    # Test import
    print("1. Testing plotting imports... ", end="")
    from pyspats import SpATS, plot_spats, plot_variogram, variogram
    import matplotlib.pyplot as plt
    print("✅ SUCCESS")
    
    # Load test data and fit model
    print("2. Loading data and fitting model... ", end="")
    data = pd.read_csv('../examples/sorghum_data.csv') 
    r_results = pd.read_csv('../examples/r_spats_summary_clean.csv')
    
    successful_traits = r_results[r_results['Convergence']]['Trait'].tolist()
    test_trait = successful_traits[0]
    
    trait_data = data.dropna(subset=[test_trait, 'PINumber']).copy()
    trait_data_renamed = trait_data.rename(columns={
        'PINumber': 'genotype',
        'Column': 'col',
        'Row': 'row', 
        'Treatment': 'treatment',
        'Block': 'block',
        test_trait: 'response'
    })
    
    model = SpATS(
        response='response',
        genotype='genotype',
        spatial=('col', 'row'),
        fixed=['treatment'],
        random=['block'], 
        data=trait_data_renamed
    )
    print("✅ SUCCESS")
    
    # Test model plotting methods
    print("3. Testing model plotting methods:")
    
    # Test plot_residuals
    print("   - plot_residuals... ", end="")
    model.plot_residuals(show=False)
    plt.close('all')
    print("✅ SUCCESS")
    
    # Test plot_fitted
    print("   - plot_fitted... ", end="")
    model.plot_fitted(show=False)
    plt.close('all')
    print("✅ SUCCESS")
    
    # Test plot_spatial
    print("   - plot_spatial... ", end="")
    model.plot_spatial(show=False)
    plt.close('all')
    print("✅ SUCCESS")
    
    # Test main plot method
    print("   - plot (main method)... ", end="")
    model.plot(all_in_one=True, figsize=(12, 8), show=False)
    plt.close('all')
    print("✅ SUCCESS")
    
    # Test standalone plotting functions
    print("4. Testing standalone plotting functions:")
    
    # Test plot_spats function
    print("   - plot_spats function... ", end="")
    fig = plot_spats(model, which='spatial')
    plt.close('all')
    print("✅ SUCCESS")
    
    # Test variogram function
    print("   - variogram calculation... ", end="")
    vario_result = variogram(model)
    print("✅ SUCCESS")
    
    # Test plot_variogram function
    print("   - plot_variogram function... ", end="")
    fig = plot_variogram(vario_result)
    plt.close('all')
    print("✅ SUCCESS")
    
    print("\n🎯 PLOTTING VALIDATION RESULT: ✅ PASS")
    print("   - All model plotting methods work")
    print("   - Standalone plotting functions work")
    print("   - Variogram calculation and plotting work")
    print("   - No plotting errors or crashes")
    
except Exception as e:
    print(f"❌ FAILED: {str(e)}")
    import traceback
    traceback.print_exc()
    print(f"\n🎯 PLOTTING VALIDATION RESULT: ❌ FAIL")

print(f"\n✅ Plotting validation complete!")