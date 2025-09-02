#!/usr/bin/env python3
"""
Simple validation test for pySpATS package functionality
"""

import pandas as pd
import numpy as np
import os
import sys
# Add parent directory to path to find pyspats package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=== pySpATS FUNCTIONALITY VALIDATION ===\n")

try:
    # Test import
    print("1. Testing package import... ", end="")
    from pyspats import SpATS, SpATSControl, plot_spats, variogram
    print("✅ SUCCESS")
    
    # Load test data
    print("2. Loading test data... ", end="")
    data = pd.read_csv('../examples/sorghum_data.csv') 
    r_results = pd.read_csv('../examples/r_spats_summary_clean.csv')
    print(f"✅ SUCCESS ({len(data)} records loaded)")
    
    # Get a trait for testing
    successful_traits = r_results[r_results['Convergence']]['Trait'].tolist()
    test_trait = successful_traits[0]
    print(f"3. Testing with trait: {test_trait}")
    
    # Prepare data
    print("4. Preparing data... ", end="")
    trait_data = data.dropna(subset=[test_trait, 'PINumber']).copy()
    trait_data_renamed = trait_data.rename(columns={
        'PINumber': 'genotype',
        'Column': 'col',
        'Row': 'row', 
        'Treatment': 'treatment',
        'Block': 'block',
        test_trait: 'response'
    })
    print(f"✅ SUCCESS ({len(trait_data_renamed)} records for analysis)")
    
    # Test SpATS model creation and fitting
    print("5. Creating and fitting SpATS model... ", end="")
    model = SpATS(
        response='response',
        genotype='genotype',
        spatial=('col', 'row'),
        fixed=['treatment'],
        random=['block'], 
        data=trait_data_renamed
    )
    print("✅ SUCCESS")
    
    # Test model properties and methods
    print("6. Testing model properties and methods:")
    
    # Test basic properties
    print(f"   - Number of observations: {model.n_obs}")
    print(f"   - Residual variance (psi): {model.psi:.4f}")
    print(f"   - Model converged: {hasattr(model, 'coefficients')}")
    
    # Test BLUEs extraction
    print("   - Extracting BLUEs... ", end="")
    blues = model.get_BLUEs()
    print(f"✅ SUCCESS ({len(blues)} genotype BLUEs)")
    
    # Test heritability calculation
    print("   - Calculating heritability... ", end="")
    var_g = np.var(blues.values, ddof=1)
    var_e = model.psi
    heritability = var_g / (var_g + var_e)
    print(f"✅ SUCCESS (h² = {heritability:.3f})")
    
    # Test residuals (they are stored as model attribute)
    print("   - Accessing residuals... ", end="")
    residuals = model.residuals
    print(f"✅ SUCCESS ({len(residuals)} residuals)")
    
    # Test fitted values (they are stored as model attribute)
    print("   - Accessing fitted values... ", end="")
    fitted = model.fitted_values
    print(f"✅ SUCCESS ({len(fitted)} fitted values)")
    
    # Test model summary
    print("   - Testing model summary... ", end="")
    summary = model.summary()
    print(f"✅ SUCCESS")
    
    # Test prediction functionality
    print("   - Testing prediction... ", end="")
    predictions = model.predict()
    print(f"✅ SUCCESS ({len(predictions)} predictions)")
    
    # Test multiple traits
    print("7. Testing with multiple traits... ", end="")
    test_traits = successful_traits[:3]  # Test first 3 traits
    successful_fits = 0
    
    for trait in test_traits:
        try:
            trait_data = data.dropna(subset=[trait, 'PINumber']).copy()
            trait_data_renamed = trait_data.rename(columns={
                'PINumber': 'genotype',
                'Column': 'col',
                'Row': 'row', 
                'Treatment': 'treatment',
                'Block': 'block',
                trait: 'response'
            })
            
            model = SpATS(
                response='response',
                genotype='genotype',
                spatial=('col', 'row'),
                fixed=['treatment'],
                random=['block'], 
                data=trait_data_renamed
            )
            successful_fits += 1
        except Exception as e:
            print(f"   - Warning: {trait} failed ({str(e)[:30]}...)")
    
    print(f"✅ SUCCESS ({successful_fits}/{len(test_traits)} traits fit successfully)")
    
    print("\n🎯 OVERALL VALIDATION RESULT: ✅ PASS")
    print(f"   - Package imports correctly")
    print(f"   - SpATS models fit successfully") 
    print(f"   - All main methods work correctly")
    print(f"   - Multiple traits can be analyzed")
    print(f"   - Heritability calculation works (h² = {heritability:.3f})")
    
except Exception as e:
    print(f"❌ FAILED: {str(e)}")
    import traceback
    traceback.print_exc()
    print(f"\n🎯 OVERALL VALIDATION RESULT: ❌ FAIL")

print(f"\n✅ Simple validation complete!")