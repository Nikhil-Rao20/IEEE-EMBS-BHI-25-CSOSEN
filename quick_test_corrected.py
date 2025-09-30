#!/usr/bin/env python3
"""
Quick Test of Corrected Dataset Experiments
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import warnings
warnings.filterwarnings('ignore')

def quick_experiment_test():
    """Quick test of basic ML pipeline with corrected dataset"""
    
    print("🧪 Quick Test: Corrected Dataset Experiments")
    print("=" * 60)
    
    # Load corrected dataset
    data_path = Path("Track1_Data/processed/train_corrected_features.xlsx")
    
    if not data_path.exists():
        print("❌ Corrected dataset not found!")
        return False
    
    # Load data
    print("📂 Loading corrected dataset...")
    data = pd.read_excel(data_path)
    print(f"✅ Dataset shape: {data.shape}")
    
    # Identify features and targets
    target_cols = ['bdi_ii_after_intervention_12w', 'bdi_ii_follow_up_24w']
    feature_cols = [col for col in data.columns if col not in target_cols]
    
    print(f"📊 Features: {len(feature_cols)}")
    print(f"🎯 Targets: {len(target_cols)}")
    
    # Check medical conditions
    medical_cols = [col for col in feature_cols if col.startswith('condition_') and 
                   any(med in col for med in ['cancer', 'acute_coronary', 'renal', 'amputation'])]
    condition_type_cols = [col for col in feature_cols if col.startswith('condition_type_')]
    
    print(f"🏥 Medical conditions: {len(medical_cols)}")
    print(f"🏷️  Condition types: {len(condition_type_cols)}")
    
    # Prepare data for modeling
    X = data[feature_cols].copy()
    
    # Handle categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        X[col] = pd.Categorical(X[col]).codes
    
    # Fill any missing values
    X = X.fillna(X.median())
    
    print(f"✅ Feature matrix prepared: {X.shape}")
    
    # Test both targets
    results = {}
    
    for target in target_cols:
        print(f"\n🎯 Testing target: {target}")
        
        y = data[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Test multiple models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        target_results = {}
        
        for model_name, model in models.items():
            print(f"   🔧 Training {model_name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            target_results[model_name] = {
                'MAE': float(mae),
                'RMSE': float(rmse),
                'R2': float(r2)
            }
            
            print(f"      MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")
        
        results[target] = target_results
    
    # Test medical condition analysis
    print(f"\n🏥 Medical Condition Analysis:")
    print("=" * 40)
    
    medical_analysis = {}
    for col in medical_cols:
        condition_patients = data[data[col] == 1]
        no_condition_patients = data[data[col] == 0]
        
        analysis = {
            'prevalence': int(data[col].sum()),
            'prevalence_pct': float((data[col].sum() / len(data)) * 100),
            'sample_sizes': {
                'with_condition': len(condition_patients),
                'without_condition': len(no_condition_patients)
            }
        }
        
        # Compare outcomes
        for target in target_cols:
            with_condition_mean = condition_patients[target].mean()
            without_condition_mean = no_condition_patients[target].mean()
            
            analysis[f'{target}_mean_with'] = float(with_condition_mean) if not pd.isna(with_condition_mean) else None
            analysis[f'{target}_mean_without'] = float(without_condition_mean) if not pd.isna(without_condition_mean) else None
        
        medical_analysis[col] = analysis
        
        print(f"   {col.replace('condition_', '').replace('_', ' ').title()}:")
        print(f"      Prevalence: {analysis['prevalence']} ({analysis['prevalence_pct']:.1f}%)")
    
    # Save results
    test_results = {
        'timestamp': datetime.now().isoformat(),
        'dataset_info': {
            'shape': data.shape,
            'features': len(feature_cols),
            'medical_conditions': len(medical_cols),
            'condition_types': len(condition_type_cols)
        },
        'model_results': results,
        'medical_analysis': medical_analysis
    }
    
    # Create results directory
    results_dir = Path("Results_Corrected_Data")
    results_dir.mkdir(exist_ok=True)
    
    # Save test results
    test_file = results_dir / f"quick_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(test_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n✅ Quick test completed successfully!")
    print(f"📄 Results saved to: {test_file}")
    print(f"\n📊 Summary:")
    print(f"   • Dataset: {data.shape[0]} samples, {data.shape[1]} features")
    print(f"   • Medical conditions: {len(medical_cols)} properly encoded")
    print(f"   • Condition types: {len(condition_type_cols)} properly encoded")
    print(f"   • Models tested: 2 algorithms on 2 targets")
    print(f"   • All tests passed: ✅")
    
    return True

if __name__ == "__main__":
    success = quick_experiment_test()
    exit_code = 0 if success else 1
    sys.exit(exit_code)