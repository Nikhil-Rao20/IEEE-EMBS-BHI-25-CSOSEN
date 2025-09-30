#!/usr/bin/env python3
"""
Main Experiment Runner for Corrected Dataset
IEEE EMBS BHI 2025 - Track 1

This script runs all experimental phases using the CORRECTED dataset with proper medical condition encoding.
"""

import sys
import os
from pathlib import Path
import json
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add the SRC_Track1 directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "SRC_Track1"
sys.path.insert(0, str(src_path))

# Import experiment modules
from experiment_framework import ExperimentFramework
from phase1_baseline_models import Phase1BaselineModels
from phase2_classical_ml import Phase2ClassicalML
from phase3_advanced_ensembles import Phase3AdvancedEnsembles
from phase4_deep_learning import Phase4DeepLearning
from phase5_timeseries import Phase5TimeSeriesModels
from results_compilation import ResultsCompilation as ResultsCompiler
from statistical_analysis import StatisticalAnalysis as StatisticalAnalyzer

def run_corrected_experiments():
    """Run all experiment phases with corrected dataset"""
    
    print("🚀 Starting Corrected Dataset Experiments")
    print("=" * 80)
    print(f"📅 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 Using CORRECTED dataset with proper medical condition encoding")
    print("📊 Dataset: train_corrected_features.xlsx")
    
    # Setup paths
    data_path = project_root / "Track1_Data" / "processed" / "train_corrected_features.xlsx"
    results_dir = project_root / "Results_Corrected_Data" / "Model_Experiments"
    
    # Verify corrected dataset exists
    if not data_path.exists():
        print(f"❌ ERROR: Corrected dataset not found at {data_path}")
        print("   Please run the corrected feature engineering notebook first!")
        return False
    
    # Load and validate corrected dataset
    print(f"\n📂 Loading corrected dataset from: {data_path}")
    try:
        data = pd.read_excel(data_path)
        print(f"✅ Dataset loaded successfully: {data.shape}")
        
        # Verify medical condition columns are present
        medical_condition_cols = [col for col in data.columns if col.startswith('condition_') and 
                                any(med in col for med in ['cancer', 'acute_coronary', 'renal', 'amputation'])]
        print(f"✅ Found {len(medical_condition_cols)} medical condition columns:")
        for col in medical_condition_cols:
            prevalence = data[col].sum()
            pct = (prevalence / len(data)) * 100
            print(f"   • {col}: {prevalence} patients ({pct:.1f}%)")
        
        # Verify condition type columns
        condition_type_cols = [col for col in data.columns if col.startswith('condition_type_')]
        print(f"✅ Found {len(condition_type_cols)} condition type columns:")
        for col in condition_type_cols:
            count = data[col].sum()
            pct = (count / len(data)) * 100
            print(f"   • {col}: {count} patients ({pct:.1f}%)")
            
    except Exception as e:
        print(f"❌ ERROR loading dataset: {e}")
        return False
    
    # Initialize experiment framework with corrected data
    framework = ExperimentFramework(random_seed=42)
    
    print(f"\n🔧 Experiment framework initialized")
    print(f"📁 Results will be saved to: {results_dir}")
    
    # Timestamp for result files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Run all phases
    phases = [
        ("Phase 1: Baseline Models", Phase1BaselineModels),
        ("Phase 2: Classical ML", Phase2ClassicalML), 
        ("Phase 3: Advanced Ensembles", Phase3AdvancedEnsembles),
        ("Phase 4: Deep Learning", Phase4DeepLearning),
        ("Phase 5: Time Series Models", Phase5TimeSeriesModels)
    ]
    
    phase_results = {}
    
    for phase_name, phase_class in phases:
        print(f"\n{'='*80}")
        print(f"🧪 Running {phase_name}")
        print(f"{'='*80}")
        
        try:
            # Initialize phase
            phase_runner = phase_class(framework)
            
            # Run phase experiments
            results = phase_runner.run_all_experiments()
            
            # Save phase results
            phase_num = phase_name.split(':')[0].split()[-1]
            result_file = results_dir / f"phase{phase_num.lower()}_corrected_results_{timestamp}.json"
            
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            phase_results[f"phase{phase_num.lower()}"] = results
            
            print(f"✅ {phase_name} completed successfully")
            print(f"📄 Results saved to: {result_file}")
            
        except Exception as e:
            print(f"❌ ERROR in {phase_name}: {e}")
            print(f"   Continuing with next phase...")
            phase_results[f"phase{phase_num.lower()}"] = {"error": str(e)}
    
    # Compile all results
    print(f"\n{'='*80}")
    print("📊 Compiling All Results")
    print(f"{'='*80}")
    
    try:
        compiler = ResultsCompiler(
            results_dir=str(results_dir),
            output_dir=str(project_root / "Results_Corrected_Data" / "Conference_Submission")
        )
        
        compilation_results = compiler.compile_all_results(
            experiment_type="corrected_dataset",
            timestamp=timestamp
        )
        
        print("✅ Results compilation completed")
        
    except Exception as e:
        print(f"❌ ERROR in results compilation: {e}")
        compilation_results = {"error": str(e)}
    
    # Statistical analysis
    print(f"\n{'='*80}")
    print("📈 Running Statistical Analysis")
    print(f"{'='*80}")
    
    try:
        analyzer = StatisticalAnalyzer(
            results_dir=str(project_root / "Results_Corrected_Data" / "Conference_Submission"),
            output_dir=str(project_root / "Results_Corrected_Data" / "Statistical_Analysis")
        )
        
        statistical_results = analyzer.run_comprehensive_analysis(
            experiment_type="corrected_dataset"
        )
        
        print("✅ Statistical analysis completed")
        
    except Exception as e:
        print(f"❌ ERROR in statistical analysis: {e}")
        statistical_results = {"error": str(e)}
    
    # Final summary
    print(f"\n{'='*80}")
    print("🎯 CORRECTED DATASET EXPERIMENTS COMPLETED")
    print(f"{'='*80}")
    print(f"📅 End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Dataset Used: train_corrected_features.xlsx")
    print(f"📁 Results Location: Results_Corrected_Data/")
    
    # Count successful phases
    successful_phases = sum(1 for result in phase_results.values() if "error" not in result)
    total_phases = len(phases)
    
    print(f"\n📈 Experiment Summary:")
    print(f"   • Successful Phases: {successful_phases}/{total_phases}")
    print(f"   • Medical Conditions: {len(medical_condition_cols)} properly encoded")
    print(f"   • Condition Types: {len(condition_type_cols)} properly encoded") 
    print(f"   • Dataset Quality: Complete with all real medical conditions")
    
    # Save experiment metadata
    experiment_metadata = {
        "experiment_type": "corrected_dataset_experiments",
        "timestamp": timestamp,
        "dataset_path": str(data_path),
        "dataset_shape": data.shape,
        "medical_conditions": len(medical_condition_cols),
        "condition_types": len(condition_type_cols),
        "successful_phases": successful_phases,
        "total_phases": total_phases,
        "phase_results": {k: ("success" if "error" not in v else "failed") for k, v in phase_results.items()},
        "compilation_status": "success" if "error" not in compilation_results else "failed",
        "statistical_analysis_status": "success" if "error" not in statistical_results else "failed"
    }
    
    metadata_path = project_root / "Results_Corrected_Data" / "experiment_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(experiment_metadata, f, indent=2, default=str)
    
    print(f"📄 Experiment metadata saved to: {metadata_path}")
    
    return successful_phases == total_phases

if __name__ == "__main__":
    success = run_corrected_experiments()
    exit_code = 0 if success else 1
    sys.exit(exit_code)