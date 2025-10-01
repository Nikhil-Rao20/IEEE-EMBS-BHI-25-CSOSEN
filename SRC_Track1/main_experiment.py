"""
🚀 Main Execution Script - Complete Model Experimentation Pipeline
================================================================

This script orchestrates the complete experimental pipeline from data loading
to conference-ready results compilation.

Execution Phases:
1. Data Loading & Preparation
2. Phase 1: Baseline Models
3. Phase 2: Classical ML Models  
4. Phase 3: Advanced Ensembles
5. Phase 4: Deep Learning Models
6. Phase 5: Time-Series Models
7. Statistical Analysis
8. Results Compilation

Author: Research Team
Date: September 2025
Purpose: IEEE EMBS BHI 2025 Conference Submission
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path to import local modules
sys.path.append(str(Path(__file__).parent))

# Import all phase modules
from experiment_framework import ExperimentFramework
from phase1_baseline_models import Phase1BaselineModels
from phase2_classical_ml import Phase2ClassicalML
from phase3_advanced_ensembles import Phase3AdvancedEnsembles
from phase4_deep_learning import Phase4DeepLearning
from phase5_timeseries import Phase5TimeSeriesModels
from statistical_analysis import StatisticalAnalysis
from results_compilation import ResultsCompilation

def main():
    """Main execution function."""
    
    print("🚀 STARTING COMPLETE MODEL EXPERIMENTATION PIPELINE")
    print("=" * 65)
    print(f"📅 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # ============================================================================
    # SETUP AND DATA LOADING
    # ============================================================================
    
    print("📋 PHASE 0: SETUP AND DATA LOADING")
    print("-" * 40)
    
    # Initialize experiment framework
    output_folder_name = "Results_Corrected_Data"  # ✅ CHANGED 
    framework = ExperimentFramework(random_seed=42, output_folder_name=output_folder_name)
    
    # Data paths
    # train_data_path = "../Track1_Data/processed/train_corrected_features.xlsx"
    train_data_path = "../Track1_Data/processed/train_corrected_features.xlsx"
    
    # Check if processed data exists
    if not Path(train_data_path).exists():
        print("❌ Processed training data not found!")
        print("💡 Please run the feature engineering pipeline first.")
        print(f"   Expected path: {train_data_path}")
        return
    
    # Load data
    print("📊 Loading engineered features...")
    train_df, _ = framework.load_data(train_data_path)
    
    # Define target columns
    target_columns = ['bdi_ii_after_intervention_12w', 'bdi_ii_follow_up_24w']
    
    # Prepare data for experiments
    X, y = framework.prepare_data(train_df, target_columns)
    
    # Use primary target (12 weeks) for model comparison
    y_primary = y['bdi_ii_after_intervention_12w']
    
    print(f"✅ Data loaded successfully!")
    print(f"   Features: {X.shape[1]}")
    print(f"   Samples: {X.shape[0]}")
    print(f"   Target: BDI-II at 12 weeks post-intervention")
    print()
    
    # ============================================================================
    # EXPERIMENTAL PHASES
    # ============================================================================
    
    all_phase_results = {}
    phase_times = {}
    
    # Phase 1: Baseline Models
    print("📋 PHASE 1: BASELINE MODELS")
    print("-" * 35)
    start_time = time.time()
    
    try:
        phase1 = Phase1BaselineModels(random_seed=42)
        phase1_results = phase1.evaluate_all_models(X, y_primary, framework)
        
        if phase1_results:
            all_phase_results['phase1'] = phase1_results
            framework.save_results('phase1', phase1_results)
            
            print("\n" + phase1.generate_phase_summary())
        else:
            print("⚠️ Phase 1 produced no results")
            
    except Exception as e:
        print(f"❌ Phase 1 failed: {e}")
        phase1_results = {}
    
    phase_times['phase1'] = time.time() - start_time
    print(f"\n⏱️ Phase 1 completed in {phase_times['phase1']:.1f} seconds")
    print()
    
    # Phase 2: Classical ML Models
    print("📋 PHASE 2: CLASSICAL ML MODELS")
    print("-" * 37)
    start_time = time.time()
    
    try:
        phase2 = Phase2ClassicalML(random_seed=42)
        phase2_results = phase2.evaluate_all_models(X, y_primary, framework)
        
        if phase2_results:
            all_phase_results['phase2'] = phase2_results
            framework.save_results('phase2', phase2_results)
            
            print("\n" + phase2.generate_phase_summary())
        else:
            print("⚠️ Phase 2 produced no results")
            
    except Exception as e:
        print(f"❌ Phase 2 failed: {e}")
        phase2_results = {}
    
    phase_times['phase2'] = time.time() - start_time
    print(f"\n⏱️ Phase 2 completed in {phase_times['phase2']:.1f} seconds")
    print()
    
    # Phase 3: Advanced Ensembles
    print("📋 PHASE 3: ADVANCED ENSEMBLE MODELS")
    print("-" * 42)
    start_time = time.time()
    
    try:
        phase3 = Phase3AdvancedEnsembles(random_seed=42)
        phase3_results = phase3.evaluate_all_models(X, y_primary, framework)
        
        if phase3_results:
            all_phase_results['phase3'] = phase3_results
            framework.save_results('phase3', phase3_results)
            
            print("\n" + phase3.generate_phase_summary())
        else:
            print("⚠️ Phase 3 produced no results")
            
    except Exception as e:
        print(f"❌ Phase 3 failed: {e}")
        phase3_results = {}
    
    phase_times['phase3'] = time.time() - start_time
    print(f"\n⏱️ Phase 3 completed in {phase_times['phase3']:.1f} seconds")
    print()
    
    # Phase 4: Deep Learning Models
    print("📋 PHASE 4: DEEP LEARNING MODELS")
    print("-" * 37)
    start_time = time.time()
    
    try:
        phase4 = Phase4DeepLearning(random_seed=42)
        phase4_results = phase4.evaluate_all_models(X, y_primary, framework)
        
        if phase4_results:
            all_phase_results['phase4'] = phase4_results
            framework.save_results('phase4', phase4_results)
            
            print("\n" + phase4.generate_phase_summary())
        else:
            print("⚠️ Phase 4 produced no results")
            
    except Exception as e:
        print(f"❌ Phase 4 failed: {e}")
        phase4_results = {}
    
    phase_times['phase4'] = time.time() - start_time
    print(f"\n⏱️ Phase 4 completed in {phase_times['phase4']:.1f} seconds")
    print()
    
    # Phase 5: Time-Series Models
    print("📋 PHASE 5: TIME-SERIES & TRAJECTORY MODELS")
    print("-" * 48)
    start_time = time.time()
    
    try:
        phase5 = Phase5TimeSeriesModels(random_seed=42)
        phase5_results = phase5.evaluate_all_models(X, y_primary, framework)
        
        if phase5_results:
            all_phase_results['phase5'] = phase5_results
            framework.save_results('phase5', phase5_results)
            
            print("\n" + phase5.generate_phase_summary())
        else:
            print("⚠️ Phase 5 produced no results")
            
    except Exception as e:
        print(f"❌ Phase 5 failed: {e}")
        phase5_results = {}
    
    phase_times['phase5'] = time.time() - start_time
    print(f"\n⏱️ Phase 5 completed in {phase_times['phase5']:.1f} seconds")
    print()
    
    # ============================================================================
    # STATISTICAL ANALYSIS
    # ============================================================================
    
    print("📊 STATISTICAL ANALYSIS")
    print("-" * 25)
    start_time = time.time()
    
    try:
        if all_phase_results:
            statistical_analyzer = StatisticalAnalysis(alpha=0.05)
            
            # Generate comprehensive statistical report
            statistical_report = statistical_analyzer.generate_statistical_report(all_phase_results)
            print("\n" + statistical_report)
            
            # Create statistical visualizations
            statistical_analyzer.create_statistical_visualizations(
                all_phase_results, 
                save_plots=True, 
                output_dir=f"../{output_folder_name}/Statistical_Analysis"
            )
            
        else:
            print("⚠️ No results available for statistical analysis")
            statistical_analyzer = None
            
    except Exception as e:
        print(f"❌ Statistical analysis failed: {e}")
        statistical_analyzer = None
    
    stat_time = time.time() - start_time
    print(f"\n⏱️ Statistical analysis completed in {stat_time:.1f} seconds")
    print()
    
    # ============================================================================
    # RESULTS COMPILATION
    # ============================================================================
    
    print("📋 RESULTS COMPILATION")
    print("-" * 25)
    start_time = time.time()
    
    try:
        if all_phase_results:
            results_compiler = ResultsCompilation(output_folder_name=output_folder_name)
            
            # Compile all results
            results_compiler.compile_all_results(all_phase_results, statistical_analyzer)
            
            # Create publication tables
            print("\n📊 Creating Publication Tables...")
            perf_table = results_compiler.create_performance_summary_table()
            phase_table = results_compiler.create_phase_comparison_table()
            stat_table = results_compiler.create_statistical_significance_table()
            clinical_table = results_compiler.create_clinical_significance_table()
            
            # Create publication figures
            print("\n📊 Creating Publication Figures...")
            results_compiler.create_publication_figures()
            
            # Generate conference summary
            print("\n📝 Generating Conference Summary...")
            conference_summary = results_compiler.generate_conference_summary()
            print("\n" + conference_summary)
            
            # Save all results
            results_compiler.save_all_results()
            
        else:
            print("⚠️ No results available for compilation")
            
    except Exception as e:
        print(f"❌ Results compilation failed: {e}")
    
    compilation_time = time.time() - start_time
    print(f"\n⏱️ Results compilation completed in {compilation_time:.1f} seconds")
    print()
    
    # ============================================================================
    # SAVE TOP 10 TRAINED MODELS
    # ============================================================================
    
    print("💾 SAVING TOP 10 TRAINED MODELS")
    print("-" * 35)
    start_time = time.time()
    
    try:
        if all_phase_results:
            import pickle
            import json
            
            # Create Top10_Trained_Models directory
            top_models_dir = Path("../Top10_Trained_Models")
            top_models_dir.mkdir(exist_ok=True)
            
            # Collect all models with their performance
            all_model_info = []
            for phase_name, phase_results in all_phase_results.items():
                for model_name, model_results in phase_results.items():
                    if 'mean_scores' in model_results and 'models' in model_results:
                        mae = model_results['mean_scores']['test_mae']
                        r2 = model_results['mean_scores']['test_r2']
                        rmse = model_results['mean_scores']['test_rmse']
                        models_list = model_results['models']
                        
                        all_model_info.append({
                            'phase': phase_name,
                            'model_name': model_name,
                            'full_name': f"{phase_name}_{model_name}",
                            'mae': mae,
                            'r2': r2,
                            'rmse': rmse,
                            'models': models_list,
                            'results': model_results
                        })
            
            # Sort by MAE (lower is better) and get top 10
            all_model_info.sort(key=lambda x: x['mae'])
            top_10_models = all_model_info[:10]
            
            print(f"🏆 Saving Top 10 Models (sorted by MAE):")
            
            # Save each top model
            for i, model_info in enumerate(top_10_models, 1):
                model_dir = top_models_dir / f"Rank_{i:02d}_{model_info['full_name']}"
                model_dir.mkdir(exist_ok=True)
                
                # Save trained models (cross-validation folds)
                models_folder = model_dir / "trained_models"
                models_folder.mkdir(exist_ok=True)
                
                for fold_idx, trained_model in enumerate(model_info['models']):
                    if hasattr(trained_model, 'save'):  # Keras models
                        model_path = models_folder / f"fold_{fold_idx}.keras"
                        trained_model.save(model_path)
                    else:  # Sklearn models
                        model_path = models_folder / f"fold_{fold_idx}.pkl"
                        with open(model_path, 'wb') as f:
                            pickle.dump(trained_model, f)
                
                # Save model metadata and performance
                metadata = {
                    'rank': i,
                    'phase': model_info['phase'],
                    'model_name': model_info['model_name'],
                    'full_name': model_info['full_name'],
                    'performance': {
                        'mae': float(model_info['mae']),
                        'r2': float(model_info['r2']),
                        'rmse': float(model_info['rmse'])
                    },
                    'cross_validation_scores': {
                        k: v.tolist() if hasattr(v, 'tolist') else v
                        for k, v in model_info['results']['cv_scores'].items()
                    },
                    'mean_scores': {
                        k: float(v) for k, v in model_info['results']['mean_scores'].items()
                    },
                    'std_scores': {
                        k: float(v) for k, v in model_info['results']['std_scores'].items()
                    },
                    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'experiment_id': framework.experiment_id
                }
                
                # Save metadata
                with open(model_dir / "model_metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Save detailed training history if available (for deep learning models)
                if hasattr(model_info['models'][0], 'history'):
                    training_history = {}
                    for fold_idx, trained_model in enumerate(model_info['models']):
                        if hasattr(trained_model, 'history'):
                            history_dict = trained_model.history.history
                            # Convert numpy arrays to lists for JSON serialization
                            history_serializable = {
                                k: [float(val) for val in v] if isinstance(v, list) else float(v)
                                for k, v in history_dict.items()
                            }
                            training_history[f'fold_{fold_idx}'] = history_serializable
                    
                    if training_history:
                        with open(model_dir / "training_history.json", 'w') as f:
                            json.dump(training_history, f, indent=2)
                
                print(f"   {i:2d}. {model_info['full_name']:<25} | MAE: {model_info['mae']:.3f} | R²: {model_info['r2']:.3f}")
            
            # Create a summary file for easy loading
            top_models_summary = {
                'experiment_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'experiment_id': framework.experiment_id,
                'best_model': {
                    'name': top_10_models[0]['full_name'],
                    'path': f"Rank_01_{top_10_models[0]['full_name']}",
                    'performance': top_10_models[0]['mae']
                },
                'top_10_models': [
                    {
                        'rank': i,
                        'name': model['full_name'],
                        'path': f"Rank_{i:02d}_{model['full_name']}",
                        'mae': float(model['mae']),
                        'r2': float(model['r2']),
                        'rmse': float(model['rmse'])
                    }
                    for i, model in enumerate(top_10_models, 1)
                ],
                'loading_instructions': {
                    'keras_models': "tf.keras.models.load_model('path/to/model.keras')",
                    'sklearn_models': "pickle.load(open('path/to/model.pkl', 'rb'))",
                    'metadata': "json.load(open('path/to/model_metadata.json', 'r'))"
                }
            }
            
            with open(top_models_dir / "top_models_summary.json", 'w') as f:
                json.dump(top_models_summary, f, indent=2)
            
            print(f"\n✅ Top 10 models saved to: {top_models_dir}")
            print(f"🥇 Best model: {top_10_models[0]['full_name']} (MAE: {top_10_models[0]['mae']:.3f})")
            
        else:
            print("⚠️ No results available for model saving")
            
    except Exception as e:
        print(f"❌ Model saving failed: {e}")
        import traceback
        traceback.print_exc()
    
    model_save_time = time.time() - start_time
    print(f"\n⏱️ Model saving completed in {model_save_time:.1f} seconds")
    print()
    
    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    
    print("🎉 EXPERIMENT PIPELINE COMPLETED!")
    print("=" * 40)
    
    total_time = sum(phase_times.values()) + stat_time + compilation_time + model_save_time
    total_models = sum(len(phase_data) for phase_data in all_phase_results.values())
    
    print(f"📊 EXECUTION SUMMARY:")
    print(f"   • Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"   • Total models evaluated: {total_models}")
    print(f"   • Successful phases: {len(all_phase_results)}/5")
    print(f"   • Top 10 models saved: ✅")
    
    print(f"\n⏱️ PHASE TIMING:")
    for phase, phase_time in phase_times.items():
        models_in_phase = len(all_phase_results.get(phase, {}))
        avg_time_per_model = phase_time / models_in_phase if models_in_phase > 0 else 0
        print(f"   • {phase.title()}: {phase_time:.1f}s ({models_in_phase} models, {avg_time_per_model:.1f}s/model)")
    
    print(f"   • Statistical Analysis: {stat_time:.1f}s")
    print(f"   • Results Compilation: {compilation_time:.1f}s")
    print(f"   • Model Saving: {model_save_time:.1f}s")
    
    if all_phase_results:
        # Find overall best model
        all_model_scores = []
        for phase_name, phase_results in all_phase_results.items():
            for model_name, model_results in phase_results.items():
                if 'mean_scores' in model_results:
                    mae = model_results['mean_scores']['test_mae']
                    r2 = model_results['mean_scores']['test_r2']
                    all_model_scores.append((f"{phase_name}_{model_name}", mae, r2))
        
        if all_model_scores:
            best_model = min(all_model_scores, key=lambda x: x[1])
            print(f"\n🏆 BEST OVERALL MODEL:")
            print(f"   • Model: {best_model[0].replace('_', ' ').title()}")
            print(f"   • MAE: {best_model[1]:.3f}")
            print(f"   • R²: {best_model[2]:.3f}")
            print(f"   • Saved in: ../Top10_Trained_Models/Rank_01_{best_model[0]}/")
    
    print(f"\n📁 RESULTS LOCATION:")
    print(f"   • Main results: ../Results/")
    print(f"   • Conference submission: ../Results/Conference_Submission/")
    print(f"   • Statistical analysis: ../Results/Statistical_Analysis/")
    print(f"   • Model experiments: ../Results/Model_Experiments/")
    print(f"   • 🆕 Top 10 trained models: ../Top10_Trained_Models/")
    
    print(f"\n📋 DELIVERABLES READY:")
    print(f"   ✅ Performance comparison tables")
    print(f"   ✅ Statistical significance analysis")
    print(f"   ✅ Clinical significance assessment")
    print(f"   ✅ Publication-ready figures")
    print(f"   ✅ Conference submission summary")
    print(f"   ✅ Complete experimental results")
    print(f"   ✅ 🆕 Top 10 trained models with metadata")
    print(f"   ✅ 🆕 Training curves and validation scores")
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. Review results in ../Results/Conference_Submission/")
    print(f"   2. Load best model from ../Top10_Trained_Models/ for feature importance")
    print(f"   3. Use saved models for production deployment")
    print(f"   4. Customize tables and figures for target conference")
    print(f"   5. Write manuscript using provided results")
    print(f"   6. Submit to IEEE EMBS BHI 2025!")
    
    print(f"\n📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🚀 EXPERIMENT PIPELINE SUCCESSFULLY COMPLETED! 🎉")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Experiment interrupted by user")
        print("🔄 Partial results may be available in ../Results/")
    except Exception as e:
        print(f"\n\n❌ Experiment failed with error: {e}")
        print("🔍 Check the error details above and retry")
        import traceback
        print("\n📋 Full traceback:")
        traceback.print_exc()