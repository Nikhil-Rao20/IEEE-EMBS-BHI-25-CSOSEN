"""
🔬 BDI-II Depression Score Prediction - Comprehensive Model Experiment Framework
=============================================================================

This module provides the complete experimental framework for systematic model comparison
from basic to advanced models for conference paper submission.

Author: Research Team
Date: September 2025
Purpose: IEEE EMBS BHI 2025 Conference Submission
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
import json
import pickle
from datetime import datetime
import time
from typing import Dict, List, Tuple, Any, Optional

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Visualization setup
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class ExperimentFramework:
    """
    Comprehensive framework for systematic model experimentation and evaluation.
    
    Features:
    - Automated model training and evaluation
    - Statistical significance testing
    - Cross-validation with multiple strategies
    - Hyperparameter optimization
    - Results tracking and visualization
    - Conference-ready reporting
    """
    
    def __init__(self, random_seed: int = 42, output_folder_name='Results'):
        """Initialize the experiment framework."""
        self.random_seed = random_seed
        self.results = {}
        self.models = {}
        self.predictions = {}
        self.feature_importance = {}
        self.training_times = {}
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set random seeds for reproducibility
        np.random.seed(random_seed)
        
        # Create output directories
        self.output_dir = Path(f"../{output_folder_name}/Model_Experiments")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔬 Experiment Framework Initialized")
        print(f"📁 Results will be saved to: {self.output_dir}")
        print(f"🆔 Experiment ID: {self.experiment_id}")
        print(f"🎲 Random Seed: {random_seed}")
    
    def load_data(self, train_path: str, test_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training and test datasets.
        
        Args:
            train_path: Path to training dataset with engineered features
            test_path: Optional path to test dataset
            
        Returns:
            Tuple of (train_df, test_df)
        """
        print("\n📊 Loading Datasets...")
        
        # Load training data
        train_df = pd.read_excel(train_path)
        print(f"✅ Training data loaded: {train_df.shape}")
        
        # Load test data if provided
        test_df = None
        if test_path and Path(test_path).exists():
            test_df = pd.read_excel(test_path)
            print(f"✅ Test data loaded: {test_df.shape}")
        else:
            print("ℹ️  No separate test data provided - will use train/validation split")
        
        return train_df, test_df
    
    def prepare_data(self, df: pd.DataFrame, target_columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare features and targets from dataset.
        
        Args:
            df: Input dataframe
            target_columns: List of target column names
            
        Returns:
            Tuple of (X, y) - features and targets
        """
        # Separate features and targets
        feature_columns = [col for col in df.columns if col not in target_columns]
        
        X = df[feature_columns].copy()
        y = df[target_columns].copy()
        
        print(f"📋 Features prepared: {X.shape[1]} features, {X.shape[0]} samples")
        print(f"🎯 Targets prepared: {y.shape[1]} targets, {y.shape[0]} samples")
        
        return X, y
    
    def get_cv_strategy(self, strategy: str = 'kfold', n_splits: int = 5):
        """Get cross-validation strategy."""
        from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
        
        if strategy == 'kfold':
            return KFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed)
        elif strategy == 'stratified':
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed)
        elif strategy == 'timeseries':
            return TimeSeriesSplit(n_splits=n_splits)
        else:
            raise ValueError(f"Unknown CV strategy: {strategy}")
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive regression metrics."""
        from sklearn.metrics import (
            mean_absolute_error, mean_squared_error, r2_score,
            explained_variance_score, max_error
        )
        
        # Define MAPE manually for compatibility
        def mean_absolute_percentage_error(y_true, y_pred):
            return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
        
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'explained_variance': explained_variance_score(y_true, y_pred),
            'max_error': max_error(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred),
            'median_ae': np.median(np.abs(y_true - y_pred))
        }
        
        # Clinical accuracy (±3 points tolerance)
        clinical_accuracy = np.mean(np.abs(y_true - y_pred) <= 3.0) * 100
        metrics['clinical_accuracy'] = clinical_accuracy
        
        return metrics
    
    def evaluate_model(self, model, X: pd.DataFrame, y: pd.Series, 
                      cv_strategy: str = 'kfold', n_splits: int = 5) -> Dict[str, Any]:
        """
        Comprehensive model evaluation with cross-validation.
        
        Args:
            model: Scikit-learn compatible model
            X: Feature matrix
            y: Target vector
            cv_strategy: Cross-validation strategy
            n_splits: Number of CV folds
            
        Returns:
            Dictionary with evaluation results
        """
        from sklearn.model_selection import cross_val_score, cross_validate
        from sklearn.metrics import make_scorer, mean_absolute_error
        
        print(f"📊 Evaluating {model.__class__.__name__}...")
        
        # Get CV strategy
        cv = self.get_cv_strategy(cv_strategy, n_splits)
        
        # Define custom scorers
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        def mape_scorer(y_true, y_pred):
            return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
        
        scorers = {
            'mae': make_scorer(mean_absolute_error, greater_is_better=False),
            'rmse': make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False),
            'r2': 'r2',
            'mape': make_scorer(mape_scorer, greater_is_better=False)
        }
        
        # Perform cross-validation
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scorers, 
                                  return_train_score=True, return_estimator=True)
        
        # Calculate statistics
        results = {
            'cv_scores': {},
            'mean_scores': {},
            'std_scores': {},
            'models': cv_results['estimator']
        }
        
        for metric in scorers.keys():
            test_scores = -cv_results[f'test_{metric}'] if metric in ['mae', 'rmse', 'mape'] else cv_results[f'test_{metric}']
            train_scores = -cv_results[f'train_{metric}'] if metric in ['mae', 'rmse', 'mape'] else cv_results[f'train_{metric}']
            
            results['cv_scores'][f'test_{metric}'] = test_scores
            results['cv_scores'][f'train_{metric}'] = train_scores
            results['mean_scores'][f'test_{metric}'] = np.mean(test_scores)
            results['mean_scores'][f'train_{metric}'] = np.mean(train_scores)
            results['std_scores'][f'test_{metric}'] = np.std(test_scores)
            results['std_scores'][f'train_{metric}'] = np.std(train_scores)
        
        return results
    
    def evaluate_model_with_hyperparams(self, model, X: pd.DataFrame, y: pd.Series, 
                                       cv_strategy: str = 'kfold', n_splits: int = 5,
                                       hyperparameter_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Enhanced model evaluation that includes hyperparameter information.
        
        Args:
            model: Scikit-learn compatible model
            X: Feature matrix
            y: Target vector
            cv_strategy: Cross-validation strategy
            n_splits: Number of CV folds
            hyperparameter_info: Dictionary containing hyperparameter search results
            
        Returns:
            Dictionary with evaluation results including hyperparameter data
        """
        # Get base evaluation results
        results = self.evaluate_model(model, X, y, cv_strategy, n_splits)
        
        # Add hyperparameter information if provided
        if hyperparameter_info:
            results['hyperparameter_info'] = hyperparameter_info
            
            # Extract model parameters
            if hasattr(model, 'get_params'):
                results['model_parameters'] = model.get_params()
            
            # Add search metadata
            if 'search_type' in hyperparameter_info:
                results['optimization_method'] = hyperparameter_info['search_type']
        
        return results
    
    def evaluate_keras_model(self, model_func, X: pd.DataFrame, y: pd.Series, 
                           cv_strategy: str = 'kfold', n_splits: int = 5,
                           epochs: int = 100, batch_size: int = 32, patience: int = 10) -> Dict[str, Any]:
        """
        Specialized evaluation for Keras models with training history capture.
        
        Args:
            model_func: Function that creates a fresh Keras model
            X: Feature matrix
            y: Target vector
            cv_strategy: Cross-validation strategy
            n_splits: Number of CV folds
            epochs: Training epochs
            batch_size: Batch size for training
            patience: Early stopping patience
            
        Returns:
            Dictionary with evaluation results including training history
        """
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        print(f"📊 Evaluating Keras model with training history capture...")
        
        # Get CV strategy
        cv = self.get_cv_strategy(cv_strategy, n_splits)
        
        fold_results = []
        training_histories = []
        trained_models = []
        
        # Manual cross-validation to capture training history
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            print(f"   Training fold {fold_idx + 1}/{n_splits}...")
            
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_val_scaled = scaler.transform(X_val_fold)
            
            # Create fresh model
            model = model_func(X_train_scaled.shape[1])
            
            # Callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss', patience=patience, restore_best_weights=True, verbose=0
            )
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=patience//2, min_lr=1e-6, verbose=0
            )
            
            # Train model
            history = model.fit(
                X_train_scaled, y_train_fold,
                validation_data=(X_val_scaled, y_val_fold),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            # Evaluate
            y_pred_train = model.predict(X_train_scaled, verbose=0).flatten()
            y_pred_val = model.predict(X_val_scaled, verbose=0).flatten()
            
            # Calculate metrics
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            fold_metrics = {
                'train_mae': mean_absolute_error(y_train_fold, y_pred_train),
                'val_mae': mean_absolute_error(y_val_fold, y_pred_val),
                'train_rmse': np.sqrt(mean_squared_error(y_train_fold, y_pred_train)),
                'val_rmse': np.sqrt(mean_squared_error(y_val_fold, y_pred_val)),
                'train_r2': r2_score(y_train_fold, y_pred_train),
                'val_r2': r2_score(y_val_fold, y_pred_val),
                'epochs_trained': len(history.history['loss'])
            }
            
            fold_results.append(fold_metrics)
            training_histories.append(history.history)
            trained_models.append(model)
        
        # Aggregate results
        results = {
            'cv_scores': {},
            'mean_scores': {},
            'std_scores': {},
            'models': trained_models,
            'training_histories': training_histories
        }
        
        # Calculate cross-validation statistics
        for metric in ['mae', 'rmse', 'r2']:
            train_scores = [fold[f'train_{metric}'] for fold in fold_results]
            val_scores = [fold[f'val_{metric}'] for fold in fold_results]
            
            results['cv_scores'][f'train_{metric}'] = train_scores
            results['cv_scores'][f'test_{metric}'] = val_scores
            results['mean_scores'][f'train_{metric}'] = np.mean(train_scores)
            results['mean_scores'][f'test_{metric}'] = np.mean(val_scores)
            results['std_scores'][f'train_{metric}'] = np.std(train_scores)
            results['std_scores'][f'test_{metric}'] = np.std(val_scores)
        
        # Add epoch information
        epochs_trained = [fold['epochs_trained'] for fold in fold_results]
        results['mean_epochs_trained'] = np.mean(epochs_trained)
        results['epochs_per_fold'] = epochs_trained
        
        print(f"   ✅ Keras evaluation complete. Average epochs: {results['mean_epochs_trained']:.1f}")
        
        return results
    
    def save_results(self, phase_name: str, results: Dict[str, Any]):
        """Save phase results to disk."""
        output_file = self.output_dir / f"{phase_name}_results_{self.experiment_id}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for model_name, model_results in results.items():
            serializable_results[model_name] = {}
            for key, value in model_results.items():
                if isinstance(value, np.ndarray):
                    serializable_results[model_name][key] = value.tolist()
                elif isinstance(value, dict):
                    serializable_results[model_name][key] = {
                        k: v.tolist() if isinstance(v, np.ndarray) else v 
                        for k, v in value.items()
                    }
                else:
                    serializable_results[model_name][key] = value
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"💾 Results saved: {output_file}")
    
    def plot_model_comparison(self, results: Dict[str, Any], metric: str = 'mae', 
                            save_plot: bool = True) -> None:
        """Create model comparison visualization."""
        model_names = list(results.keys())
        scores = [results[model]['mean_scores'][f'test_{metric}'] for model in model_names]
        errors = [results[model]['std_scores'][f'test_{metric}'] for model in model_names]
        
        plt.figure(figsize=(14, 8))
        bars = plt.bar(range(len(model_names)), scores, yerr=errors, 
                      capsize=5, alpha=0.7, color=sns.color_palette("husl", len(model_names)))
        
        plt.xlabel('Models')
        plt.ylabel(f'{metric.upper()} Score')
        plt.title(f'Model Comparison - {metric.upper()} with Error Bars')
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_plot:
            plot_file = self.output_dir / f"model_comparison_{metric}_{self.experiment_id}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved: {plot_file}")
        
        plt.show()
    
    def generate_summary_report(self, all_results: Dict[str, Dict]) -> str:
        """Generate comprehensive summary report."""
        report = []
        report.append("🏆 MODEL EXPERIMENT SUMMARY REPORT")
        report.append("=" * 50)
        report.append(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"🆔 Experiment ID: {self.experiment_id}")
        report.append("")
        
        # Best models by metric
        metrics = ['mae', 'rmse', 'r2', 'clinical_accuracy']
        
        for metric in metrics:
            report.append(f"🥇 BEST MODELS BY {metric.upper()}:")
            report.append("-" * 30)
            
            # Collect all model results
            all_model_results = []
            for phase, phase_results in all_results.items():
                for model_name, model_results in phase_results.items():
                    score = model_results['mean_scores'][f'test_{metric}']
                    all_model_results.append((f"{phase}_{model_name}", score))
            
            # Sort by metric (lower is better for mae, rmse, mape; higher is better for r2, clinical_accuracy)
            reverse = metric in ['r2', 'clinical_accuracy']
            all_model_results.sort(key=lambda x: x[1], reverse=reverse)
            
            # Top 5 models
            for i, (model_name, score) in enumerate(all_model_results[:5]):
                report.append(f"  {i+1}. {model_name}: {score:.4f}")
            report.append("")
        
        # Phase summary
        report.append("📊 PHASE SUMMARY:")
        report.append("-" * 20)
        for phase, phase_results in all_results.items():
            report.append(f"  {phase}: {len(phase_results)} models evaluated")
        
        report_text = "\n".join(report)
        
        # Save report
        report_file = self.output_dir / f"experiment_summary_{self.experiment_id}.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print(f"📋 Summary report saved: {report_file}")
        return report_text

if __name__ == "__main__":
    # Example usage
    framework = ExperimentFramework()
    print("🚀 Experiment Framework Ready!")