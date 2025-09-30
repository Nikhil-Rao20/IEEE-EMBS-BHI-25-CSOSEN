"""
🔬 Phase 1: Baseline Models - Linear and Statistical Models
=========================================================

Implements interpretable baseline models for BDI-II depression score prediction.
These models serve as the foundation for comparison with more complex approaches.

Models included:
- Linear Regression variants (Ridge, Lasso, Elastic Net)
- Polynomial Regression
- Bayesian Linear Regression
- Generalized Linear Models
- Robust Regression
- Decision Trees
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, 
    HuberRegressor, RANSACRegressor, BayesianRidge
)
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
import time
from typing import Dict, List, Tuple, Any

class Phase1BaselineModels:
    """
    Phase 1: Baseline Models Implementation
    
    Focus: Interpretable models that provide baseline performance
    Target: Establish minimum performance threshold and feature understanding
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize Phase 1 models."""
        self.random_seed = random_seed
        self.models = {}
        self.results = {}
        self.scalers = {}
        
        print("🔧 Phase 1: Initializing Baseline Models")
        print("=" * 45)
    
    def create_models(self) -> Dict[str, Any]:
        """Create all Phase 1 models with default parameters."""
        
        models = {
            # Linear Models
            'linear_regression': LinearRegression(),
            
            'ridge_regression': Ridge(
                alpha=1.0,
                random_state=self.random_seed
            ),
            
            'lasso_regression': Lasso(
                alpha=1.0,
                random_state=self.random_seed,
                max_iter=2000
            ),
            
            'elastic_net': ElasticNet(
                alpha=1.0,
                l1_ratio=0.5,
                random_state=self.random_seed,
                max_iter=2000
            ),
            
            'bayesian_ridge': BayesianRidge(
                compute_score=True
            ),
            
            # Polynomial Regression
            'polynomial_2': Pipeline([
                ('scaler', StandardScaler()),
                ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                ('linear', LinearRegression())
            ]),
            
            'polynomial_3': Pipeline([
                ('scaler', StandardScaler()),
                ('poly', PolynomialFeatures(degree=3, include_bias=False)),
                ('linear', Ridge(alpha=1.0, random_state=self.random_seed))
            ]),
            
            # Robust Regression
            'huber_regressor': HuberRegressor(
                epsilon=1.35,
                max_iter=200
            ),
            
            'ransac_regressor': RANSACRegressor(
                estimator=LinearRegression(),
                random_state=self.random_seed,
                max_trials=200
            ),
            
            # Decision Tree (Simple)
            'decision_tree': DecisionTreeRegressor(
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=self.random_seed
            )
        }
        
        self.models = models
        print(f"✅ Created {len(models)} baseline models")
        return models
    
    def get_hyperparameter_grids(self) -> Dict[str, Dict]:
        """Define hyperparameter grids for optimization."""
        
        param_grids = {
            'ridge_regression': {
                'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
            },
            
            'lasso_regression': {
                'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
            },
            
            'elastic_net': {
                'alpha': [0.01, 0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            },
            
            'huber_regressor': {
                'epsilon': [1.1, 1.35, 1.5, 2.0]
            },
            
            'decision_tree': {
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [5, 10, 20],
                'min_samples_leaf': [2, 5, 10]
            }
        }
        
        return param_grids
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, 
                                cv_folds: int = 5) -> Dict[str, Any]:
        """
        Optimize hyperparameters using GridSearchCV.
        
        Args:
            X: Feature matrix
            y: Target vector
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with optimized models
        """
        print("\n🔍 Optimizing Hyperparameters...")
        print("-" * 35)
        
        param_grids = self.get_hyperparameter_grids()
        optimized_models = {}
        
        # Prepare data with scaling for linear models
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        self.scalers['standard'] = scaler
        
        for model_name, model in self.models.items():
            print(f"  🎯 Optimizing {model_name}...")
            
            start_time = time.time()
            
            if model_name in param_grids:
                # Use scaled data for linear models
                if any(linear_type in model_name.lower() for linear_type in 
                      ['ridge', 'lasso', 'elastic', 'linear', 'bayesian']):
                    X_input = X_scaled_df
                else:
                    X_input = X
                
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grids[model_name],
                    cv=cv_folds,
                    scoring='neg_mean_absolute_error',
                    n_jobs=-1
                )
                
                grid_search.fit(X_input, y)
                optimized_models[model_name] = grid_search.best_estimator_
                
                optimization_time = time.time() - start_time
                print(f"    ✅ Best params: {grid_search.best_params_}")
                print(f"    ⏱️ Time: {optimization_time:.2f}s")
                
            else:
                # No hyperparameter optimization needed
                if any(linear_type in model_name.lower() for linear_type in 
                      ['ridge', 'lasso', 'elastic', 'linear', 'bayesian']):
                    model.fit(X_scaled_df, y)
                else:
                    model.fit(X, y)
                optimized_models[model_name] = model
                print(f"    ✅ Using default parameters")
        
        print(f"\n🎉 Hyperparameter optimization completed!")
        return optimized_models
    
    def evaluate_all_models(self, X: pd.DataFrame, y: pd.Series, 
                          framework, cv_strategy: str = 'kfold') -> Dict[str, Any]:
        """
        Evaluate all Phase 1 models.
        
        Args:
            X: Feature matrix
            y: Target vector
            framework: ExperimentFramework instance
            cv_strategy: Cross-validation strategy
            
        Returns:
            Dictionary with all evaluation results
        """
        print("\n📊 Evaluating Phase 1 Models...")
        print("=" * 35)
        
        # Create and optimize models
        self.create_models()
        optimized_models = self.optimize_hyperparameters(X, y)
        
        results = {}
        
        for model_name, model in optimized_models.items():
            print(f"\n🔍 Evaluating {model_name}...")
            
            # Prepare input data (scaled for linear models)
            if any(linear_type in model_name.lower() for linear_type in 
                  ['ridge', 'lasso', 'elastic', 'linear', 'bayesian']):
                X_input = pd.DataFrame(
                    self.scalers['standard'].transform(X),
                    columns=X.columns, 
                    index=X.index
                )
            else:
                X_input = X
            
            # Evaluate model
            model_results = framework.evaluate_model(
                model=model,
                X=X_input,
                y=y,
                cv_strategy=cv_strategy
            )
            
            results[model_name] = model_results
            
            # Print key metrics
            mae = model_results['mean_scores']['test_mae']
            rmse = model_results['mean_scores']['test_rmse']
            r2 = model_results['mean_scores']['test_r2']
            
            print(f"  📈 MAE: {mae:.3f} ± {model_results['std_scores']['test_mae']:.3f}")
            print(f"  📈 RMSE: {rmse:.3f} ± {model_results['std_scores']['test_rmse']:.3f}")
            print(f"  📈 R²: {r2:.3f} ± {model_results['std_scores']['test_r2']:.3f}")
        
        self.results = results
        return results
    
    def get_feature_importance(self, model_name: str, feature_names: List[str]) -> Dict[str, float]:
        """
        Extract feature importance from linear models.
        
        Args:
            model_name: Name of the model
            feature_names: List of feature names
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if model_name not in self.models:
            return {}
        
        model = self.models[model_name]
        importance_dict = {}
        
        try:
            if hasattr(model, 'coef_'):
                # Linear models
                coefficients = model.coef_
                if len(coefficients.shape) > 1:
                    coefficients = coefficients[0]  # Multi-output case
                
                # Use absolute values for importance
                importance_scores = np.abs(coefficients)
                importance_dict = dict(zip(feature_names, importance_scores))
                
            elif hasattr(model, 'feature_importances_'):
                # Tree-based models
                importance_scores = model.feature_importances_
                importance_dict = dict(zip(feature_names, importance_scores))
                
            elif isinstance(model, Pipeline):
                # Pipeline models
                final_estimator = model.steps[-1][1]
                if hasattr(final_estimator, 'coef_'):
                    coefficients = final_estimator.coef_
                    if len(coefficients.shape) > 1:
                        coefficients = coefficients[0]
                    
                    # For polynomial features, we need to get original feature names
                    if 'poly' in [step[0] for step in model.steps]:
                        # This is simplified - in practice, you'd want to map back to original features
                        poly_features = model.named_steps['poly'].get_feature_names_out(feature_names)
                        importance_scores = np.abs(coefficients)
                        importance_dict = dict(zip(poly_features, importance_scores))
                    else:
                        importance_scores = np.abs(coefficients)
                        importance_dict = dict(zip(feature_names, importance_scores))
        
        except Exception as e:
            print(f"⚠️ Could not extract feature importance for {model_name}: {e}")
        
        return importance_dict
    
    def generate_phase_summary(self) -> str:
        """Generate Phase 1 summary report."""
        if not self.results:
            return "No results available for Phase 1."
        
        summary = []
        summary.append("📋 PHASE 1: BASELINE MODELS SUMMARY")
        summary.append("=" * 40)
        summary.append("")
        
        # Model performance summary
        summary.append("🏆 MODEL PERFORMANCE RANKING (by MAE):")
        summary.append("-" * 35)
        
        # Sort models by MAE
        model_scores = [(name, results['mean_scores']['test_mae']) 
                       for name, results in self.results.items()]
        model_scores.sort(key=lambda x: x[1])
        
        for i, (model_name, mae) in enumerate(model_scores):
            r2 = self.results[model_name]['mean_scores']['test_r2']
            summary.append(f"  {i+1:2d}. {model_name:20s} - MAE: {mae:.3f}, R²: {r2:.3f}")
        
        summary.append("")
        summary.append("💡 KEY INSIGHTS:")
        summary.append("-" * 15)
        
        best_model = model_scores[0][0]
        best_mae = model_scores[0][1]
        worst_mae = model_scores[-1][1]
        
        summary.append(f"  • Best performing model: {best_model}")
        summary.append(f"  • Best MAE achieved: {best_mae:.3f}")
        summary.append(f"  • Performance range: {best_mae:.3f} - {worst_mae:.3f}")
        summary.append(f"  • Total models evaluated: {len(self.results)}")
        
        return "\n".join(summary)

if __name__ == "__main__":
    # Example usage
    phase1 = Phase1BaselineModels()
    print("🚀 Phase 1 Baseline Models Ready!")