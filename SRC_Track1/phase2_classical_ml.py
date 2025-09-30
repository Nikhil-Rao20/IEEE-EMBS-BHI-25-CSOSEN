"""
🔬 Phase 2: Classical Machine Learning Models
===========================================

Implements proven machine learning algorithms for BDI-II depression score prediction.
These models provide improved prediction accuracy while maintaining interpretability.

Models included:
- Ensemble Methods (Random Forest, Extra Trees, AdaBoost, Gradient Boosting)
- Support Vector Regression (Linear, RBF, Polynomial kernels)
- Instance-Based Learning (KNN, Radius Neighbors)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, 
    AdaBoostRegressor, GradientBoostingRegressor
)
from sklearn.svm import SVR, NuSVR
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
import time
from typing import Dict, List, Tuple, Any
from scipy.stats import randint, uniform

class Phase2ClassicalML:
    """
    Phase 2: Classical Machine Learning Models Implementation
    
    Focus: Proven ML algorithms with strong predictive performance
    Target: Improve accuracy while maintaining model interpretability
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize Phase 2 models."""
        self.random_seed = random_seed
        self.models = {}
        self.results = {}
        self.scalers = {}
        
        print("🔧 Phase 2: Initializing Classical ML Models")
        print("=" * 45)
    
    def create_models(self) -> Dict[str, Any]:
        """Create all Phase 2 models with default parameters."""
        
        models = {
            # Ensemble Methods
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_seed,
                n_jobs=-1
            ),
            
            'extra_trees': ExtraTreesRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_seed,
                n_jobs=-1
            ),
            
            'ada_boost': AdaBoostRegressor(
                n_estimators=100,
                learning_rate=1.0,
                random_state=self.random_seed
            ),
            
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_seed
            ),
            
            # Support Vector Regression
            'svr_linear': Pipeline([
                ('scaler', StandardScaler()),
                ('svr', SVR(kernel='linear', C=1.0))
            ]),
            
            'svr_rbf': Pipeline([
                ('scaler', StandardScaler()),
                ('svr', SVR(kernel='rbf', C=1.0, gamma='scale'))
            ]),
            
            'svr_poly': Pipeline([
                ('scaler', StandardScaler()),
                ('svr', SVR(kernel='poly', C=1.0, degree=3, gamma='scale'))
            ]),
            
            'nu_svr': Pipeline([
                ('scaler', RobustScaler()),
                ('svr', NuSVR(nu=0.5, kernel='rbf', gamma='scale'))
            ]),
            
            # Instance-Based Learning
            'knn_regressor': Pipeline([
                ('scaler', StandardScaler()),
                ('knn', KNeighborsRegressor(n_neighbors=5, weights='distance'))
            ]),
            
            'knn_uniform': Pipeline([
                ('scaler', StandardScaler()),
                ('knn', KNeighborsRegressor(n_neighbors=7, weights='uniform'))
            ]),
            
            'radius_neighbors': Pipeline([
                ('scaler', StandardScaler()),
                ('radius', RadiusNeighborsRegressor(radius=1.0, weights='distance'))
            ])
        }
        
        self.models = models
        print(f"✅ Created {len(models)} classical ML models")
        return models
    
    def get_hyperparameter_grids(self) -> Dict[str, Dict]:
        """Define hyperparameter grids for optimization."""
        
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            
            'extra_trees': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            
            'ada_boost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.5, 1.0, 2.0]
            },
            
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            
            'svr_linear': {
                'svr__C': [0.1, 1.0, 10.0, 100.0],
                'svr__epsilon': [0.01, 0.1, 0.2]
            },
            
            'svr_rbf': {
                'svr__C': [0.1, 1.0, 10.0, 100.0],
                'svr__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
                'svr__epsilon': [0.01, 0.1, 0.2]
            },
            
            'svr_poly': {
                'svr__C': [0.1, 1.0, 10.0],
                'svr__degree': [2, 3, 4],
                'svr__gamma': ['scale', 'auto'],
                'svr__epsilon': [0.1, 0.2]
            },
            
            'nu_svr': {
                'svr__nu': [0.1, 0.3, 0.5, 0.7, 0.9],
                'svr__gamma': ['scale', 'auto', 0.01, 0.1]
            },
            
            'knn_regressor': {
                'knn__n_neighbors': [3, 5, 7, 9, 11, 15],
                'knn__weights': ['uniform', 'distance'],
                'knn__p': [1, 2]  # Manhattan vs Euclidean distance
            },
            
            'knn_uniform': {
                'knn__n_neighbors': [3, 5, 7, 9, 11],
                'knn__weights': ['uniform'],
                'knn__p': [1, 2]
            }
        }
        
        return param_grids
    
    def get_randomized_search_grids(self) -> Dict[str, Dict]:
        """Define parameter distributions for randomized search (for expensive models)."""
        
        param_distributions = {
            'random_forest': {
                'n_estimators': randint(50, 300),
                'max_depth': [3, 5, 7, 10, 15, None],
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'max_features': ['sqrt', 'log2', None]
            },
            
            'extra_trees': {
                'n_estimators': randint(50, 300),
                'max_depth': [3, 5, 7, 10, 15, None],
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'max_features': ['sqrt', 'log2', None]
            },
            
            'gradient_boosting': {
                'n_estimators': randint(50, 200),
                'learning_rate': uniform(0.01, 0.3),
                'max_depth': randint(3, 10),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10)
            }
        }
        
        return param_distributions
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, 
                                cv_folds: int = 5, use_randomized: bool = True) -> Dict[str, Any]:
        """
        Optimize hyperparameters using GridSearchCV or RandomizedSearchCV.
        
        Args:
            X: Feature matrix
            y: Target vector
            cv_folds: Number of cross-validation folds
            use_randomized: Whether to use randomized search for expensive models
            
        Returns:
            Dictionary with optimized models
        """
        print("\n🔍 Optimizing Hyperparameters...")
        print("-" * 35)
        
        param_grids = self.get_hyperparameter_grids()
        param_distributions = self.get_randomized_search_grids()
        optimized_models = {}
        
        # Models that benefit from randomized search (computationally expensive)
        expensive_models = ['random_forest', 'extra_trees', 'gradient_boosting']
        
        for model_name, model in self.models.items():
            print(f"  🎯 Optimizing {model_name}...")
            
            start_time = time.time()
            
            if model_name in param_grids:
                try:
                    if use_randomized and model_name in expensive_models and model_name in param_distributions:
                        # Use RandomizedSearchCV for expensive models
                        search = RandomizedSearchCV(
                            estimator=model,
                            param_distributions=param_distributions[model_name],
                            n_iter=50,  # Number of parameter settings sampled
                            cv=cv_folds,
                            scoring='neg_mean_absolute_error',
                            n_jobs=-1,
                            random_state=self.random_seed,
                            verbose=0
                        )
                    else:
                        # Use GridSearchCV for less expensive models
                        search = GridSearchCV(
                            estimator=model,
                            param_grid=param_grids[model_name],
                            cv=cv_folds,
                            scoring='neg_mean_absolute_error',
                            n_jobs=-1,
                            verbose=0
                        )
                    
                    search.fit(X, y)
                    optimized_models[model_name] = search.best_estimator_
                    
                    optimization_time = time.time() - start_time
                    print(f"    ✅ Best score: {-search.best_score_:.3f}")
                    print(f"    ✅ Best params: {search.best_params_}")
                    print(f"    ⏱️ Time: {optimization_time:.2f}s")
                    
                except Exception as e:
                    print(f"    ❌ Optimization failed: {e}")
                    # Use default model if optimization fails
                    model.fit(X, y)
                    optimized_models[model_name] = model
                    
            else:
                # No hyperparameter optimization needed
                model.fit(X, y)
                optimized_models[model_name] = model
                print(f"    ✅ Using default parameters")
        
        print(f"\n🎉 Hyperparameter optimization completed!")
        return optimized_models
    
    def evaluate_all_models(self, X: pd.DataFrame, y: pd.Series, 
                          framework, cv_strategy: str = 'kfold') -> Dict[str, Any]:
        """
        Evaluate all Phase 2 models.
        
        Args:
            X: Feature matrix
            y: Target vector
            framework: ExperimentFramework instance
            cv_strategy: Cross-validation strategy
            
        Returns:
            Dictionary with all evaluation results
        """
        print("\n📊 Evaluating Phase 2 Models...")
        print("=" * 35)
        
        # Create and optimize models
        self.create_models()
        optimized_models = self.optimize_hyperparameters(X, y)
        
        results = {}
        
        for model_name, model in optimized_models.items():
            print(f"\n🔍 Evaluating {model_name}...")
            
            # Evaluate model
            model_results = framework.evaluate_model(
                model=model,
                X=X,
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
        Extract feature importance from tree-based models.
        
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
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importance_scores = model.feature_importances_
                importance_dict = dict(zip(feature_names, importance_scores))
                
            elif isinstance(model, Pipeline):
                # Check if the final step has feature importances
                final_estimator = model.steps[-1][1]
                if hasattr(final_estimator, 'feature_importances_'):
                    importance_scores = final_estimator.feature_importances_
                    importance_dict = dict(zip(feature_names, importance_scores))
                elif hasattr(final_estimator, 'coef_'):
                    # For SVR models
                    coefficients = final_estimator.coef_
                    if len(coefficients.shape) > 1:
                        coefficients = coefficients[0]
                    importance_scores = np.abs(coefficients)
                    importance_dict = dict(zip(feature_names, importance_scores))
        
        except Exception as e:
            print(f"⚠️ Could not extract feature importance for {model_name}: {e}")
        
        return importance_dict
    
    def generate_phase_summary(self) -> str:
        """Generate Phase 2 summary report."""
        if not self.results:
            return "No results available for Phase 2."
        
        summary = []
        summary.append("📋 PHASE 2: CLASSICAL ML MODELS SUMMARY")
        summary.append("=" * 42)
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
        
        # Model type analysis
        ensemble_models = [name for name in self.results.keys() if any(ens in name for ens in ['forest', 'trees', 'boost'])]
        svm_models = [name for name in self.results.keys() if 'svr' in name]
        knn_models = [name for name in self.results.keys() if 'knn' in name or 'neighbors' in name]
        
        if ensemble_models:
            best_ensemble = min(ensemble_models, key=lambda x: self.results[x]['mean_scores']['test_mae'])
            summary.append(f"  • Best ensemble model: {best_ensemble}")
        
        if svm_models:
            best_svm = min(svm_models, key=lambda x: self.results[x]['mean_scores']['test_mae'])
            summary.append(f"  • Best SVM model: {best_svm}")
        
        if knn_models:
            best_knn = min(knn_models, key=lambda x: self.results[x]['mean_scores']['test_mae'])
            summary.append(f"  • Best KNN model: {best_knn}")
        
        return "\n".join(summary)

if __name__ == "__main__":
    # Example usage
    phase2 = Phase2ClassicalML()
    print("🚀 Phase 2 Classical ML Models Ready!")