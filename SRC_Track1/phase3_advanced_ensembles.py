"""
🔬 Phase 3: Advanced Ensemble Models
==================================

Implements state-of-the-art ensemble methods for BDI-II depression score prediction.
These models represent the current best practices in machine learning.

Models included:
- Modern Boosting (XGBoost, LightGBM, CatBoost)
- Stacking Ensembles (Meta-learner approaches)
- Voting Ensembles (Combination strategies)
- Blending Techniques
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import time
from typing import Dict, List, Tuple, Any, Optional
from scipy.stats import randint, uniform

# Try to import advanced boosting libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = False  # Disabled for speed
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM disabled for faster execution")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost not available. Install with: pip install catboost")

# Fallback to sklearn ensemble methods if advanced libraries not available
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor

class Phase3AdvancedEnsembles:
    """
    Phase 3: Advanced Ensemble Models Implementation
    
    Focus: State-of-the-art ensemble methods for maximum predictive performance
    Target: Achieve best possible accuracy using modern techniques
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize Phase 3 models."""
        self.random_seed = random_seed
        self.models = {}
        self.results = {}
        self.base_models = {}
        
        print("🔧 Phase 3: Initializing Advanced Ensemble Models")
        print("=" * 50)
        
        # Check availability of advanced libraries
        print(f"📊 XGBoost available: {XGBOOST_AVAILABLE}")
        print(f"📊 LightGBM available: {LIGHTGBM_AVAILABLE}")
        print(f"📊 CatBoost available: {CATBOOST_AVAILABLE}")
    
    def create_base_models(self) -> Dict[str, Any]:
        """Create base models for ensemble methods."""
        
        base_models = {
            'rf_base': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=self.random_seed,
                n_jobs=-1
            ),
            
            'et_base': ExtraTreesRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=self.random_seed,
                n_jobs=-1
            ),
            
            'gb_base': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.random_seed
            )
        }
        
        self.base_models = base_models
        return base_models
    
    def create_models(self) -> Dict[str, Any]:
        """Create all Phase 3 models with default parameters."""
        
        models = {}
        
        # Create base models first
        self.create_base_models()
        
        # Modern Boosting Models
        if XGBOOST_AVAILABLE:
            models['xgboost'] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_seed,
                n_jobs=-1,
                verbosity=0
            )
        
        # LightGBM - disabled for speed optimization
        # if LIGHTGBM_AVAILABLE:
        #     models['lightgbm'] = lgb.LGBMRegressor(
        #         n_estimators=100,
        #         max_depth=6,
        #         learning_rate=0.1,
        #         subsample=0.8,
        #         colsample_bytree=0.8,
        #         random_state=self.random_seed,
        #         n_jobs=-1,
        #         verbosity=-1
        #     )
        
        if CATBOOST_AVAILABLE:
            models['catboost'] = cb.CatBoostRegressor(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_state=self.random_seed,
                verbose=False,
                thread_count=-1
            )
        
        # Voting Ensembles
        voting_estimators = [
            ('rf', self.base_models['rf_base']),
            ('et', self.base_models['et_base']),
            ('gb', self.base_models['gb_base'])
        ]
        
        models['voting_regressor'] = VotingRegressor(
            estimators=voting_estimators,
            n_jobs=-1
        )
        
        # Stacking Ensembles
        models['stacking_regressor'] = StackingRegressor(
            estimators=voting_estimators,
            final_estimator=RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]),
            cv=5,
            n_jobs=-1
        )
        
        # Advanced Stacking with more diverse base models
        if XGBOOST_AVAILABLE:
            advanced_estimators = voting_estimators + [('xgb', models['xgboost'])]
            
            models['advanced_stacking'] = StackingRegressor(
                estimators=advanced_estimators,
                final_estimator=RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]),
                cv=5,
                n_jobs=-1
            )
        
        # Blended Ensemble (Custom implementation)
        models['blended_ensemble'] = BlendedEnsemble(
            base_models=list(self.base_models.values()),
            random_state=self.random_seed
        )
        
        self.models = models
        print(f"✅ Created {len(models)} advanced ensemble models")
        return models
    
    def get_hyperparameter_grids(self) -> Dict[str, Dict]:
        """Define hyperparameter grids for optimization."""
        
        param_grids = {}
        
        # XGBoost parameters
        if XGBOOST_AVAILABLE:
            param_grids['xgboost'] = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 1],
                'reg_lambda': [1, 1.5, 2]
            }
        
        # LightGBM parameters (disabled for speed)
        # if LIGHTGBM_AVAILABLE:
        #     param_grids['lightgbm'] = {
        #         'n_estimators': [100, 200, 300],
        #         'max_depth': [3, 6, 9],
        #         'learning_rate': [0.01, 0.1, 0.2],
        #         'subsample': [0.8, 0.9, 1.0],
        #         'colsample_bytree': [0.8, 0.9, 1.0],
        #         'reg_alpha': [0, 0.1, 1],
        #         'reg_lambda': [1, 1.5, 2],
        #         'num_leaves': [31, 50, 100]
        #     }
        
        # CatBoost parameters
        if CATBOOST_AVAILABLE:
            param_grids['catboost'] = {
                'iterations': [100, 200, 300],
                'depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'l2_leaf_reg': [1, 3, 5],
                'border_count': [32, 128, 255]
            }
        
        return param_grids
    
    def get_randomized_search_grids(self) -> Dict[str, Dict]:
        """Define parameter distributions for randomized search."""
        
        param_distributions = {}
        
        # XGBoost distributions
        if XGBOOST_AVAILABLE:
            param_distributions['xgboost'] = {
                'n_estimators': randint(50, 500),
                'max_depth': randint(3, 15),
                'learning_rate': uniform(0.01, 0.3),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4),
                'reg_alpha': uniform(0, 2),
                'reg_lambda': uniform(0, 2)
            }
        
        # LightGBM distributions (disabled for speed)
        # if LIGHTGBM_AVAILABLE:
        #     param_distributions['lightgbm'] = {
        #         'n_estimators': randint(50, 500),
        #         'max_depth': randint(3, 15),
        #         'learning_rate': uniform(0.01, 0.3),
        #         'subsample': uniform(0.6, 0.4),
        #         'colsample_bytree': uniform(0.6, 0.4),
        #         'reg_alpha': uniform(0, 2),
        #         'reg_lambda': uniform(0, 2),
        #         'num_leaves': randint(20, 200)
        #     }
        
        # CatBoost distributions
        if CATBOOST_AVAILABLE:
            param_distributions['catboost'] = {
                'iterations': randint(50, 500),
                'depth': randint(3, 10),
                'learning_rate': uniform(0.01, 0.3),
                'l2_leaf_reg': uniform(1, 10),
                'border_count': [32, 128, 255]
            }
        
        return param_distributions
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, 
                                cv_folds: int = 5, use_randomized: bool = True,
                                n_iter: int = 50) -> Dict[str, Any]:
        """
        Optimize hyperparameters for advanced models.
        
        Args:
            X: Feature matrix
            y: Target vector
            cv_folds: Number of cross-validation folds
            use_randomized: Whether to use randomized search
            n_iter: Number of iterations for randomized search
            
        Returns:
            Dictionary with optimized models
        """
        print("\n🔍 Optimizing Advanced Model Hyperparameters...")
        print("-" * 45)
        
        param_grids = self.get_hyperparameter_grids()
        param_distributions = self.get_randomized_search_grids()
        optimized_models = {}
        
        # Models that should use randomized search
        expensive_models = ['xgboost', 'catboost']  # Removed lightgbm for speed
        
        for model_name, model in self.models.items():
            print(f"  🎯 Optimizing {model_name}...")
            
            start_time = time.time()
            
            try:
                if model_name in param_grids and model_name in expensive_models:
                    if use_randomized and model_name in param_distributions:
                        # Use RandomizedSearchCV for expensive models
                        search = RandomizedSearchCV(
                            estimator=model,
                            param_distributions=param_distributions[model_name],
                            n_iter=n_iter,
                            cv=cv_folds,
                            scoring='neg_mean_absolute_error',
                            n_jobs=-1,
                            random_state=self.random_seed,
                            verbose=0
                        )
                    else:
                        # Use GridSearchCV with reduced grid
                        reduced_grid = {}
                        for param, values in param_grids[model_name].items():
                            if len(values) > 3:
                                # Take every other value to reduce search space
                                reduced_grid[param] = values[::2]
                            else:
                                reduced_grid[param] = values
                        
                        search = GridSearchCV(
                            estimator=model,
                            param_grid=reduced_grid,
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
                    
                else:
                    # No hyperparameter optimization or ensemble methods
                    model.fit(X, y)
                    optimized_models[model_name] = model
                    
                    optimization_time = time.time() - start_time
                    print(f"    ✅ Using default parameters")
                    print(f"    ⏱️ Time: {optimization_time:.2f}s")
                    
            except Exception as e:
                print(f"    ❌ Optimization failed: {e}")
                # Use default model if optimization fails
                try:
                    model.fit(X, y)
                    optimized_models[model_name] = model
                except Exception as fit_error:
                    print(f"    ❌ Model fitting also failed: {fit_error}")
                    continue
        
        print(f"\n🎉 Advanced model optimization completed!")
        return optimized_models
    
    def evaluate_all_models(self, X: pd.DataFrame, y: pd.Series, 
                          framework, cv_strategy: str = 'kfold') -> Dict[str, Any]:
        """
        Evaluate all Phase 3 models.
        
        Args:
            X: Feature matrix
            y: Target vector
            framework: ExperimentFramework instance
            cv_strategy: Cross-validation strategy
            
        Returns:
            Dictionary with all evaluation results
        """
        print("\n📊 Evaluating Phase 3 Models...")
        print("=" * 35)
        
        # Create and optimize models
        self.create_models()
        optimized_models = self.optimize_hyperparameters(X, y)
        
        results = {}
        
        for model_name, model in optimized_models.items():
            print(f"\n🔍 Evaluating {model_name}...")
            
            try:
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
                
            except Exception as e:
                print(f"  ❌ Evaluation failed: {e}")
                continue
        
        self.results = results
        return results
    
    def get_feature_importance(self, model_name: str, feature_names: List[str]) -> Dict[str, float]:
        """
        Extract feature importance from ensemble models.
        
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
                # XGBoost, LightGBM, CatBoost, tree-based models
                importance_scores = model.feature_importances_
                importance_dict = dict(zip(feature_names, importance_scores))
                
            elif hasattr(model, 'estimators_'):
                # Voting or Stacking regressors
                if hasattr(model, 'final_estimator_'):
                    # Stacking regressor - use final estimator coefficients if available
                    if hasattr(model.final_estimator_, 'coef_'):
                        # Get base model predictions first (not directly accessible)
                        importance_dict = {'stacking_ensemble': 1.0}  # Placeholder
                else:
                    # Voting regressor - average feature importances from base models
                    all_importances = []
                    for estimator in model.estimators_:
                        if hasattr(estimator, 'feature_importances_'):
                            all_importances.append(estimator.feature_importances_)
                    
                    if all_importances:
                        avg_importance = np.mean(all_importances, axis=0)
                        importance_dict = dict(zip(feature_names, avg_importance))
        
        except Exception as e:
            print(f"⚠️ Could not extract feature importance for {model_name}: {e}")
        
        return importance_dict
    
    def generate_phase_summary(self) -> str:
        """Generate Phase 3 summary report."""
        if not self.results:
            return "No results available for Phase 3."
        
        summary = []
        summary.append("📋 PHASE 3: ADVANCED ENSEMBLE MODELS SUMMARY")
        summary.append("=" * 46)
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
        boosting_models = [name for name in self.results.keys() if any(boost in name for boost in ['xgboost', 'lightgbm', 'catboost'])]
        ensemble_models = [name for name in self.results.keys() if any(ens in name for ens in ['voting', 'stacking', 'blended'])]
        
        if boosting_models:
            best_boosting = min(boosting_models, key=lambda x: self.results[x]['mean_scores']['test_mae'])
            summary.append(f"  • Best boosting model: {best_boosting}")
        
        if ensemble_models:
            best_ensemble = min(ensemble_models, key=lambda x: self.results[x]['mean_scores']['test_mae'])
            summary.append(f"  • Best ensemble model: {best_ensemble}")
        
        return "\n".join(summary)


class BlendedEnsemble:
    """
    Custom blended ensemble implementation.
    
    Combines predictions from multiple base models using optimized weights.
    """
    
    def __init__(self, base_models: List[Any], random_state: int = 42):
        """Initialize blended ensemble."""
        self.base_models = base_models
        self.random_state = random_state
        self.weights = None
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit the blended ensemble."""
        from sklearn.model_selection import cross_val_predict
        from sklearn.linear_model import LinearRegression
        
        # Get cross-validated predictions from base models
        meta_features = []
        
        for model in self.base_models:
            # Get cross-validated predictions
            cv_preds = cross_val_predict(model, X, y, cv=5)
            meta_features.append(cv_preds)
        
        # Stack meta features
        meta_X = np.column_stack(meta_features)
        
        # Train meta-model (simple linear regression for blending weights)
        meta_model = LinearRegression()
        meta_model.fit(meta_X, y)
        
        # Store weights (coefficients)
        self.weights = meta_model.coef_
        
        # Fit all base models on full data
        for model in self.base_models:
            model.fit(X, y)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the blended ensemble."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions.")
        
        # Get predictions from all base models
        predictions = []
        for model in self.base_models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Weighted average of predictions
        predictions_array = np.column_stack(predictions)
        blended_pred = np.dot(predictions_array, self.weights)
        
        return blended_pred


if __name__ == "__main__":
    # Example usage
    phase3 = Phase3AdvancedEnsembles()
    print("🚀 Phase 3 Advanced Ensemble Models Ready!")