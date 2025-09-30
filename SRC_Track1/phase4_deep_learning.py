"""
🔬 Phase 4: Deep Learning Models
==============================

Implements neural network architectures for BDI-II depression score prediction.
These models can capture complex non-linear patterns in the data.

Models included:
- Multi-Layer Perceptron (MLP)
- Deep Neural Networks with regularization
- Residual Networks for tabular data
- Attention-based models
- Specialized tabular architectures
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, RegressorMixin
import time
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import deep learning libraries
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers, callbacks
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Add, MultiHeadAttention, LayerNormalization
    TENSORFLOW_AVAILABLE = True
    
    # Configure TensorFlow to reduce verbosity
    tf.get_logger().setLevel('ERROR')
    
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available. Install with: pip install tensorflow")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠️ PyTorch not available. Install with: pip install torch")

# Fallback to sklearn neural networks if deep learning libraries not available
from sklearn.neural_network import MLPRegressor

class Phase4DeepLearning:
    """
    Phase 4: Deep Learning Models Implementation
    
    Focus: Neural networks for capturing complex patterns
    Target: Explore deep learning capabilities for tabular depression data
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize Phase 4 models."""
        self.random_seed = random_seed
        self.models = {}
        self.results = {}
        self.scalers = {}
        self.history = {}
        
        # Set random seeds
        np.random.seed(random_seed)
        if TENSORFLOW_AVAILABLE:
            tf.random.set_seed(random_seed)
        if PYTORCH_AVAILABLE:
            torch.manual_seed(random_seed)
        
        print("🔧 Phase 4: Initializing Deep Learning Models")
        print("=" * 47)
        print(f"📊 TensorFlow available: {TENSORFLOW_AVAILABLE}")
        print(f"📊 PyTorch available: {PYTORCH_AVAILABLE}")
    
    def create_sklearn_models(self) -> Dict[str, Any]:
        """Create sklearn-based neural network models."""
        
        models = {
            'mlp_small': MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                solver='adam',
                alpha=0.01,
                batch_size='auto',
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=500,
                random_state=self.random_seed,
                early_stopping=True,
                validation_fraction=0.2
            ),
            
            'mlp_medium': MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.01,
                batch_size='auto',
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=500,
                random_state=self.random_seed,
                early_stopping=True,
                validation_fraction=0.2
            ),
            
            'mlp_large': MLPRegressor(
                hidden_layer_sizes=(256, 128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.01,
                batch_size='auto',
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=500,
                random_state=self.random_seed,
                early_stopping=True,
                validation_fraction=0.2
            )
        }
        
        return models
    
    def create_tensorflow_models(self, input_dim: int) -> Dict[str, Any]:
        """Create TensorFlow/Keras models."""
        if not TENSORFLOW_AVAILABLE:
            return {}
        
        models = {}
        
        # Model builder functions
        def build_mlp_simple(input_shape, **kwargs):
            model = Sequential([
                Dense(64, activation='relu', input_shape=(input_shape,)),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            return model
        
        def build_mlp_deep(input_shape, **kwargs):
            model = Sequential([
                Dense(128, activation='relu', input_shape=(input_shape,)),
                BatchNormalization(),
                Dropout(0.3),
                Dense(64, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                Dense(32, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            return model
        
        def build_resnet(input_shape, **kwargs):
            input_layer = Input(shape=(input_shape,))
            
            # First block
            x = Dense(128, activation='relu')(input_layer)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            
            # Residual block 1
            residual1 = x
            x = Dense(128, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            x = Dense(128, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Add()([x, residual1])  # Residual connection
            
            # Second block
            x = Dense(64, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            
            # Residual block 2
            residual2 = Dense(64)(x)  # Match dimensions
            y = Dense(64, activation='relu')(x)
            y = BatchNormalization()(y)
            y = Dropout(0.2)(y)
            y = Dense(64, activation='relu')(y)
            y = BatchNormalization()(y)
            y = Add()([y, residual2])  # Residual connection
            
            # Final layers
            y = Dense(32, activation='relu')(y)
            y = Dropout(0.2)(y)
            output = Dense(1)(y)
            
            model = Model(inputs=input_layer, outputs=output)
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            return model
        
        def build_attention(input_shape, **kwargs):
            attention_input = Input(shape=(input_shape,))
            
            # Reshape for attention (treat features as sequence)
            x = layers.Reshape((1, input_shape))(attention_input)  # Add sequence dimension
            x = Dense(64)(x)  # Feature embedding
            
            # Multi-head attention
            attention_output = MultiHeadAttention(
                num_heads=4, key_dim=16
            )(x, x)
            
            # Layer normalization and residual connection
            x = LayerNormalization()(attention_output + x)
            
            # Flatten and continue with dense layers
            x = layers.Flatten()(x)  # Remove sequence dimension
            x = Dense(64, activation='relu')(x)
            x = Dropout(0.3)(x)
            x = Dense(32, activation='relu')(x)
            x = Dropout(0.2)(x)
            output = Dense(1)(x)
            
            model = Model(inputs=attention_input, outputs=output)
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            return model
        
        # Create wrapped models with builder functions
        models['tf_mlp_simple'] = KerasRegressorWrapper(
            model_builder=build_mlp_simple, epochs=100, batch_size=32
        )
        models['tf_mlp_deep'] = KerasRegressorWrapper(
            model_builder=build_mlp_deep, epochs=150, batch_size=32
        )
        models['tf_resnet'] = KerasRegressorWrapper(
            model_builder=build_resnet, epochs=200, batch_size=32
        )
        models['tf_attention'] = KerasRegressorWrapper(
            model_builder=build_attention, epochs=150, batch_size=32
        )
        
        return models
    
    def create_models(self, input_dim: int) -> Dict[str, Any]:
        """Create all Phase 4 models."""
        
        models = {}
        
        # Add sklearn models
        sklearn_models = self.create_sklearn_models()
        models.update(sklearn_models)
        
        # Add TensorFlow models if available
        if TENSORFLOW_AVAILABLE:
            tf_models = self.create_tensorflow_models(input_dim)
            models.update(tf_models)
        
        self.models = models
        print(f"✅ Created {len(models)} deep learning models")
        return models
    
    def get_hyperparameter_grids(self) -> Dict[str, Dict]:
        """Define hyperparameter grids for optimization."""
        
        param_grids = {
            'mlp_small': {
                'hidden_layer_sizes': [(32, 16), (64, 32), (128, 64)],
                'alpha': [0.001, 0.01, 0.1],
                'learning_rate_init': [0.001, 0.01, 0.1]
            },
            
            'mlp_medium': {
                'hidden_layer_sizes': [(64, 32, 16), (128, 64, 32), (256, 128, 64)],
                'alpha': [0.001, 0.01, 0.1],
                'learning_rate_init': [0.001, 0.01]
            },
            
            'mlp_large': {
                'hidden_layer_sizes': [(128, 64, 32, 16), (256, 128, 64, 32)],
                'alpha': [0.001, 0.01],
                'learning_rate_init': [0.001, 0.01]
            }
        }
        
        return param_grids
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, 
                                cv_folds: int = 3) -> Dict[str, Any]:
        """
        Optimize hyperparameters for neural network models.
        
        Args:
            X: Feature matrix
            y: Target vector
            cv_folds: Number of cross-validation folds (reduced for neural networks)
            
        Returns:
            Dictionary with optimized models
        """
        print("\n🔍 Optimizing Deep Learning Hyperparameters...")
        print("-" * 45)
        
        from sklearn.model_selection import GridSearchCV
        
        param_grids = self.get_hyperparameter_grids()
        optimized_models = {}
        
        # Prepare scaled data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        self.scalers['standard'] = scaler
        
        for model_name, model in self.models.items():
            print(f"  🎯 Optimizing {model_name}...")
            
            start_time = time.time()
            
            try:
                if model_name in param_grids:
                    # Only optimize sklearn models due to computational cost
                    grid_search = GridSearchCV(
                        estimator=model,
                        param_grid=param_grids[model_name],
                        cv=cv_folds,
                        scoring='neg_mean_absolute_error',
                        n_jobs=1,  # Neural networks don't parallelize well
                        verbose=0
                    )
                    
                    grid_search.fit(X_scaled_df, y)
                    optimized_models[model_name] = grid_search.best_estimator_
                    
                    optimization_time = time.time() - start_time
                    print(f"    ✅ Best score: {-grid_search.best_score_:.3f}")
                    print(f"    ✅ Best params: {grid_search.best_params_}")
                    print(f"    ⏱️ Time: {optimization_time:.2f}s")
                    
                else:
                    # TensorFlow models - use default parameters
                    model.fit(X_scaled_df, y)
                    optimized_models[model_name] = model
                    
                    optimization_time = time.time() - start_time
                    print(f"    ✅ Using default parameters")
                    print(f"    ⏱️ Time: {optimization_time:.2f}s")
                    
            except Exception as e:
                print(f"    ❌ Optimization failed: {e}")
                # Try with default parameters
                try:
                    model.fit(X_scaled_df, y)
                    optimized_models[model_name] = model
                except Exception as fit_error:
                    print(f"    ❌ Model fitting also failed: {fit_error}")
                    continue
        
        print(f"\n🎉 Deep learning optimization completed!")
        return optimized_models
    
    def evaluate_all_models(self, X: pd.DataFrame, y: pd.Series, 
                          framework, cv_strategy: str = 'kfold') -> Dict[str, Any]:
        """
        Evaluate all Phase 4 models.
        
        Args:
            X: Feature matrix
            y: Target vector
            framework: ExperimentFramework instance
            cv_strategy: Cross-validation strategy
            
        Returns:
            Dictionary with all evaluation results
        """
        print("\n📊 Evaluating Phase 4 Models...")
        print("=" * 35)
        
        # Create models
        self.create_models(input_dim=X.shape[1])
        optimized_models = self.optimize_hyperparameters(X, y)
        
        results = {}
        
        # Prepare scaled data
        if 'standard' not in self.scalers:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            self.scalers['standard'] = scaler
        else:
            X_scaled_df = pd.DataFrame(
                self.scalers['standard'].transform(X),
                columns=X.columns, 
                index=X.index
            )
        
        for model_name, model in optimized_models.items():
            print(f"\n🔍 Evaluating {model_name}...")
            
            try:
                # Evaluate model with scaled data
                model_results = framework.evaluate_model(
                    model=model,
                    X=X_scaled_df,
                    y=y,
                    cv_strategy=cv_strategy,
                    n_splits=3  # Reduced folds for neural networks
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
    
    def generate_phase_summary(self) -> str:
        """Generate Phase 4 summary report."""
        if not self.results:
            return "No results available for Phase 4."
        
        summary = []
        summary.append("📋 PHASE 4: DEEP LEARNING MODELS SUMMARY")
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
        sklearn_models = [name for name in self.results.keys() if name.startswith('mlp')]
        tf_models = [name for name in self.results.keys() if name.startswith('tf_')]
        
        if sklearn_models:
            best_sklearn = min(sklearn_models, key=lambda x: self.results[x]['mean_scores']['test_mae'])
            summary.append(f"  • Best sklearn model: {best_sklearn}")
        
        if tf_models:
            best_tf = min(tf_models, key=lambda x: self.results[x]['mean_scores']['test_mae'])
            summary.append(f"  • Best TensorFlow model: {best_tf}")
        
        return "\n".join(summary)


class KerasRegressorWrapper(BaseEstimator, RegressorMixin):
    """
    Wrapper for Keras models to make them compatible with sklearn interface.
    """
    
    def __init__(self, model_builder=None, epochs=100, batch_size=32, validation_split=0.2, verbose=0, **model_params):
        """Initialize the wrapper with a model builder function."""
        self.model_builder = model_builder
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.verbose = verbose
        self.model_params = model_params
        self.model_ = None
        self.history = None
        self.is_fitted = False
    
    def _build_model(self, input_shape):
        """Build a new model instance."""
        if self.model_builder is None:
            raise ValueError("model_builder function must be provided")
        return self.model_builder(input_shape, **self.model_params)
    
    def fit(self, X, y):
        """Fit the Keras model."""
        # Build a fresh model for each fit
        input_shape = X.shape[1] if len(X.shape) == 2 else X.shape[1:]
        self.model_ = self._build_model(input_shape)
        
        # Add early stopping and reduce LR on plateau
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss', patience=20, restore_best_weights=True
        )
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6
        )
        
        self.history = self.model_.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=[early_stopping, reduce_lr],
            verbose=self.verbose
        )
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make predictions."""
        if not self.is_fitted or self.model_ is None:
            raise ValueError("Model must be fitted before making predictions.")
        
        predictions = self.model_.predict(X, verbose=0)
        return predictions.flatten()  # Ensure 1D output for regression
    
    def score(self, X, y):
        """Return R² score."""
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        params = {
            'model_builder': self.model_builder,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'validation_split': self.validation_split,
            'verbose': self.verbose
        }
        # Add model-specific parameters
        params.update(self.model_params)
        return params
    
    def set_params(self, **params):
        """Set the parameters of this estimator."""
        model_param_keys = set(self.model_params.keys())
        
        for key, value in params.items():
            if key in ['model_builder', 'epochs', 'batch_size', 'validation_split', 'verbose']:
                setattr(self, key, value)
            elif key in model_param_keys:
                self.model_params[key] = value
            else:
                # Add new model parameter
                self.model_params[key] = value
        
        # Reset fitted state when parameters change
        self.is_fitted = False
        self.model_ = None
        return self


if __name__ == "__main__":
    # Example usage
    phase4 = Phase4DeepLearning()
    print("🚀 Phase 4 Deep Learning Models Ready!")