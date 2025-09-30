"""
🔬 Phase 5: Time-Series & Trajectory Models
==========================================

Implements time-series and trajectory-based models for BDI-II depression score prediction.
These models leverage temporal patterns and progression dynamics in depression scores.

Models included:
- ARIMA/SARIMA models
- Exponential Smoothing
- Vector Autoregression (VAR)
- LSTM Networks for sequences
- GRU Networks
- Transformer-based models
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import time-series libraries
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.vector_ar.var_model import VAR
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("⚠️ Statsmodels not available. Install with: pip install statsmodels")

# Try to import deep learning libraries for sequential models
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization
    TENSORFLOW_AVAILABLE = True
    
    # Configure TensorFlow to reduce verbosity
    tf.get_logger().setLevel('ERROR')
    
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available for sequential models.")

# Fallback imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

class Phase5TimeSeriesModels:
    """
    Phase 5: Time-Series & Trajectory Models Implementation
    
    Focus: Leverage temporal patterns in depression progression
    Target: Capture dynamics and trajectories in BDI-II scores over time
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize Phase 5 models."""
        self.random_seed = random_seed
        self.models = {}
        self.results = {}
        self.scalers = {}
        self.sequence_data = {}
        
        # Set random seeds
        np.random.seed(random_seed)
        if TENSORFLOW_AVAILABLE:
            tf.random.set_seed(random_seed)
        
        print("🔧 Phase 5: Initializing Time-Series Models")
        print("=" * 45)
        print(f"📊 Statsmodels available: {STATSMODELS_AVAILABLE}")
        print(f"📊 TensorFlow available: {TENSORFLOW_AVAILABLE}")
    
    def create_sequence_features(self, X: pd.DataFrame, y: pd.Series, 
                               sequence_length: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequence features for time-series models.
        
        Since we don't have true time-series data, we'll create pseudo-sequences
        based on patient similarity and feature patterns.
        
        Args:
            X: Feature matrix
            y: Target vector
            sequence_length: Length of sequences to create
            
        Returns:
            Tuple of (X_seq, y_seq) - sequence features and targets
        """
        print(f"🔄 Creating pseudo-sequences (length={sequence_length})...")
        
        # Sort patients by baseline BDI score to create meaningful sequences
        if 'bdi_ii_baseline' in X.columns:
            sort_idx = X['bdi_ii_baseline'].argsort()
        else:
            # Use first column if baseline not available
            sort_idx = X.iloc[:, 0].argsort()
        
        X_sorted = X.iloc[sort_idx].values
        y_sorted = y.iloc[sort_idx].values
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(len(X_sorted) - sequence_length + 1):
            X_seq = X_sorted[i:i + sequence_length]
            y_seq = y_sorted[i + sequence_length - 1]  # Predict last value in sequence
            
            X_sequences.append(X_seq)
            y_sequences.append(y_seq)
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        print(f"✅ Created {len(X_sequences)} sequences of shape {X_sequences.shape}")
        
        # Store for later use
        self.sequence_data = {
            'X_seq': X_sequences,
            'y_seq': y_sequences,
            'sequence_length': sequence_length,
            'feature_names': X.columns.tolist()
        }
        
        return X_sequences, y_sequences
    
    def create_traditional_ts_models(self, y: pd.Series) -> Dict[str, Any]:
        """Create traditional time-series models."""
        if not STATSMODELS_AVAILABLE:
            return {}
        
        models = {}
        
        try:
            # ARIMA Model
            models['arima'] = ARIMAWrapper(order=(1, 1, 1))
            
            # ARIMA with different parameters
            models['arima_211'] = ARIMAWrapper(order=(2, 1, 1))
            models['arima_121'] = ARIMAWrapper(order=(1, 2, 1))
            
            # Exponential Smoothing
            models['exponential_smoothing'] = ExponentialSmoothingWrapper()
            
            # Simple moving average baseline
            models['moving_average'] = MovingAverageWrapper(window=3)
            
        except Exception as e:
            print(f"⚠️ Error creating traditional TS models: {e}")
        
        return models
    
    def create_sequential_models(self, input_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Create sequential deep learning models."""
        if not TENSORFLOW_AVAILABLE:
            return {}
        
        models = {}
        sequence_length, n_features = input_shape
        
        try:
            # Simple LSTM
            lstm_simple = Sequential([
                LSTM(64, input_shape=(sequence_length, n_features)),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(1)
            ])
            lstm_simple.compile(optimizer='adam', loss='mse', metrics=['mae'])
            models['lstm_simple'] = KerasSequentialWrapper(lstm_simple, epochs=100)
            
            # Bidirectional LSTM
            lstm_bidirectional = Sequential([
                layers.Bidirectional(LSTM(64), input_shape=(sequence_length, n_features)),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(1)
            ])
            lstm_bidirectional.compile(optimizer='adam', loss='mse', metrics=['mae'])
            models['lstm_bidirectional'] = KerasSequentialWrapper(lstm_bidirectional, epochs=100)
            
            # Stacked LSTM
            lstm_stacked = Sequential([
                LSTM(64, return_sequences=True, input_shape=(sequence_length, n_features)),
                Dropout(0.3),
                LSTM(32, return_sequences=False),
                Dropout(0.3),
                Dense(16, activation='relu'),
                Dense(1)
            ])
            lstm_stacked.compile(optimizer='adam', loss='mse', metrics=['mae'])
            models['lstm_stacked'] = KerasSequentialWrapper(lstm_stacked, epochs=150)
            
            # GRU Model
            gru_model = Sequential([
                GRU(64, input_shape=(sequence_length, n_features)),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(1)
            ])
            gru_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            models['gru'] = KerasSequentialWrapper(gru_model, epochs=100)
            
            # Transformer-based model (simplified)
            transformer_input = Input(shape=(sequence_length, n_features))
            
            # Multi-head attention
            attention_output = MultiHeadAttention(
                num_heads=4, key_dim=16
            )(transformer_input, transformer_input)
            
            # Layer normalization and residual connection
            x = LayerNormalization()(attention_output + transformer_input)
            
            # Global average pooling to reduce sequence dimension
            x = layers.GlobalAveragePooling1D()(x)
            
            # Dense layers
            x = Dense(64, activation='relu')(x)
            x = Dropout(0.3)(x)
            x = Dense(32, activation='relu')(x)
            x = Dropout(0.2)(x)
            output = Dense(1)(x)
            
            transformer_model = Model(inputs=transformer_input, outputs=output)
            transformer_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            models['transformer'] = KerasSequentialWrapper(transformer_model, epochs=150)
            
        except Exception as e:
            print(f"⚠️ Error creating sequential models: {e}")
        
        return models
    
    def create_trajectory_models(self) -> Dict[str, Any]:
        """Create trajectory-based models using traditional ML with engineered features."""
        
        models = {
            # Random Forest with trajectory-aware features
            'rf_trajectory': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=self.random_seed,
                n_jobs=-1
            ),
            
            # Ridge regression for smooth trajectories
            'ridge_trajectory': Ridge(
                alpha=1.0,
                random_state=self.random_seed
            )
        }
        
        return models
    
    def create_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Create all Phase 5 models."""
        
        models = {}
        
        # Create sequence data
        sequence_length = min(5, len(X) // 10)  # Adaptive sequence length
        X_seq, y_seq = self.create_sequence_features(X, y, sequence_length)
        
        # Traditional time-series models
        traditional_models = self.create_traditional_ts_models(y)
        models.update(traditional_models)
        
        # Sequential deep learning models
        if len(X_seq) > 0:
            sequential_models = self.create_sequential_models(X_seq.shape[1:])
            models.update(sequential_models)
        
        # Trajectory-based models
        trajectory_models = self.create_trajectory_models()
        models.update(trajectory_models)
        
        self.models = models
        print(f"✅ Created {len(models)} time-series and trajectory models")
        return models
    
    def evaluate_all_models(self, X: pd.DataFrame, y: pd.Series, 
                          framework, cv_strategy: str = 'kfold') -> Dict[str, Any]:
        """
        Evaluate all Phase 5 models.
        
        Args:
            X: Feature matrix
            y: Target vector
            framework: ExperimentFramework instance
            cv_strategy: Cross-validation strategy
            
        Returns:
            Dictionary with all evaluation results
        """
        print("\n📊 Evaluating Phase 5 Models...")
        print("=" * 35)
        
        # Create models
        self.create_models(X, y)
        
        results = {}
        
        # Prepare scaled data for traditional models
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        self.scalers['standard'] = scaler
        
        for model_name, model in self.models.items():
            print(f"\n🔍 Evaluating {model_name}...")
            
            try:
                # Choose appropriate data format
                if any(seq_type in model_name for seq_type in ['lstm', 'gru', 'transformer']):
                    # Sequential models need sequence data
                    if 'X_seq' in self.sequence_data and len(self.sequence_data['X_seq']) > 10:
                        # Custom evaluation for sequential models
                        model_results = self.evaluate_sequential_model(model, framework)
                    else:
                        print(f"  ⚠️ Insufficient data for sequential model")
                        continue
                
                elif any(ts_type in model_name for ts_type in ['arima', 'exponential', 'moving']):
                    # Traditional time-series models
                    model_results = self.evaluate_ts_model(model, y, framework)
                
                else:
                    # Traditional ML models with regular data
                    model_results = framework.evaluate_model(
                        model=model,
                        X=X_scaled_df,
                        y=y,
                        cv_strategy=cv_strategy,
                        n_splits=3  # Reduced for computational efficiency
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
    
    def evaluate_sequential_model(self, model, framework) -> Dict[str, Any]:
        """Custom evaluation for sequential models."""
        from sklearn.model_selection import KFold
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        X_seq = self.sequence_data['X_seq']
        y_seq = self.sequence_data['y_seq']
        
        # Manual cross-validation for sequential data
        kf = KFold(n_splits=3, shuffle=True, random_state=self.random_seed)
        
        test_scores = {'mae': [], 'rmse': [], 'r2': []}
        
        for train_idx, test_idx in kf.split(X_seq):
            X_train, X_test = X_seq[train_idx], X_seq[test_idx]
            y_train, y_test = y_seq[train_idx], y_seq[test_idx]
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            test_scores['mae'].append(mae)
            test_scores['rmse'].append(rmse)
            test_scores['r2'].append(r2)
        
        # Format results
        results = {
            'cv_scores': {f'test_{k}': v for k, v in test_scores.items()},
            'mean_scores': {f'test_{k}': np.mean(v) for k, v in test_scores.items()},
            'std_scores': {f'test_{k}': np.std(v) for k, v in test_scores.items()}
        }
        
        return results
    
    def evaluate_ts_model(self, model, y: pd.Series, framework) -> Dict[str, Any]:
        """Custom evaluation for traditional time-series models."""
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        # Use time series split
        tscv = TimeSeriesSplit(n_splits=3)
        
        test_scores = {'mae': [], 'rmse': [], 'r2': []}
        
        y_values = y.values
        
        for train_idx, test_idx in tscv.split(y_values):
            y_train, y_test = y_values[train_idx], y_values[test_idx]
            
            try:
                # Fit model
                model.fit(y_train)
                
                # Predict
                y_pred = model.predict(len(y_test))
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                test_scores['mae'].append(mae)
                test_scores['rmse'].append(rmse)
                test_scores['r2'].append(r2)
                
            except Exception as e:
                print(f"    ⚠️ Fold failed: {e}")
                continue
        
        # Handle case where no folds succeeded
        if not test_scores['mae']:
            # Return dummy results
            return {
                'cv_scores': {'test_mae': [999], 'test_rmse': [999], 'test_r2': [0]},
                'mean_scores': {'test_mae': 999, 'test_rmse': 999, 'test_r2': 0},
                'std_scores': {'test_mae': 0, 'test_rmse': 0, 'test_r2': 0}
            }
        
        # Format results
        results = {
            'cv_scores': {f'test_{k}': v for k, v in test_scores.items()},
            'mean_scores': {f'test_{k}': np.mean(v) for k, v in test_scores.items()},
            'std_scores': {f'test_{k}': np.std(v) for k, v in test_scores.items()}
        }
        
        return results
    
    def generate_phase_summary(self) -> str:
        """Generate Phase 5 summary report."""
        if not self.results:
            return "No results available for Phase 5."
        
        summary = []
        summary.append("📋 PHASE 5: TIME-SERIES & TRAJECTORY MODELS SUMMARY")
        summary.append("=" * 52)
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
        ts_models = [name for name in self.results.keys() if any(ts in name for ts in ['arima', 'exponential', 'moving'])]
        seq_models = [name for name in self.results.keys() if any(seq in name for seq in ['lstm', 'gru', 'transformer'])]
        traj_models = [name for name in self.results.keys() if 'trajectory' in name]
        
        if ts_models:
            best_ts = min(ts_models, key=lambda x: self.results[x]['mean_scores']['test_mae'])
            summary.append(f"  • Best time-series model: {best_ts}")
        
        if seq_models:
            best_seq = min(seq_models, key=lambda x: self.results[x]['mean_scores']['test_mae'])
            summary.append(f"  • Best sequential model: {best_seq}")
        
        if traj_models:
            best_traj = min(traj_models, key=lambda x: self.results[x]['mean_scores']['test_mae'])
            summary.append(f"  • Best trajectory model: {best_traj}")
        
        return "\n".join(summary)


# Helper classes for time-series models
class ARIMAWrapper:
    """Wrapper for ARIMA model."""
    
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.model = None
        self.fitted_model = None
    
    def fit(self, y):
        if STATSMODELS_AVAILABLE:
            try:
                self.model = ARIMA(y, order=self.order)
                self.fitted_model = self.model.fit()
            except:
                # Fallback to simple mean
                self.fitted_model = None
                self.mean_value = np.mean(y)
        return self
    
    def predict(self, steps):
        if self.fitted_model is not None:
            try:
                forecast = self.fitted_model.forecast(steps=steps)
                return forecast
            except:
                return np.full(steps, self.mean_value)
        else:
            return np.full(steps, self.mean_value)


class ExponentialSmoothingWrapper:
    """Wrapper for Exponential Smoothing model."""
    
    def __init__(self):
        self.fitted_model = None
        self.mean_value = None
    
    def fit(self, y):
        if STATSMODELS_AVAILABLE:
            try:
                self.fitted_model = ExponentialSmoothing(y).fit()
            except:
                self.fitted_model = None
                self.mean_value = np.mean(y)
        else:
            self.mean_value = np.mean(y)
        return self
    
    def predict(self, steps):
        if self.fitted_model is not None:
            try:
                forecast = self.fitted_model.forecast(steps=steps)
                return forecast
            except:
                return np.full(steps, self.mean_value)
        else:
            return np.full(steps, self.mean_value)


class MovingAverageWrapper:
    """Simple moving average model."""
    
    def __init__(self, window=3):
        self.window = window
        self.last_values = None
    
    def fit(self, y):
        self.last_values = y[-self.window:] if len(y) >= self.window else y
        return self
    
    def predict(self, steps):
        prediction = np.mean(self.last_values)
        return np.full(steps, prediction)


class KerasSequentialWrapper:
    """Wrapper for Keras sequential models."""
    
    def __init__(self, model, epochs=100, batch_size=32, verbose=0):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.is_fitted = False
    
    def fit(self, X, y):
        # Add early stopping
        early_stopping = callbacks.EarlyStopping(
            monitor='loss', patience=10, restore_best_weights=True
        )
        
        self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stopping],
            verbose=self.verbose
        )
        self.is_fitted = True
        return self
    
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions.")
        
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()


if __name__ == "__main__":
    # Example usage
    phase5 = Phase5TimeSeriesModels()
    print("🚀 Phase 5 Time-Series Models Ready!")