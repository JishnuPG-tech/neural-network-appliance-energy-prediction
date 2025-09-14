"""
Machine Learning Model Module for Appliance Energy Prediction

This module contains classes and functions for building, training,
and evaluating machine learning models for appliance energy consumption prediction.
Supports both traditional ML models and neural networks.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Any, Optional, Union
import joblib
import logging
import os

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ElectricityPredictor:
    """
    A comprehensive class for electricity consumption prediction.
    
    This class provides methods for model training, evaluation,
    and prediction for electricity consumption data.
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the ElectricityPredictor.
        
        Args:
            model_type (str): Type of model to use
        """
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.model_params = {}
        
        # Initialize model based on type
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the machine learning model."""
        model_map = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(random_state=42),
            'lasso': Lasso(random_state=42),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                random_state=42
            ),
            'svr': SVR(kernel='rbf')
        }
        
        if self.model_type not in model_map:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        self.model = model_map[self.model_type]
        logger.info(f"Initialized {self.model_type} model")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              feature_names: Optional[list] = None) -> Dict[str, float]:
        """
        Train the model on the provided data.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target values
            feature_names (list, optional): Names of features
            
        Returns:
            Dict[str, float]: Training metrics
        """
        logger.info(f"Training {self.model_type} model...")
        
        # Store feature names
        if feature_names:
            self.feature_names = feature_names
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate training metrics
        y_pred_train = self.model.predict(X_train)
        metrics = self._calculate_metrics(y_train, y_pred_train)
        
        logger.info(f"Model training completed. R² Score: {metrics['r2']:.4f}")
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X (np.ndarray): Features for prediction
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test target values
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        logger.info("Evaluating model...")
        
        y_pred = self.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred)
        
        logger.info(f"Model evaluation completed. R² Score: {metrics['r2']:.4f}")
        return metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            
        Returns:
            Dict[str, float]: Metrics dictionary
        """
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance for tree-based models.
        
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names or [f'feature_{i}' for i in range(len(self.model.feature_importances_))],
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            logger.warning(f"{self.model_type} does not support feature importance")
            return None
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray,
                            param_grid: Dict[str, Any], cv: int = 5) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target values
            param_grid (Dict[str, Any]): Parameter grid for tuning
            cv (int): Number of cross-validation folds
            
        Returns:
            Dict[str, Any]: Best parameters and score
        """
        logger.info("Starting hyperparameter tuning...")
        
        grid_search = GridSearchCV(
            self.model, param_grid, cv=cv, 
            scoring='neg_mean_squared_error', n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.model_params = grid_search.best_params_
        self.is_trained = True
        
        logger.info(f"Hyperparameter tuning completed. Best score: {-grid_search.best_score_:.4f}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': -grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Target values
            cv (int): Number of cross-validation folds
            
        Returns:
            Dict[str, float]: Cross-validation results
        """
        logger.info("Performing cross-validation...")
        
        cv_scores = cross_val_score(
            self.model, X, y, cv=cv, 
            scoring='neg_mean_squared_error', n_jobs=-1
        )
        
        rmse_scores = np.sqrt(-cv_scores)
        
        results = {
            'mean_rmse': rmse_scores.mean(),
            'std_rmse': rmse_scores.std(),
            'min_rmse': rmse_scores.min(),
            'max_rmse': rmse_scores.max()
        }
        
        logger.info(f"Cross-validation completed. Mean RMSE: {results['mean_rmse']:.4f} ± {results['std_rmse']:.4f}")
        return results
    
    def save_model(self, file_path: str):
        """
        Save the trained model to disk.
        
        Args:
            file_path (str): Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'model_params': self.model_params
        }
        
        joblib.dump(model_data, file_path)
        logger.info(f"Model saved to {file_path}")
    
    @classmethod
    def load_model(cls, file_path: str) -> 'ElectricityPredictor':
        """
        Load a trained model from disk.
        
        Args:
            file_path (str): Path to the saved model
            
        Returns:
            ElectricityPredictor: Loaded model instance
        """
        model_data = joblib.load(file_path)
        
        predictor = cls(model_data['model_type'])
        predictor.model = model_data['model']
        predictor.feature_names = model_data['feature_names']
        predictor.model_params = model_data['model_params']
        predictor.is_trained = True
        
        logger.info(f"Model loaded from {file_path}")
        return predictor


class ApplianceEnergyPredictor:
    """
    Neural network-based predictor for appliance energy consumption.
    
    This class provides methods for building, training, and evaluating
    neural networks for appliance energy prediction.
    """
    
    def __init__(self, input_dim: int = 50, hidden_layers: list = None, 
                 learning_rate: float = 0.001, dropout_rate: float = 0.2):
        """
        Initialize the ApplianceEnergyPredictor.
        
        Args:
            input_dim (int): Number of input features
            hidden_layers (list): List of hidden layer sizes
            learning_rate (float): Learning rate for optimizer
            dropout_rate (float): Dropout rate for regularization
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for neural network models")
        
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers or [256, 128, 64, 32]
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = None
        self.training_history = None
        
        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
    
    def build_model(self):
        """Build the neural network architecture."""
        model = Sequential()
        
        # Input layer
        model.add(Dense(self.hidden_layers[0], 
                       input_dim=self.input_dim, 
                       activation='relu',
                       name='input_layer'))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # Hidden layers
        for i, units in enumerate(self.hidden_layers[1:], 1):
            model.add(Dense(units, activation='relu', name=f'hidden_{i}'))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(1, activation='linear', name='output'))
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, 
                     loss='mse',
                     metrics=['mae', 'mse'])
        
        self.model = model
        logger.info(f"Neural network model built with architecture: {self.hidden_layers}")
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 100, batch_size: int = 32,
              feature_names: Optional[list] = None) -> Dict[str, Any]:
        """
        Train the neural network model.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target values
            X_val (np.ndarray, optional): Validation features
            y_val (np.ndarray, optional): Validation target values
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            feature_names (list, optional): Names of features
            
        Returns:
            Dict[str, Any]: Training history and metrics
        """
        logger.info("Training neural network model...")
        
        # Store feature names
        if feature_names:
            self.feature_names = feature_names
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            validation_data = (X_val_scaled, y_val)
        else:
            validation_data = None
        
        # Build model if not already built
        if self.model is None:
            self.input_dim = X_train_scaled.shape[1]
            self.build_model()
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.training_history = history.history
        self.is_trained = True
        
        # Calculate training metrics
        y_pred_train = self.predict(X_train)
        metrics = self._calculate_metrics(y_train, y_pred_train)
        
        logger.info(f"Neural network training completed. R² Score: {metrics['r2']:.4f}")
        return {
            'history': self.training_history,
            'metrics': metrics
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X (np.ndarray): Features for prediction
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled, verbose=0)
        return predictions.flatten()
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test target values
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        logger.info("Evaluating neural network model...")
        
        y_pred = self.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred)
        
        logger.info(f"Neural network evaluation completed. R² Score: {metrics['r2']:.4f}")
        return metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100
        }
    
    def save_model(self, model_path: str, scaler_path: str = None, 
                   feature_names_path: str = None, metadata_path: str = None):
        """
        Save the trained model and associated components.
        
        Args:
            model_path (str): Path to save the model
            scaler_path (str, optional): Path to save the scaler
            feature_names_path (str, optional): Path to save feature names
            metadata_path (str, optional): Path to save metadata
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Save model
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save scaler
        if scaler_path:
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Scaler saved to {scaler_path}")
        
        # Save feature names
        if feature_names_path and self.feature_names:
            joblib.dump(self.feature_names, feature_names_path)
            logger.info(f"Feature names saved to {feature_names_path}")
        
        # Save metadata
        if metadata_path:
            metadata = {
                'input_dim': self.input_dim,
                'hidden_layers': self.hidden_layers,
                'learning_rate': self.learning_rate,
                'dropout_rate': self.dropout_rate,
                'feature_count': len(self.feature_names) if self.feature_names else None
            }
            joblib.dump(metadata, metadata_path)
            logger.info(f"Metadata saved to {metadata_path}")
    
    @classmethod
    def load_model(cls, model_path: str, scaler_path: str = None,
                   feature_names_path: str = None, metadata_path: str = None) -> 'ApplianceEnergyPredictor':
        """
        Load a trained model and associated components.
        
        Args:
            model_path (str): Path to the saved model
            scaler_path (str, optional): Path to the saved scaler
            feature_names_path (str, optional): Path to the saved feature names
            metadata_path (str, optional): Path to the saved metadata
            
        Returns:
            ApplianceEnergyPredictor: Loaded model instance
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for neural network models")
        
        # Load metadata first if available
        if metadata_path and os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            predictor = cls(
                input_dim=metadata.get('input_dim', 50),
                hidden_layers=metadata.get('hidden_layers', [256, 128, 64, 32]),
                learning_rate=metadata.get('learning_rate', 0.001),
                dropout_rate=metadata.get('dropout_rate', 0.2)
            )
        else:
            predictor = cls()
        
        # Load model
        predictor.model = tf.keras.models.load_model(model_path)
        predictor.is_trained = True
        logger.info(f"Model loaded from {model_path}")
        
        # Load scaler
        if scaler_path and os.path.exists(scaler_path):
            predictor.scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded from {scaler_path}")
        
        # Load feature names
        if feature_names_path and os.path.exists(feature_names_path):
            predictor.feature_names = joblib.load(feature_names_path)
            logger.info(f"Feature names loaded from {feature_names_path}")
        
        return predictor


class ModelComparison:
    """
    A class for comparing multiple machine learning models including neural networks.
    """
    
    def __init__(self):
        """Initialize the ModelComparison."""
        self.models = {}
        self.results = {}
    
    def add_model(self, name: str, model_type: str, **kwargs):
        """
        Add a model for comparison.
        
        Args:
            name (str): Name for the model
            model_type (str): Type of the model ('neural_network' for NN, or sklearn model types)
            **kwargs: Additional arguments for neural network models
        """
        if model_type == 'neural_network':
            if TENSORFLOW_AVAILABLE:
                self.models[name] = ApplianceEnergyPredictor(**kwargs)
            else:
                logger.warning(f"TensorFlow not available. Skipping neural network model: {name}")
        else:
            self.models[name] = ElectricityPredictor(model_type)
    
    def compare_models(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray,
                      X_val: np.ndarray = None, y_val: np.ndarray = None,
                      feature_names: Optional[list] = None, **nn_kwargs) -> pd.DataFrame:
        """
        Train and compare all models.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target values
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test target values
            X_val (np.ndarray, optional): Validation features for neural networks
            y_val (np.ndarray, optional): Validation target values for neural networks
            feature_names (list, optional): Names of features
            **nn_kwargs: Additional arguments for neural network training
            
        Returns:
            pd.DataFrame: Comparison results
        """
        logger.info("Starting model comparison...")
        
        comparison_results = []
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            try:
                # Check if it's a neural network model
                if isinstance(model, ApplianceEnergyPredictor):
                    # Train neural network with validation data
                    train_result = model.train(
                        X_train, y_train, X_val, y_val, 
                        feature_names=feature_names, **nn_kwargs
                    )
                    train_metrics = train_result['metrics']
                else:
                    # Train traditional ML model
                    train_metrics = model.train(X_train, y_train, feature_names)
                
                # Evaluate model
                test_metrics = model.evaluate(X_test, y_test)
                
                # Store results
                result = {
                    'Model': name,
                    'Train_RMSE': train_metrics['rmse'],
                    'Test_RMSE': test_metrics['rmse'],
                    'Train_R2': train_metrics['r2'],
                    'Test_R2': test_metrics['r2'],
                    'Test_MAE': test_metrics['mae'],
                    'Test_MAPE': test_metrics['mape']
                }
                
                comparison_results.append(result)
                self.results[name] = {'model': model, 'metrics': test_metrics}
                
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                # Add failed result
                result = {
                    'Model': name,
                    'Train_RMSE': np.nan,
                    'Test_RMSE': np.nan,
                    'Train_R2': np.nan,
                    'Test_R2': np.nan,
                    'Test_MAE': np.nan,
                    'Test_MAPE': np.nan
                }
                comparison_results.append(result)
        
        results_df = pd.DataFrame(comparison_results)
        results_df = results_df.sort_values('Test_R2', ascending=False)
        
        logger.info("Model comparison completed")
        return results_df
    
    def get_best_model(self) -> Tuple[str, Union[ElectricityPredictor, ApplianceEnergyPredictor]]:
        """
        Get the best performing model.
        
        Returns:
            Tuple[str, ElectricityPredictor]: Name and instance of best model
        """
        if not self.results:
            raise ValueError("No models have been compared yet")
        
        best_name = max(self.results.keys(), 
                       key=lambda x: self.results[x]['metrics']['r2'])
        best_model = self.results[best_name]['model']
        
        return best_name, best_model


def create_appliance_model(input_dim: int = 50, architecture: str = 'standard') -> ApplianceEnergyPredictor:
    """
    Create a pre-configured appliance energy prediction model.
    
    Args:
        input_dim (int): Number of input features
        architecture (str): Model architecture ('standard', 'deep', 'wide')
        
    Returns:
        ApplianceEnergyPredictor: Configured model instance
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required for neural network models")
    
    architectures = {
        'standard': [256, 128, 64, 32],
        'deep': [512, 256, 128, 64, 32, 16],
        'wide': [512, 512, 256, 256, 128],
        'compact': [128, 64, 32]
    }
    
    if architecture not in architectures:
        logger.warning(f"Unknown architecture '{architecture}'. Using 'standard'.")
        architecture = 'standard'
    
    hidden_layers = architectures[architecture]
    
    model = ApplianceEnergyPredictor(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        learning_rate=0.001,
        dropout_rate=0.2
    )
    
    logger.info(f"Created appliance model with {architecture} architecture: {hidden_layers}")
    return model


def load_appliance_model_from_directory(model_dir: str) -> ApplianceEnergyPredictor:
    """
    Load an appliance model from a directory containing all components.
    
    Args:
        model_dir (str): Directory containing model files
        
    Returns:
        ApplianceEnergyPredictor: Loaded model instance
    """
    model_path = os.path.join(model_dir, 'appliance_energy_model.h5')
    scaler_path = os.path.join(model_dir, 'feature_scaler.pkl')
    feature_names_path = os.path.join(model_dir, 'feature_names.pkl')
    metadata_path = os.path.join(model_dir, 'model_metadata.pkl')
    
    return ApplianceEnergyPredictor.load_model(
        model_path=model_path,
        scaler_path=scaler_path,
        feature_names_path=feature_names_path,
        metadata_path=metadata_path
    )


def get_default_param_grids() -> Dict[str, Dict[str, Any]]:
    """
    Get default parameter grids for hyperparameter tuning.
    
    Returns:
        Dict[str, Dict[str, Any]]: Parameter grids for different models
    """
    param_grids = {
        'random_forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'gradient_boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        },
        'ridge': {
            'alpha': [0.1, 1.0, 10.0, 100.0]
        },
        'lasso': {
            'alpha': [0.01, 0.1, 1.0, 10.0]
        },
        'svr': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1],
            'kernel': ['rbf', 'linear', 'poly']
        }
    }
    
    return param_grids
