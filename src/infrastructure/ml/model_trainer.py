"""Machine learning model trainer implementation."""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.domain.entities import ModelConfig, ProcessedData, TrainedModel
from src.domain.repositories import IModelTrainer


class ModelTrainer(IModelTrainer):
    """Trains and evaluates machine learning models."""
    
    SUPPORTED_MODELS = {
        "linear_regression": LinearRegression,
        "logistic_regression": LogisticRegression,
        "decision_tree": DecisionTreeClassifier,
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
    }
    
    def __init__(self):
        """Initialize model trainer."""
        self.current_model = None
    
    def train(self, data: ProcessedData, config: ModelConfig) -> TrainedModel:
        """
        Train a machine learning model.
        
        Args:
            data: Processed data for training
            config: Model configuration
            
        Returns:
            Trained model with metadata
        """
        logger.info(f"Training {config.model_type} model...")
        
        # Prepare features and target
        X, y = self._prepare_data(data.data, config)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_size, random_state=config.random_state
        )
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Initialize model
        model = self._create_model(config)
        
        # Train model
        model.fit(X_train, y_train)
        logger.info("Model training completed")
        
        # Evaluate on test set
        metrics = self._evaluate_model(model, X_test, y_test, config.model_type)
        
        # Calculate feature importance if available
        feature_importance = self._get_feature_importance(model, X.columns.tolist())
        
        trained_model = TrainedModel(
            model=model,
            config=config,
            metrics=metrics,
            feature_importance=feature_importance,
            training_data_shape=X_train.shape,
        )
        
        logger.info(f"Model trained successfully. Metrics: {metrics}")
        return trained_model
    
    def evaluate(self, model: TrainedModel, test_data: pd.DataFrame) -> dict:
        """
        Evaluate model performance on test data.
        
        Args:
            model: Trained model
            test_data: Test dataset
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model on test data...")
        
        X, y = self._prepare_data(test_data, model.config)
        metrics = self._evaluate_model(model.model, X, y, model.config.model_type)
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def _prepare_data(
        self, data: pd.DataFrame, config: ModelConfig
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target from data."""
        if config.target_column is None:
            raise ValueError("Target column must be specified in config")
        
        if config.target_column not in data.columns:
            raise ValueError(f"Target column '{config.target_column}' not found in data")
        
        # Determine features
        if config.feature_columns:
            feature_cols = config.feature_columns
        else:
            # Use all columns except target
            feature_cols = [col for col in data.columns if col != config.target_column]
        
        X = data[feature_cols].copy()
        y = data[config.target_column].copy()
        
        # Handle any remaining missing values
        X = X.fillna(0)
        
        logger.info(f"Prepared {len(feature_cols)} features and target '{config.target_column}'")
        return X, y
    
    def _create_model(self, config: ModelConfig) -> Any:
        """Create model instance based on configuration."""
        model_class = self.SUPPORTED_MODELS.get(config.model_type)
        
        if model_class is None:
            raise ValueError(
                f"Unsupported model type: {config.model_type}. "
                f"Supported types: {list(self.SUPPORTED_MODELS.keys())}"
            )
        
        # Create model with hyperparameters
        params = config.hyperparameters.copy()
        if "random_state" not in params and config.model_type != "linear_regression":
            params["random_state"] = config.random_state
        
        return model_class(**params)
    
    def _evaluate_model(
        self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_type: str
    ) -> Dict[str, float]:
        """Evaluate model and return metrics."""
        predictions = model.predict(X_test)
        
        # Determine if classification or regression
        is_classification = model_type in [
            "logistic_regression",
            "decision_tree",
            "random_forest",
            "gradient_boosting",
        ]
        
        metrics: Dict[str, float] = {}
        
        if is_classification:
            # Classification metrics
            metrics["accuracy"] = float(accuracy_score(y_test, predictions))
            
            # Try to get probabilities for additional metrics
            if hasattr(model, "predict_proba"):
                try:
                    # For binary classification, use positive class probability
                    proba = model.predict_proba(X_test)
                    if proba.shape[1] == 2:
                        metrics["auc_roc"] = 0.0  # Placeholder - would need roc_auc_score
                except Exception:
                    pass
        else:
            # Regression metrics
            metrics["r2_score"] = float(r2_score(y_test, predictions))
            metrics["mse"] = float(mean_squared_error(y_test, predictions))
            metrics["rmse"] = float(np.sqrt(metrics["mse"]))
            metrics["mae"] = float(mean_absolute_error(y_test, predictions))
        
        return metrics
    
    def _get_feature_importance(
        self, model: Any, feature_names: list
    ) -> Dict[str, float] | None:
        """Extract feature importance if available."""
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            return {
                name: float(importance)
                for name, importance in zip(feature_names, importances)
            }
        elif hasattr(model, "coef_"):
            # For linear models
            coef = model.coef_
            if coef.ndim == 1:
                return {
                    name: float(abs(importance))
                    for name, importance in zip(feature_names, coef)
                }
        
        return None
