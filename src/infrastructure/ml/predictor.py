"""Prediction service implementation."""

from typing import Any

import pandas as pd
from loguru import logger

from src.domain.entities import Prediction, TrainedModel
from src.domain.repositories import IPredictor


class Predictor(IPredictor):
    """Makes predictions using trained models."""
    
    def predict(self, model: TrainedModel, data: pd.DataFrame) -> Prediction:
        """
        Make predictions using the trained model.
        
        Args:
            model: Trained model
            data: Input data for predictions
            
        Returns:
            Prediction object with results
        """
        logger.info(f"Making predictions with {model.config.model_type} model...")
        
        # Prepare features
        X = self._prepare_features(data, model)
        
        # Make predictions
        predictions = model.model.predict(X)
        predictions_series = pd.Series(predictions, index=data.index, name="prediction")
        
        # Get probabilities if available (for classification)
        probabilities = None
        confidence_scores = None
        
        if hasattr(model.model, "predict_proba"):
            try:
                proba = model.model.predict_proba(X)
                probabilities = pd.DataFrame(
                    proba,
                    index=data.index,
                    columns=[f"class_{i}" for i in range(proba.shape[1])],
                )
                
                # Confidence is the max probability for each prediction
                confidence_scores = pd.Series(
                    proba.max(axis=1), index=data.index, name="confidence"
                )
                
                logger.info("Generated prediction probabilities")
            except Exception as e:
                logger.warning(f"Could not generate probabilities: {e}")
        
        prediction = Prediction(
            predictions=predictions_series,
            probabilities=probabilities,
            model_used=model.config.model_type,
            confidence_scores=confidence_scores,
            metadata={
                "n_samples": len(data),
                "n_features": X.shape[1],
            },
        )
        
        logger.info(f"Generated {len(predictions_series)} predictions")
        return prediction
    
    def _prepare_features(self, data: pd.DataFrame, model: TrainedModel) -> pd.DataFrame:
        """Prepare features matching the training data."""
        # Use the same features as during training
        if model.config.feature_columns:
            feature_cols = model.config.feature_columns
        else:
            # Get all columns except target
            feature_cols = [
                col for col in data.columns if col != model.config.target_column
            ]
        
        # Select and prepare features
        X = data[feature_cols].copy()
        
        # Handle missing values
        X = X.fillna(0)
        
        # Encode categorical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                # Use label encoding for categorical columns
                X[col] = pd.Categorical(X[col]).codes
        
        return X
