"""Use case for making predictions."""

from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from src.domain.entities import Prediction, TrainedModel
from src.domain.repositories import IModelRepository, IPredictor


class PredictionUseCase:
    """Handles the prediction workflow."""
    
    def __init__(
        self,
        predictor: IPredictor,
        model_repository: IModelRepository,
    ):
        """
        Initialize prediction use case.
        
        Args:
            predictor: Predictor implementation
            model_repository: Model repository for loading models
        """
        self.predictor = predictor
        self.model_repository = model_repository
    
    def execute(
        self,
        data: pd.DataFrame,
        model: Optional[TrainedModel] = None,
        model_path: Optional[Path] = None,
    ) -> Prediction:
        """
        Execute the prediction workflow.
        
        Args:
            data: Input data for predictions
            model: Trained model (if already loaded)
            model_path: Path to load model from (if model not provided)
            
        Returns:
            Prediction results
        """
        logger.info("Starting prediction workflow...")
        
        # Load model if not provided
        if model is None:
            if model_path is None:
                raise ValueError("Either model or model_path must be provided")
            
            logger.info(f"Loading model from {model_path}")
            model = self.model_repository.load(model_path)
        
        # Make predictions
        prediction = self.predictor.predict(model, data)
        
        logger.info(f"Prediction workflow completed: {len(prediction.predictions)} predictions")
        
        return prediction
