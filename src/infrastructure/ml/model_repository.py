"""Model repository for persistence."""

import pickle
from pathlib import Path
from typing import List

from loguru import logger

from src.domain.entities import TrainedModel
from src.domain.repositories import IModelRepository


class ModelRepository(IModelRepository):
    """Handles model saving and loading."""
    
    def save(self, model: TrainedModel, path: Path) -> None:
        """
        Save a trained model to disk.
        
        Args:
            model: Trained model to save
            path: File path for saving
        """
        logger.info(f"Saving model to {path}")
        
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(path, "wb") as f:
                pickle.dump(model, f)
            
            # Update model path
            model.model_path = str(path)
            
            logger.info(f"Model saved successfully: {path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load(self, path: Path) -> TrainedModel:
        """
        Load a trained model from disk.
        
        Args:
            path: File path to load from
            
        Returns:
            Loaded trained model
        """
        logger.info(f"Loading model from {path}")
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
            
            logger.info(f"Model loaded successfully: {path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def list_models(self, directory: Path) -> List[str]:
        """
        List all available models in a directory.
        
        Args:
            directory: Directory to search for models
            
        Returns:
            List of model file names
        """
        if not directory.exists():
            logger.warning(f"Model directory does not exist: {directory}")
            return []
        
        models = list(directory.glob("*.pkl"))
        model_names = [m.name for m in models]
        
        logger.info(f"Found {len(model_names)} models in {directory}")
        return model_names
