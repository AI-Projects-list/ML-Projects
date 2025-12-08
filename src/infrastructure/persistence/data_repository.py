"""Data repository for persistence."""

import pickle
from pathlib import Path

from loguru import logger

from src.domain.entities import ProcessedData
from src.domain.repositories import IDataRepository


class DataRepository(IDataRepository):
    """Handles data saving and loading."""
    
    def save(self, data: ProcessedData, path: Path) -> None:
        """
        Save processed data to disk.
        
        Args:
            data: Processed data to save
            path: File path for saving
        """
        logger.info(f"Saving processed data to {path}")
        
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(path, "wb") as f:
                pickle.dump(data, f)
            
            logger.info(f"Data saved successfully: {path}")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise
    
    def load(self, path: Path) -> ProcessedData:
        """
        Load processed data from disk.
        
        Args:
            path: File path to load from
            
        Returns:
            Loaded processed data
        """
        logger.info(f"Loading processed data from {path}")
        
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            
            logger.info(f"Data loaded successfully: {path}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
