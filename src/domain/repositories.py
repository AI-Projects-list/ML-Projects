"""Repository interfaces (ports) for the domain layer."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.domain.entities import (
    DataSource,
    EDAReport,
    ModelConfig,
    Prediction,
    ProcessedData,
    TrainedModel,
)


class IDataReader(ABC):
    """Interface for reading data from various sources."""
    
    @abstractmethod
    def can_read(self, source: DataSource) -> bool:
        """Check if this reader can handle the given source."""
        pass
    
    @abstractmethod
    def read(self, source: DataSource) -> pd.DataFrame:
        """Read data from the source."""
        pass


class IDataProcessor(ABC):
    """Interface for data processing operations."""
    
    @abstractmethod
    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean the data."""
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        pass
    
    @abstractmethod
    def validate(self, data: pd.DataFrame) -> bool:
        """Validate the data quality."""
        pass


class IEDAAnalyzer(ABC):
    """Interface for exploratory data analysis."""
    
    @abstractmethod
    def analyze(self, data: ProcessedData) -> EDAReport:
        """Perform exploratory data analysis."""
        pass
    
    @abstractmethod
    def generate_visualizations(self, data: ProcessedData, output_dir: Path) -> List[str]:
        """Generate visualization plots."""
        pass


class IModelTrainer(ABC):
    """Interface for model training."""
    
    @abstractmethod
    def train(self, data: ProcessedData, config: ModelConfig) -> TrainedModel:
        """Train a machine learning model."""
        pass
    
    @abstractmethod
    def evaluate(self, model: TrainedModel, test_data: pd.DataFrame) -> dict:
        """Evaluate model performance."""
        pass


class IPredictor(ABC):
    """Interface for making predictions."""
    
    @abstractmethod
    def predict(self, model: TrainedModel, data: pd.DataFrame) -> Prediction:
        """Make predictions using the trained model."""
        pass


class IModelRepository(ABC):
    """Interface for model persistence."""
    
    @abstractmethod
    def save(self, model: TrainedModel, path: Path) -> None:
        """Save a trained model."""
        pass
    
    @abstractmethod
    def load(self, path: Path) -> TrainedModel:
        """Load a trained model."""
        pass
    
    @abstractmethod
    def list_models(self, directory: Path) -> List[str]:
        """List all available models."""
        pass


class IDataRepository(ABC):
    """Interface for data persistence."""
    
    @abstractmethod
    def save(self, data: ProcessedData, path: Path) -> None:
        """Save processed data."""
        pass
    
    @abstractmethod
    def load(self, path: Path) -> ProcessedData:
        """Load processed data."""
        pass
