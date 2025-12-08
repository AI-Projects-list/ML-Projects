"""Domain entities representing core business objects."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd


class DataSourceType(Enum):
    """Types of data sources supported."""
    
    CSV = "csv"
    TXT = "txt"
    PDF = "pdf"
    PDF_SCAN = "pdf_scan"
    DATAFRAME = "dataframe"


class ProcessingStatus(Enum):
    """Status of data processing."""
    
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class DataSource:
    """Represents a data source."""
    
    source_type: DataSourceType
    path: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ProcessedData:
    """Represents processed data ready for analysis."""
    
    data: pd.DataFrame
    source: DataSource
    processing_steps: List[str] = field(default_factory=list)
    status: ProcessingStatus = ProcessingStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    processed_at: Optional[datetime] = None
    
    def mark_completed(self) -> None:
        """Mark data processing as completed."""
        self.status = ProcessingStatus.COMPLETED
        self.processed_at = datetime.now()
    
    def mark_failed(self) -> None:
        """Mark data processing as failed."""
        self.status = ProcessingStatus.FAILED
        self.processed_at = datetime.now()
    
    def add_processing_step(self, step: str) -> None:
        """Add a processing step to the history."""
        self.processing_steps.append(step)


@dataclass
class EDAReport:
    """Represents exploratory data analysis report."""
    
    data_shape: tuple
    column_types: Dict[str, str]
    missing_values: Dict[str, int]
    statistics: Dict[str, Any]
    correlations: Optional[pd.DataFrame] = None
    visualizations: Dict[str, str] = field(default_factory=dict)
    insights: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)


@dataclass
class ModelConfig:
    """Configuration for ML models."""
    
    model_type: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    target_column: Optional[str] = None
    feature_columns: List[str] = field(default_factory=list)
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class TrainedModel:
    """Represents a trained machine learning model."""
    
    model: Any
    config: ModelConfig
    metrics: Dict[str, float] = field(default_factory=dict)
    feature_importance: Optional[Dict[str, float]] = None
    training_data_shape: Optional[tuple] = None
    trained_at: datetime = field(default_factory=datetime.now)
    model_path: Optional[str] = None


@dataclass
class Prediction:
    """Represents model predictions."""
    
    predictions: pd.Series
    probabilities: Optional[pd.DataFrame] = None
    model_used: str = ""
    confidence_scores: Optional[pd.Series] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    predicted_at: datetime = field(default_factory=datetime.now)
