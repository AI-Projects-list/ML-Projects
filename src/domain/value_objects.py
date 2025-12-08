"""Value objects for the domain layer."""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class ColumnSchema:
    """Represents a column schema definition."""
    
    name: str
    dtype: str
    nullable: bool = True
    constraints: Dict[str, Any] = None
    
    def __post_init__(self) -> None:
        """Validate the column schema."""
        if self.constraints is None:
            object.__setattr__(self, 'constraints', {})


@dataclass(frozen=True)
class DataQualityMetrics:
    """Represents data quality metrics."""
    
    completeness: float  # 0-1 score
    consistency: float  # 0-1 score
    validity: float  # 0-1 score
    total_rows: int
    total_columns: int
    missing_cells: int
    duplicate_rows: int
    
    @property
    def overall_quality(self) -> float:
        """Calculate overall quality score."""
        return (self.completeness + self.consistency + self.validity) / 3
    
    def is_acceptable(self, threshold: float = 0.7) -> bool:
        """Check if data quality meets the threshold."""
        return self.overall_quality >= threshold


@dataclass(frozen=True)
class FeatureEngineering:
    """Represents feature engineering specifications."""
    
    numerical_features: List[str]
    categorical_features: List[str]
    datetime_features: List[str]
    derived_features: Dict[str, str]  # feature_name: formula/description
    
    @property
    def all_features(self) -> List[str]:
        """Get all feature names."""
        return (
            self.numerical_features
            + self.categorical_features
            + self.datetime_features
            + list(self.derived_features.keys())
        )
