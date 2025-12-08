"""Data processor implementation."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.domain.repositories import IDataProcessor
from src.domain.value_objects import DataQualityMetrics


class DataProcessor(IDataProcessor):
    """Handles data cleaning, transformation, and validation."""
    
    def __init__(
        self,
        missing_threshold: float = 0.5,
        duplicate_handling: str = "remove",
    ):
        """
        Initialize data processor.
        
        Args:
            missing_threshold: Max allowed missing ratio per column (0-1)
            duplicate_handling: How to handle duplicates ('remove', 'keep_first', 'keep_last')
        """
        self.missing_threshold = missing_threshold
        self.duplicate_handling = duplicate_handling
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
    
    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the data by handling missing values, duplicates, and outliers.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning...")
        df = data.copy()
        original_shape = df.shape
        
        # Handle missing values
        logger.info("Handling missing values...")
        df = self._handle_missing_values(df)
        
        # Remove duplicates
        if self.duplicate_handling == "remove":
            before_dup = len(df)
            df = df.drop_duplicates()
            logger.info(f"Removed {before_dup - len(df)} duplicate rows")
        elif self.duplicate_handling == "keep_first":
            df = df.drop_duplicates(keep="first")
        elif self.duplicate_handling == "keep_last":
            df = df.drop_duplicates(keep="last")
        
        # Remove columns with excessive missing values
        missing_ratio = df.isnull().sum() / len(df)
        cols_to_drop = missing_ratio[missing_ratio > self.missing_threshold].index.tolist()
        if cols_to_drop:
            logger.warning(f"Dropping columns with >{self.missing_threshold*100}% missing: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)
        
        logger.info(f"Cleaning completed: {original_shape} -> {df.shape}")
        return df
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data (encoding, scaling, feature engineering).
        
        Args:
            data: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        logger.info("Starting data transformation...")
        df = data.copy()
        
        # Identify column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
        
        logger.info(f"Numeric columns: {len(numeric_cols)}")
        logger.info(f"Categorical columns: {len(categorical_cols)}")
        logger.info(f"Datetime columns: {len(datetime_cols)}")
        
        # Encode categorical variables
        if categorical_cols:
            df = self._encode_categorical(df, categorical_cols)
        
        # Handle datetime features
        if datetime_cols:
            df = self._extract_datetime_features(df, datetime_cols)
        
        # Scale numeric features (optional - can be enabled via metadata)
        # Scaling is typically done after train/test split to avoid data leakage
        
        logger.info(f"Transformation completed: {df.shape}")
        return df
    
    def validate(self, data: pd.DataFrame) -> bool:
        """
        Validate data quality.
        
        Args:
            data: Input DataFrame
            
        Returns:
            True if data passes validation
        """
        logger.info("Validating data quality...")
        
        metrics = self.calculate_quality_metrics(data)
        
        logger.info(f"Data Quality Metrics:")
        logger.info(f"  Completeness: {metrics.completeness:.2%}")
        logger.info(f"  Consistency: {metrics.consistency:.2%}")
        logger.info(f"  Validity: {metrics.validity:.2%}")
        logger.info(f"  Overall Quality: {metrics.overall_quality:.2%}")
        
        is_valid = metrics.is_acceptable(threshold=0.7)
        
        if is_valid:
            logger.info("✓ Data validation passed")
        else:
            logger.warning("✗ Data validation failed - quality below threshold")
        
        return is_valid
    
    def calculate_quality_metrics(self, data: pd.DataFrame) -> DataQualityMetrics:
        """Calculate data quality metrics."""
        total_cells = data.shape[0] * data.shape[1]
        missing_cells = data.isnull().sum().sum()
        duplicate_rows = data.duplicated().sum()
        
        # Completeness: ratio of non-missing values
        completeness = 1 - (missing_cells / total_cells) if total_cells > 0 else 0
        
        # Consistency: ratio of non-duplicate rows
        consistency = 1 - (duplicate_rows / len(data)) if len(data) > 0 else 0
        
        # Validity: check for valid data types and ranges
        validity_score = self._calculate_validity_score(data)
        
        return DataQualityMetrics(
            completeness=completeness,
            consistency=consistency,
            validity=validity_score,
            total_rows=data.shape[0],
            total_columns=data.shape[1],
            missing_cells=int(missing_cells),
            duplicate_rows=int(duplicate_rows),
        )
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on column type."""
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in [np.float64, np.int64]:
                    # Numeric: fill with median
                    df[col].fillna(df[col].median(), inplace=True)
                elif df[col].dtype == "object":
                    # Categorical: fill with mode or 'Unknown'
                    mode_val = df[col].mode()
                    fill_val = mode_val[0] if len(mode_val) > 0 else "Unknown"
                    df[col].fillna(fill_val, inplace=True)
        
        return df
    
    def _encode_categorical(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Encode categorical variables."""
        for col in cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                df[col] = self.encoders[col].fit_transform(df[col].astype(str))
            else:
                df[col] = self.encoders[col].transform(df[col].astype(str))
            
            logger.info(f"Encoded column: {col}")
        
        return df
    
    def _extract_datetime_features(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Extract features from datetime columns."""
        for col in cols:
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_dayofweek"] = df[col].dt.dayofweek
            
            logger.info(f"Extracted datetime features from: {col}")
        
        return df
    
    def _calculate_validity_score(self, df: pd.DataFrame) -> float:
        """Calculate validity score based on data type consistency."""
        valid_cols = 0
        
        for col in df.columns:
            # Check if column values are consistent with their dtype
            try:
                if df[col].dtype in [np.float64, np.int64]:
                    # Check for inf values
                    if not df[col].replace([np.inf, -np.inf], np.nan).isnull().all():
                        valid_cols += 1
                else:
                    valid_cols += 1
            except Exception:
                pass
        
        return valid_cols / len(df.columns) if len(df.columns) > 0 else 0
