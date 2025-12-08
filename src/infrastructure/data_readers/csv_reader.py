"""CSV data reader implementation."""

from pathlib import Path
from typing import Any, Dict

import pandas as pd
from loguru import logger

from src.domain.entities import DataSource, DataSourceType
from src.domain.repositories import IDataReader


class CSVDataReader(IDataReader):
    """Reads data from CSV files."""
    
    def __init__(self, default_encoding: str = "utf-8", **kwargs: Any):
        """
        Initialize CSV reader.
        
        Args:
            default_encoding: Default file encoding
            **kwargs: Additional parameters for pd.read_csv
        """
        self.default_encoding = default_encoding
        self.read_params = kwargs
    
    def can_read(self, source: DataSource) -> bool:
        """Check if this reader can handle the given source."""
        if source.source_type != DataSourceType.CSV:
            return False
        
        path = Path(source.path)
        return path.exists() and path.suffix.lower() == ".csv"
    
    def read(self, source: DataSource) -> pd.DataFrame:
        """
        Read data from CSV file.
        
        Args:
            source: Data source to read from
            
        Returns:
            DataFrame containing the data
        """
        if not self.can_read(source):
            raise ValueError(f"Cannot read source: {source.path}")
        
        logger.info(f"Reading CSV file: {source.path}")
        
        # Get encoding from metadata or use default
        encoding = source.metadata.get("encoding", self.default_encoding)
        
        # Merge source metadata with default read params
        read_params = {**self.read_params, **source.metadata.get("read_params", {})}
        
        try:
            df = pd.read_csv(source.path, encoding=encoding, **read_params)
            logger.info(f"Successfully read {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            raise
