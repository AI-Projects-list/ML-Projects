"""Text file data reader implementation."""

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from loguru import logger

from src.domain.entities import DataSource, DataSourceType
from src.domain.repositories import IDataReader


class TextDataReader(IDataReader):
    """Reads data from text files."""
    
    def __init__(self, default_encoding: str = "utf-8"):
        """
        Initialize text reader.
        
        Args:
            default_encoding: Default file encoding
        """
        self.default_encoding = default_encoding
    
    def can_read(self, source: DataSource) -> bool:
        """Check if this reader can handle the given source."""
        if source.source_type != DataSourceType.TXT:
            return False
        
        path = Path(source.path)
        return path.exists() and path.suffix.lower() in [".txt", ".text"]
    
    def read(self, source: DataSource) -> pd.DataFrame:
        """
        Read data from text file.
        
        Args:
            source: Data source to read from
            
        Returns:
            DataFrame with text content (one row per line or single row for full text)
        """
        if not self.can_read(source):
            raise ValueError(f"Cannot read source: {source.path}")
        
        logger.info(f"Reading text file: {source.path}")
        
        encoding = source.metadata.get("encoding", self.default_encoding)
        mode = source.metadata.get("mode", "lines")  # 'lines' or 'full'
        
        try:
            with open(source.path, "r", encoding=encoding) as f:
                content = f.read()
            
            if mode == "lines":
                # Split into lines and create DataFrame
                lines = content.split("\n")
                df = pd.DataFrame({"text": lines, "line_number": range(1, len(lines) + 1)})
                logger.info(f"Read {len(lines)} lines from text file")
            else:
                # Single row with full text
                df = pd.DataFrame({"text": [content]})
                logger.info(f"Read text file with {len(content)} characters")
            
            return df
        except Exception as e:
            logger.error(f"Error reading text file: {e}")
            raise
