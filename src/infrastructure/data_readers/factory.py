"""Data reader factory for managing multiple readers."""

from typing import List

from loguru import logger

from src.domain.entities import DataSource
from src.domain.repositories import IDataReader
from src.infrastructure.data_readers.csv_reader import CSVDataReader
from src.infrastructure.data_readers.pdf_reader import PDFDataReader
from src.infrastructure.data_readers.scanned_pdf_reader import ScannedPDFDataReader
from src.infrastructure.data_readers.text_reader import TextDataReader


class DataReaderFactory:
    """Factory for creating and managing data readers."""
    
    def __init__(self) -> None:
        """Initialize the factory with all available readers."""
        self.readers: List[IDataReader] = [
            CSVDataReader(),
            TextDataReader(),
            PDFDataReader(use_pdfplumber=True),
            ScannedPDFDataReader(),
        ]
    
    def add_reader(self, reader: IDataReader) -> None:
        """Add a custom reader to the factory."""
        self.readers.append(reader)
        logger.info(f"Added custom reader: {reader.__class__.__name__}")
    
    def get_reader(self, source: DataSource) -> IDataReader:
        """
        Get the appropriate reader for a data source.
        
        Args:
            source: Data source to read
            
        Returns:
            Appropriate data reader
            
        Raises:
            ValueError: If no suitable reader is found
        """
        for reader in self.readers:
            if reader.can_read(source):
                logger.info(
                    f"Selected reader: {reader.__class__.__name__} for {source.path}"
                )
                return reader
        
        raise ValueError(
            f"No suitable reader found for source type: {source.source_type}"
        )
