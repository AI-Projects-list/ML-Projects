"""Use case for data ingestion and preprocessing."""

from pathlib import Path

from loguru import logger

from src.domain.entities import DataSource, ProcessedData, ProcessingStatus
from src.domain.repositories import IDataProcessor
from src.infrastructure.data_readers.factory import DataReaderFactory


class DataIngestionUseCase:
    """Handles the complete data ingestion pipeline."""
    
    def __init__(
        self,
        reader_factory: DataReaderFactory,
        processor: IDataProcessor,
    ):
        """
        Initialize data ingestion use case.
        
        Args:
            reader_factory: Factory for creating data readers
            processor: Data processor for cleaning and transformation
        """
        self.reader_factory = reader_factory
        self.processor = processor
    
    def execute(
        self,
        source: DataSource,
        clean: bool = True,
        transform: bool = True,
        validate: bool = True,
    ) -> ProcessedData:
        """
        Execute the data ingestion pipeline.
        
        Args:
            source: Data source to ingest
            clean: Whether to clean the data
            transform: Whether to transform the data
            validate: Whether to validate the data
            
        Returns:
            Processed data ready for analysis
        """
        logger.info(f"Starting data ingestion for {source.path}")
        
        # Read data
        reader = self.reader_factory.get_reader(source)
        raw_data = reader.read(source)
        
        # Create ProcessedData object
        processed_data = ProcessedData(
            data=raw_data,
            source=source,
            status=ProcessingStatus.IN_PROGRESS,
        )
        
        try:
            # Clean data
            if clean:
                logger.info("Cleaning data...")
                processed_data.data = self.processor.clean(processed_data.data)
                processed_data.add_processing_step("cleaned")
            
            # Transform data
            if transform:
                logger.info("Transforming data...")
                processed_data.data = self.processor.transform(processed_data.data)
                processed_data.add_processing_step("transformed")
            
            # Validate data
            if validate:
                logger.info("Validating data...")
                is_valid = self.processor.validate(processed_data.data)
                processed_data.metadata["validation_passed"] = is_valid
                processed_data.add_processing_step("validated")
            
            # Mark as completed
            processed_data.mark_completed()
            logger.info("Data ingestion completed successfully")
            
        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            processed_data.mark_failed()
            raise
        
        return processed_data
