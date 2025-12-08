"""Example: Processing PDF documents."""

from pathlib import Path

from loguru import logger

from src.domain.entities import DataSource, DataSourceType
from src.infrastructure.config.container import Container
from src.infrastructure.config.logging import setup_logging
from src.infrastructure.config.settings import get_settings


def main() -> None:
    """Run example PDF processing."""
    # Initialize
    settings = get_settings()
    setup_logging(settings)
    container = Container(settings)
    
    logger.info("Starting PDF processing example")
    
    # Process text-based PDF
    pdf_source = DataSource(
        source_type=DataSourceType.PDF,
        path="data/raw/document.pdf",
        metadata={
            "extract_tables": True,  # Extract tables from PDF
        },
    )
    
    ingestion = container.data_ingestion_use_case
    processed_data = ingestion.execute(
        source=pdf_source,
        clean=True,
        transform=False,  # Text data might not need transformation
    )
    
    logger.info(f"Extracted text from {len(processed_data.data)} pages")
    logger.info(f"Sample text:\n{processed_data.data['text'].iloc[0][:200]}...")
    
    # Save processed data
    output_path = Path("data/processed/pdf_data.pkl")
    container.data_repository.save(processed_data, output_path)
    logger.info(f"Saved processed PDF data to {output_path}")


if __name__ == "__main__":
    main()
