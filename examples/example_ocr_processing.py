"""Example: OCR processing for scanned PDFs."""

from pathlib import Path

from loguru import logger

from src.domain.entities import DataSource, DataSourceType
from src.infrastructure.config.container import Container
from src.infrastructure.config.logging import setup_logging
from src.infrastructure.config.settings import get_settings


def main() -> None:
    """Run example scanned PDF processing with OCR."""
    # Initialize
    settings = get_settings()
    setup_logging(settings)
    container = Container(settings)
    
    logger.info("Starting scanned PDF (OCR) processing example")
    
    # Process scanned PDF with OCR
    scanned_pdf_source = DataSource(
        source_type=DataSourceType.PDF_SCAN,
        path="data/raw/scanned_document.pdf",
        metadata={
            "language": "eng",  # OCR language
            "dpi": 300,  # Resolution for image conversion
        },
    )
    
    ingestion = container.data_ingestion_use_case
    processed_data = ingestion.execute(
        source=scanned_pdf_source,
        clean=True,
        transform=False,
    )
    
    logger.info(f"OCR completed for {len(processed_data.data)} pages")
    
    # Display OCR confidence
    if "ocr_confidence" in processed_data.data.columns:
        avg_confidence = processed_data.data["ocr_confidence"].mean()
        logger.info(f"Average OCR confidence: {avg_confidence:.2f}%")
    
    # Show sample text
    logger.info(f"Sample OCR text:\n{processed_data.data['text'].iloc[0][:200]}...")
    
    # Save processed data
    output_path = Path("data/processed/ocr_data.pkl")
    container.data_repository.save(processed_data, output_path)
    logger.info(f"Saved OCR processed data to {output_path}")


if __name__ == "__main__":
    main()
