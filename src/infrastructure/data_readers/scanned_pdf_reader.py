"""Scanned PDF (OCR) data reader implementation."""

import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytesseract
from loguru import logger
from pdf2image import convert_from_path
from PIL import Image

from src.domain.entities import DataSource, DataSourceType
from src.domain.repositories import IDataReader


class ScannedPDFDataReader(IDataReader):
    """Reads data from scanned PDF files using OCR."""
    
    def __init__(
        self,
        tesseract_cmd: str | None = None,
        language: str = "eng",
        dpi: int = 300,
    ):
        """
        Initialize scanned PDF reader with OCR.
        
        Args:
            tesseract_cmd: Path to tesseract executable (if not in PATH)
            language: OCR language code
            dpi: DPI for PDF to image conversion
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        self.language = language
        self.dpi = dpi
    
    def can_read(self, source: DataSource) -> bool:
        """Check if this reader can handle the given source."""
        if source.source_type != DataSourceType.PDF_SCAN:
            return False
        
        path = Path(source.path)
        return path.exists() and path.suffix.lower() == ".pdf"
    
    def read(self, source: DataSource) -> pd.DataFrame:
        """
        Read data from scanned PDF using OCR.
        
        Args:
            source: Data source to read from
            
        Returns:
            DataFrame with OCR-extracted text and page information
        """
        if not self.can_read(source):
            raise ValueError(f"Cannot read source: {source.path}")
        
        logger.info(f"Reading scanned PDF file with OCR: {source.path}")
        
        language = source.metadata.get("language", self.language)
        dpi = source.metadata.get("dpi", self.dpi)
        
        try:
            # Convert PDF to images
            logger.info(f"Converting PDF to images (DPI: {dpi})...")
            images = convert_from_path(source.path, dpi=dpi)
            logger.info(f"Converted {len(images)} pages to images")
            
            # Perform OCR on each image
            data: List[Dict[str, Any]] = []
            
            for page_num, image in enumerate(images, start=1):
                logger.info(f"Performing OCR on page {page_num}/{len(images)}")
                
                # Extract text using Tesseract
                text = pytesseract.image_to_string(image, lang=language)
                
                # Get OCR confidence data
                ocr_data = pytesseract.image_to_data(
                    image, lang=language, output_type=pytesseract.Output.DICT
                )
                
                # Calculate average confidence
                confidences = [
                    int(conf) for conf in ocr_data["conf"] if conf != "-1"
                ]
                avg_confidence = (
                    sum(confidences) / len(confidences) if confidences else 0
                )
                
                data.append({
                    "page_number": page_num,
                    "text": text.strip(),
                    "ocr_confidence": round(avg_confidence, 2),
                    "word_count": len(text.split()),
                })
            
            df = pd.DataFrame(data)
            logger.info(
                f"OCR completed for {len(df)} pages. "
                f"Average confidence: {df['ocr_confidence'].mean():.2f}%"
            )
            return df
            
        except Exception as e:
            logger.error(f"Error reading scanned PDF with OCR: {e}")
            raise
