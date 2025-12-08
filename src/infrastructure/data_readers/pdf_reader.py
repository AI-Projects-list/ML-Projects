"""PDF data reader implementation."""

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pdfplumber
from loguru import logger
from PyPDF2 import PdfReader

from src.domain.entities import DataSource, DataSourceType
from src.domain.repositories import IDataReader


class PDFDataReader(IDataReader):
    """Reads data from PDF files (text-based PDFs)."""
    
    def __init__(self, use_pdfplumber: bool = True):
        """
        Initialize PDF reader.
        
        Args:
            use_pdfplumber: If True, use pdfplumber (better table extraction),
                          otherwise use PyPDF2
        """
        self.use_pdfplumber = use_pdfplumber
    
    def can_read(self, source: DataSource) -> bool:
        """Check if this reader can handle the given source."""
        if source.source_type != DataSourceType.PDF:
            return False
        
        path = Path(source.path)
        return path.exists() and path.suffix.lower() == ".pdf"
    
    def read(self, source: DataSource) -> pd.DataFrame:
        """
        Read data from PDF file.
        
        Args:
            source: Data source to read from
            
        Returns:
            DataFrame with extracted text and page information
        """
        if not self.can_read(source):
            raise ValueError(f"Cannot read source: {source.path}")
        
        logger.info(f"Reading PDF file: {source.path}")
        
        extract_tables = source.metadata.get("extract_tables", False)
        
        if self.use_pdfplumber:
            return self._read_with_pdfplumber(source.path, extract_tables)
        else:
            return self._read_with_pypdf2(source.path)
    
    def _read_with_pdfplumber(self, path: str, extract_tables: bool) -> pd.DataFrame:
        """Read PDF using pdfplumber (better for tables)."""
        try:
            data: List[Dict[str, Any]] = []
            
            with pdfplumber.open(path) as pdf:
                logger.info(f"PDF has {len(pdf.pages)} pages")
                
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    
                    row: Dict[str, Any] = {
                        "page_number": page_num,
                        "text": text if text else "",
                    }
                    
                    # Extract tables if requested
                    if extract_tables:
                        tables = page.extract_tables()
                        if tables:
                            row["tables"] = str(tables)
                            row["table_count"] = len(tables)
                        else:
                            row["tables"] = None
                            row["table_count"] = 0
                    
                    data.append(row)
            
            df = pd.DataFrame(data)
            logger.info(f"Extracted text from {len(df)} pages")
            return df
            
        except Exception as e:
            logger.error(f"Error reading PDF with pdfplumber: {e}")
            raise
    
    def _read_with_pypdf2(self, path: str) -> pd.DataFrame:
        """Read PDF using PyPDF2 (simpler, text only)."""
        try:
            reader = PdfReader(path)
            data: List[Dict[str, Any]] = []
            
            logger.info(f"PDF has {len(reader.pages)} pages")
            
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                data.append({
                    "page_number": page_num,
                    "text": text if text else "",
                })
            
            df = pd.DataFrame(data)
            logger.info(f"Extracted text from {len(df)} pages")
            return df
            
        except Exception as e:
            logger.error(f"Error reading PDF with PyPDF2: {e}")
            raise
