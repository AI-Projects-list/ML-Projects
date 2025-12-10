# Data Reader Factory - Comprehensive Documentation

## File Information
- **Source File**: `src/infrastructure/data_readers/factory.py`
- **Purpose**: Factory for creating appropriate data readers based on source type
- **Layer**: Infrastructure Layer (Data Readers)
- **Pattern**: Factory Pattern, Chain of Responsibility

## Complete Annotated Code

```python
"""Data reader factory for managing multiple readers."""
# WHAT: Module docstring
# WHY: Document factory purpose
# HOW: Python docstring
# BENEFIT: Clear module intent
# TRADE-OFF: Brief description

from typing import List
# WHAT: Import List type hint
# WHY: Type readers collection
# HOW: Typing module
# BENEFIT: Type safety for reader list
# TRADE-OFF: Import overhead

from loguru import logger
# WHAT: Import logger
# WHY: Log reader selection
# HOW: Loguru singleton
# BENEFIT: Observability
# TRADE-OFF: External dependency

from src.domain.entities import DataSource
# WHAT: Import DataSource entity
# WHY: Type-safe source parameter
# HOW: Domain import
# BENEFIT: Encapsulated source metadata
# TRADE-OFF: Domain coupling

from src.domain.repositories import IDataReader
# WHAT: Import reader interface
# WHY: Type readers list
# HOW: Repository pattern interface
# BENEFIT: Polymorphism
# TRADE-OFF: Abstraction layer

from src.infrastructure.data_readers.csv_reader import CSVDataReader
from src.infrastructure.data_readers.pdf_reader import PDFDataReader
from src.infrastructure.data_readers.scanned_pdf_reader import ScannedPDFDataReader
from src.infrastructure.data_readers.text_reader import TextDataReader
# WHAT: Import all concrete readers
# WHY: Register available readers
# HOW: Individual imports
# BENEFIT: All readers available
# TRADE-OFF: Many imports, tight coupling


class DataReaderFactory:
    # WHAT: Factory class for reader creation
    # WHY: Centralize reader selection logic
    # HOW: Factory pattern implementation
    # BENEFIT: Single responsibility, extensible
    # TRADE-OFF: Additional layer

    """Factory for creating and managing data readers."""
    # WHAT: Class docstring
    # WHY: Document factory purpose
    # HOW: Single-line description
    # BENEFIT: Clear responsibility
    # TRADE-OFF: Brief

    def __init__(self) -> None:
        # WHAT: Constructor initializing reader list
        # WHY: Register all available readers
        # HOW: Create list of reader instances
        # BENEFIT: Pre-configured readers ready
        # TRADE-OFF: Upfront instantiation

        """Initialize the factory with all available readers."""
        # WHAT: Constructor docstring
        # WHY: Document initialization
        # HOW: Brief description
        # BENEFIT: Clear purpose
        # TRADE-OFF: Could list readers

        self.readers: List[IDataReader] = [
            # WHAT: List of registered readers
            # WHY: Chain of responsibility for can_read checks
            # HOW: List with type hint
            # BENEFIT: Ordered reader checking
            # TRADE-OFF: Memory for all readers

            CSVDataReader(),
            # WHAT: CSV file reader
            # WHY: Handle .csv files
            # HOW: Default instantiation
            # BENEFIT: Pandas-based CSV reading
            # TRADE-OFF: Memory for instance

            TextDataReader(),
            # WHAT: Text file reader
            # WHY: Handle .txt files
            # HOW: Default instantiation
            # BENEFIT: Simple text reading
            # TRADE-OFF: Instance overhead

            PDFDataReader(use_pdfplumber=True),
            # WHAT: PDF reader with pdfplumber
            # WHY: Better table extraction
            # HOW: Configured with pdfplumber=True
            # BENEFIT: Quality PDF parsing
            # TRADE-OFF: Additional dependency

            ScannedPDFDataReader(),
            # WHAT: OCR-based PDF reader
            # WHY: Handle scanned/image PDFs
            # HOW: Default instantiation
            # BENEFIT: OCR capabilities
            # TRADE-OFF: Slower, OCR dependency

        ]
    
    def add_reader(self, reader: IDataReader) -> None:
        # WHAT: Method to register custom reader
        # WHY: Extensibility - add new reader types
        # HOW: Append to readers list
        # BENEFIT: Open/Closed principle
        # TRADE-OFF: Runtime modification

        """Add a custom reader to the factory."""
        # WHAT: Method docstring
        # WHY: Document extensibility
        # HOW: Brief description
        # BENEFIT: Clear API
        # TRADE-OFF: Could document order importance

        self.readers.append(reader)
        # WHAT: Add reader to list
        # WHY: Make available for selection
        # HOW: List append
        # BENEFIT: Extended capabilities
        # TRADE-OFF: Mutable state

        logger.info(f"Added custom reader: {reader.__class__.__name__}")
        # WHAT: Log reader addition
        # WHY: Track factory configuration
        # HOW: Info log with class name
        # BENEFIT: Observability
        # TRADE-OFF: Logging overhead
    
    def get_reader(self, source: DataSource) -> IDataReader:
        # WHAT: Factory method to get appropriate reader
        # WHY: Core factory responsibility
        # HOW: Chain of responsibility pattern
        # BENEFIT: Automatic reader selection
        # TRADE-OFF: Linear search

        """
        Get the appropriate reader for a data source.

        Args:
            source: Data source to read

        Returns:
            Appropriate data reader

        Raises:
            ValueError: If no suitable reader is found
        """
        # WHAT: Method docstring
        # WHY: Document factory logic
        # HOW: Args/Returns/Raises sections
        # BENEFIT: Clear contract
        # TRADE-OFF: Verbose

        for reader in self.readers:
            # WHAT: Iterate through readers
            # WHY: Find first matching reader
            # HOW: For loop over readers list
            # BENEFIT: Chain of responsibility
            # TRADE-OFF: O(n) complexity

            if reader.can_read(source):
                # WHAT: Check if reader can handle source
                # WHY: Polymorphic capability checking
                # HOW: Call can_read interface method
                # BENEFIT: Flexible reader selection
                # TRADE-OFF: Method call overhead per reader

                logger.info(
                    f"Selected reader: {reader.__class__.__name__} for {source.path}"
                )
                # WHAT: Log selected reader
                # WHY: Observability for debugging
                # HOW: Info log with reader and source
                # BENEFIT: Track reader selection
                # TRADE-OFF: Logging I/O

                return reader
                # WHAT: Return matching reader
                # WHY: Provide appropriate reader
                # HOW: Early return on first match
                # BENEFIT: Efficient selection
                # TRADE-OFF: Order-dependent

        raise ValueError(
            f"No suitable reader found for source type: {source.source_type}"
        )
        # WHAT: Raise error if no reader found
        # WHY: Fail fast with clear error
        # HOW: ValueError with source type
        # BENEFIT: Clear error message
        # TRADE-OFF: Exception overhead
```

---

## Design Patterns

### 1. **Factory Pattern**
- **Purpose**: Encapsulate reader creation logic
- **Benefits**: Single creation point, extensible
- **Trade-offs**: Additional indirection

### 2. **Chain of Responsibility**
- **Purpose**: First matching reader handles request
- **Benefits**: Flexible, extensible, ordered
- **Trade-offs**: Linear search, order-dependent

### 3. **Strategy Pattern**
- **Purpose**: Different readers for different sources
- **Benefits**: Polymorphic reader selection
- **Trade-offs**: Multiple reader classes

---

## Pros & Cons

### Pros ✅
1. **Automatic Selection** - No manual reader choice needed
2. **Extensible** - Add custom readers easily
3. **Type Safe** - IDataReader interface
4. **Observable** - Logs reader selection
5. **Pre-configured** - All readers ready
6. **Clean API** - Simple get_reader method

### Cons ❌
1. **Eager Instantiation** - All readers created upfront
2. **Linear Search** - O(n) reader checking
3. **Order Dependent** - First match wins
4. **Tight Coupling** - Imports all concrete readers
5. **No Caching** - Creates reader each time
6. **Mutable State** - Can add readers after construction

---

## Usage Examples

### Example 1: Basic Usage
```python
factory = DataReaderFactory()
source = DataSource(path="data.csv", data_type="csv")
reader = factory.get_reader(source)
data = reader.read(source)
```

### Example 2: Custom Reader
```python
class ExcelReader(IDataReader):
    def can_read(self, source): 
        return source.path.endswith('.xlsx')
    def read(self, source):
        return pd.read_excel(source.path)

factory = DataReaderFactory()
factory.add_reader(ExcelReader())
```

### Example 3: Different Sources
```python
factory = DataReaderFactory()
csv_reader = factory.get_reader(DataSource(path="data.csv", data_type="csv"))
pdf_reader = factory.get_reader(DataSource(path="doc.pdf", data_type="pdf"))
txt_reader = factory.get_reader(DataSource(path="text.txt", data_type="txt"))
```

---

## Related Files
- **Interface**: repositories.py (IDataReader)
- **Readers**: csv_reader.py, pdf_reader.py, text_reader.py, scanned_pdf_reader.py
- **Domain**: entities.py (DataSource)
- **Used By**: data_ingestion.py use case
