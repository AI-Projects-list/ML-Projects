# Data Ingestion Use Case - Detailed Code Documentation

**File**: `src/application/use_cases/data_ingestion.py`  
**Purpose**: Orchestrate complete data ingestion pipeline  
**Layer**: Application (Use Case orchestration)  
**Pattern**: Use Case Pattern + Template Method

---

## Overview

This use case **orchestrates** the data ingestion process by coordinating multiple domain services. It's the **APPLICATION LAYER** responsibility to orchestrate, not to implement business logic.

**Key Principle**: Use cases orchestrate, repositories implement.

---

## Complete Code with Line-by-Line Comments

```python
"""Use case for data ingestion and preprocessing."""
# WHAT: Module for data ingestion orchestration
# WHY: Coordinate reading, cleaning, transforming data
# LAYER: Application (not domain or infrastructure)
# RESPONSIBILITY: Orchestration only, no business logic

from pathlib import Path
# WHAT: Path handling
# WHY: Type hints for file paths
# UNUSED IN THIS FILE: Could be removed
# NOTE: Imported but not used (dead code)

from loguru import logger
# WHAT: Structured logging library
# WHY: Better than print(), production-ready
# BENEFIT: Automatic timestamps, levels, formatting
# TRADE-OFF: External dependency
# ALTERNATIVE: Python's logging module (stdlib)

from src.domain.entities import DataSource, ProcessedData, ProcessingStatus
# WHAT: Import domain entities
# WHY: Work with rich domain types
# DEPENDENCY: application → domain (correct direction)
# BENEFIT: Type-safe orchestration

from src.domain.repositories import IDataProcessor
# WHAT: Import processor interface
# WHY: Depend on abstraction, not implementation
# PATTERN: Dependency Inversion Principle
# BENEFIT: Testable, swappable implementations

from src.infrastructure.data_readers.factory import DataReaderFactory
# WHAT: Import reader factory
# WHY: Create appropriate readers dynamically
# NOTE: This violates clean architecture!
# PROBLEM: Application depends on infrastructure
# FIX: Should depend on IDataReaderFactory interface
# TRADE-OFF: Pragmatic choice for simplicity


class DataIngestionUseCase:
    """Handles the complete data ingestion pipeline."""
    # WHAT: Use case for data ingestion
    # WHY: Orchestrate reading → cleaning → transforming → validating
    # PATTERN: Use Case Pattern (application service)
    # RESPONSIBILITY: Coordinate, delegate, log progress
    
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
        # WHAT: Constructor with dependency injection
        # WHY: Inject dependencies, not create them
        # PATTERN: Dependency Injection
        # BENEFIT: Easy to test, flexible, SOLID compliant
        
        self.reader_factory = reader_factory
        # WHAT: Store factory reference
        # WHY: Create readers based on source type
        # PROBLEM: Should be IDataReaderFactory interface
        
        self.processor = processor
        # WHAT: Store processor reference
        # WHY: Delegate cleaning/transforming
        # GOOD: Depends on interface, not implementation
    
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
        # WHAT: Main execution method
        # WHY: Standard use case pattern (execute method)
        # PARAMETERS: Feature flags for pipeline steps
        # RETURN: Rich ProcessedData entity
        
        # WHY boolean parameters?
        # - Flexible pipeline (skip steps)
        # - Useful for debugging
        # - Reusable for different scenarios
        # TRADE-OFF: Not as explicit as separate methods
        
        logger.info(f"Starting data ingestion for {source.path}")
        # WHAT: Log pipeline start
        # WHY: Observability, debugging
        # FORMAT: f-string with source path
        # BENEFIT: Track which file is being processed
        
        # Read data
        reader = self.reader_factory.get_reader(source)
        # WHAT: Get appropriate reader
        # WHY: Different readers for CSV, PDF, TXT
        # PATTERN: Factory Pattern
        # HOW: Factory checks source.source_type
        # BENEFIT: Extensible (add new readers without changing this code)
        
        raw_data = reader.read(source)
        # WHAT: Read data from source
        # WHY: Load into pandas DataFrame
        # RETURN: DataFrame (standard format)
        # TRADE-OFF: Loads entire file into memory
        
        # Create ProcessedData object
        processed_data = ProcessedData(
            data=raw_data,
            source=source,
            status=ProcessingStatus.IN_PROGRESS,
        )
        # WHAT: Wrap DataFrame in rich entity
        # WHY: Track metadata, status, processing steps
        # BENEFIT: Domain-driven design
        # STATUS: IN_PROGRESS (will change to COMPLETED)
        # ALTERNATIVE: Could pass raw DataFrame through pipeline
        # TRADE-OFF: More complex, but richer context
        
        try:
            # WHAT: Error handling block
            # WHY: Catch processing errors, mark as failed
            # PATTERN: Try-except for error handling
            # BENEFIT: Graceful failure with status tracking
            
            # Clean data
            if clean:
                # WHAT: Conditional cleaning step
                # WHY: Optional step (user can skip)
                # USE CASE: Skip for already-clean data
                
                logger.info("Cleaning data...")
                # WHAT: Log step start
                # WHY: Observability
                
                processed_data.data = self.processor.clean(processed_data.data)
                # WHAT: Delegate cleaning to processor
                # WHY: Separation of concerns
                # MUTATES: processed_data.data (DataFrame replacement)
                # PATTERN: Strategy pattern (processor implements cleaning)
                # TRADE-OFF: Functional vs imperative style
                
                processed_data.add_processing_step("cleaned")
                # WHAT: Record processing step
                # WHY: Track data lineage
                # BENEFIT: Reproducibility, debugging
                # STORAGE: Appends to processing_steps list
            
            # Transform data
            if transform:
                # WHAT: Conditional transformation step
                # WHY: Optional step
                # USE CASE: Skip for analysis-only workflows
                
                logger.info("Transforming data...")
                processed_data.data = self.processor.transform(processed_data.data)
                # WHAT: Delegate transformation
                # WHY: Encoding, scaling, feature engineering
                # MUTATES: processed_data.data
                # DEPENDENCY: Must run after cleaning
                
                processed_data.add_processing_step("transformed")
                # WHAT: Record transformation step
                # WHY: Track pipeline history
            
            # Validate data
            if validate:
                # WHAT: Conditional validation step
                # WHY: Optional quality check
                # USE CASE: Skip for trusted data sources
                
                logger.info("Validating data...")
                is_valid = self.processor.validate(processed_data.data)
                # WHAT: Check data quality
                # WHY: Ensure data meets standards
                # RETURN: Boolean (pass/fail)
                # TRADE-OFF: Doesn't halt on failure
                
                processed_data.metadata["validation_passed"] = is_valid
                # WHAT: Store validation result
                # WHY: Record for downstream use
                # STORAGE: In metadata dict
                # TRADE-OFF: Could raise exception on failure
                # ALTERNATIVE: Fail fast approach
                
                processed_data.add_processing_step("validated")
                # WHAT: Record validation step
            
            # Mark as completed
            processed_data.mark_completed()
            # WHAT: Set status to COMPLETED
            # WHY: Indicate successful processing
            # MUTATES: status field, completed_at timestamp
            # BENEFIT: Rich status tracking
            
            logger.info("Data ingestion completed successfully")
            # WHAT: Log success
            # WHY: Observability
            
        except Exception as e:
            # WHAT: Catch any processing errors
            # WHY: Graceful error handling
            # SCOPE: Broad exception (could be more specific)
            
            logger.error(f"Data ingestion failed: {e}")
            # WHAT: Log error details
            # WHY: Debugging
            # FORMAT: Include exception message
            
            processed_data.mark_failed()
            # WHAT: Set status to FAILED
            # WHY: Track failure state
            # MUTATES: status field
            
            raise
            # WHAT: Re-raise exception
            # WHY: Let caller handle it
            # PATTERN: Log and re-raise
            # BENEFIT: Don't swallow exceptions
        
        return processed_data
        # WHAT: Return processed data entity
        # WHY: Provide result to caller
        # INCLUDES: Data + metadata + status + history
        # BENEFIT: Rich context for downstream use
```

---

## Design Patterns Used

### 1. **Use Case Pattern** (Clean Architecture)
- **WHAT**: Application service orchestrating workflow
- **WHY**: Separate orchestration from implementation
- **BENEFIT**: Testable, reusable business workflows

### 2. **Template Method Pattern**
- **WHAT**: Fixed algorithm structure with variable steps
- **WHY**: Flexible pipeline (skip steps via flags)
- **STEPS**: Read → Clean → Transform → Validate

### 3. **Factory Pattern**
- **WHAT**: `reader_factory.get_reader(source)`
- **WHY**: Create appropriate reader dynamically
- **BENEFIT**: Open/Closed Principle

### 4. **Dependency Injection**
- **WHAT**: Pass dependencies in constructor
- **WHY**: Loose coupling, testable
- **BENEFIT**: SOLID principles

---

## Pros & Cons

### ✅ Pros

1. **Flexible Pipeline**: Boolean flags allow skipping steps
2. **Error Handling**: Try-except with status tracking
3. **Observability**: Extensive logging
4. **Rich Return Type**: ProcessedData with metadata
5. **Separation of Concerns**: Delegates to specialized classes
6. **Testable**: Dependency injection enables mocking

### ❌ Cons

1. **Architecture Violation**: Depends on `DataReaderFactory` (infrastructure)
   - **FIX**: Create `IDataReaderFactory` interface in domain
2. **Broad Exception Handling**: Catches all exceptions
   - **FIX**: Catch specific exceptions (IOError, ValueError)
3. **Boolean Parameters**: Less explicit than separate methods
   - **ALTERNATIVE**: `execute_full()`, `execute_cleaning_only()`
4. **No Progress Callbacks**: Can't track long-running operations
   - **FIX**: Add callback parameter
5. **Validation Doesn't Fail**: Stores result but continues
   - **TRADE-OFF**: Could raise exception on validation failure
6. **Memory Usage**: Loads entire dataset into memory
   - **FIX**: Add chunking support

---

## Usage Example

```python
# Setup dependencies
reader_factory = DataReaderFactory()
processor = DataProcessor()

# Create use case
use_case = DataIngestionUseCase(reader_factory, processor)

# Execute pipeline
source = DataSource(
    path=Path("data.csv"),
    source_type=DataSourceType.CSV,
)

result = use_case.execute(
    source=source,
    clean=True,        # Run cleaning
    transform=True,    # Run transformation
    validate=True,     # Run validation
)

print(f"Status: {result.status}")
print(f"Steps: {result.processing_steps}")
print(f"Valid: {result.metadata['validation_passed']}")
```

---

## Testing Example

```python
# Mock dependencies
class MockFactory:
    def get_reader(self, source):
        return MockReader()

class MockProcessor:
    def clean(self, data):
        return data
    def transform(self, data):
        return data
    def validate(self, data):
        return True

# Test use case
use_case = DataIngestionUseCase(MockFactory(), MockProcessor())
result = use_case.execute(source)

assert result.status == ProcessingStatus.COMPLETED
assert "cleaned" in result.processing_steps
```

---

**Total Lines**: 95  
**Complexity**: Low  
**Dependencies**: 2 (factory, processor)  
**Testability**: High (DI)  
**Architecture Compliance**: Medium (violates DIP)
