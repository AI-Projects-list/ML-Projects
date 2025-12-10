# Data Ingestion Use Case - Comprehensive Documentation

## File Information
- **Source File**: `src/application/use_cases/data_ingestion.py`
- **Purpose**: Orchestrates the complete data ingestion pipeline from reading to preprocessing
- **Layer**: Application Layer (Use Cases)
- **Pattern**: Use Case Pattern, Facade Pattern, Chain of Responsibility

## Complete Annotated Code

```python
"""Use case for data ingestion and preprocessing."""
# WHAT: Module-level docstring documenting the use case purpose
# WHY: Provides clear documentation at the module level for developers and tools
# HOW: Python docstring convention with triple quotes at module top
# BENEFIT: Improved discoverability, helps IDE auto-completion and documentation generators
# TRADE-OFF: Single line docstring - could be expanded to explain data ingestion workflow

from pathlib import Path
# WHAT: Import Path class from pathlib standard library
# WHY: Modern, object-oriented approach to file system path manipulation
# HOW: Import Path class for cross-platform path handling
# BENEFIT: Cross-platform compatibility (Windows/Linux/Mac), immutable path objects
# TRADE-OFF: Additional import vs using string paths, but benefits outweigh costs

from loguru import logger
# WHAT: Import pre-configured logger from loguru library
# WHY: Provides structured logging with automatic formatting and context
# HOW: Import logger singleton from loguru (configured in infrastructure)
# BENEFIT: Beautiful console output, automatic exception catching, structured logs
# TRADE-OFF: External dependency vs standard logging, but superior developer experience

from src.domain.entities import DataSource, ProcessedData, ProcessingStatus
# WHAT: Import domain entities representing data states in the pipeline
# WHY: Use domain models to enforce business rules and maintain clean architecture
# HOW: Import three entity classes: DataSource (input), ProcessedData (output), ProcessingStatus (state)
# BENEFIT: Type safety, business logic encapsulation, separation of concerns
# TRADE-OFF: More classes to manage vs simple dictionaries, but enforces domain integrity

from src.domain.repositories import IDataProcessor
# WHAT: Import interface (abstract base class) for data processing
# WHY: Dependency inversion - depend on abstractions not concrete implementations
# HOW: Import IDataProcessor protocol defining clean/transform/validate contract
# BENEFIT: Testability (easy to mock), flexibility (swap implementations), loose coupling
# TRADE-OFF: Additional abstraction layer vs direct implementation, but enables SOLID principles

from src.infrastructure.data_readers.factory import DataReaderFactory
# WHAT: Import factory class that creates appropriate data readers
# WHY: Factory Pattern to encapsulate reader creation logic based on file type
# HOW: Import DataReaderFactory which creates CSV/PDF/Text/Scanned PDF readers
# BENEFIT: Single point of reader creation, easy to add new reader types, encapsulated logic
# TRADE-OFF: Additional indirection vs creating readers directly, but improves maintainability


class DataIngestionUseCase:
    # WHAT: Use case class orchestrating the data ingestion pipeline
    # WHY: Application layer use case pattern to coordinate domain objects and infrastructure
    # HOW: Class encapsulating the business workflow for ingesting data
    # BENEFIT: Single responsibility (ingestion only), testable, reusable across interfaces
    # TRADE-OFF: Additional class vs procedural function, but provides better organization

    """Handles the complete data ingestion pipeline."""
    # WHAT: Class-level docstring describing the use case responsibility
    # WHY: Documents the class purpose for developers and documentation tools
    # HOW: Concise single-line docstring stating primary responsibility
    # BENEFIT: Clear understanding of class purpose, better IDE support
    # TRADE-OFF: Brief description - could expand to describe pipeline steps

    def __init__(
        # WHAT: Constructor method signature using multi-line parameter layout
        # WHY: Dependency injection pattern for testability and flexibility
        # HOW: Accept dependencies as constructor parameters (not created internally)
        # BENEFIT: Testable (inject mocks), flexible (swap implementations), explicit dependencies
        # TRADE-OFF: More verbose than creating dependencies internally, but enables testing

        self,
        # WHAT: Reference to the instance being initialized
        # WHY: Python requirement for instance methods
        # HOW: First parameter of instance methods by convention
        # BENEFIT: Access to instance attributes and methods
        # TRADE-OFF: None - required by Python

        reader_factory: DataReaderFactory,
        # WHAT: Injected dependency for creating appropriate data readers
        # WHY: Factory pattern to create CSV/PDF/Text readers based on file type
        # HOW: Type-hinted parameter expecting DataReaderFactory instance
        # BENEFIT: Testable (inject mock factory), single reader creation point, extensible
        # TRADE-OFF: Additional parameter vs hardcoded factory, but enables dependency injection

        processor: IDataProcessor,
        # WHAT: Injected dependency for data cleaning, transformation, and validation
        # WHY: Interface dependency for loose coupling and testability
        # HOW: Type-hinted parameter expecting IDataProcessor implementation
        # BENEFIT: Testable (inject mock), swappable implementations (different processors)
        # TRADE-OFF: Abstraction overhead vs concrete dependency, but enables SOLID principles

    ):
        """
        Initialize data ingestion use case.

        Args:
            reader_factory: Factory for creating data readers
            processor: Data processor for cleaning and transformation
        """
        # WHAT: Multi-line docstring documenting constructor parameters
        # WHY: Provides clear documentation for dependency injection requirements
        # HOW: Google-style docstring format with Args section
        # BENEFIT: IDE auto-complete support, clear parameter documentation
        # TRADE-OFF: Verbose documentation vs brief comments, but improves usability

        self.reader_factory = reader_factory
        # WHAT: Store reader factory as instance attribute
        # WHY: Make factory available throughout use case lifecycle
        # HOW: Assign injected factory to instance variable
        # BENEFIT: Access factory in execute method without passing as parameter
        # TRADE-OFF: State management (mutable instance) vs stateless function, but appropriate for use case

        self.processor = processor
        # WHAT: Store processor as instance attribute
        # WHY: Make processor available for clean/transform/validate operations
        # HOW: Assign injected processor to instance variable
        # BENEFIT: Reuse processor across multiple executions, access in execute method
        # TRADE-OFF: Holds reference to processor (memory) vs creating on-demand, but enables reuse

    def execute(
        # WHAT: Main execution method signature for the ingestion pipeline
        # WHY: Execute is standard naming for use case entry points (Command pattern)
        # HOW: Public method coordinating the complete ingestion workflow
        # BENEFIT: Clear entry point, standardized interface across use cases
        # TRADE-OFF: Generic name "execute" vs specific name like "ingest_data", but consistent pattern

        self,
        # WHAT: Instance reference for accessing factory and processor
        # WHY: Required for Python instance methods
        # HOW: Access self.reader_factory and self.processor within method
        # BENEFIT: Access to injected dependencies
        # TRADE-OFF: None - required by Python

        source: DataSource,
        # WHAT: Domain entity representing the data source to ingest
        # WHY: Encapsulates source metadata (path, type, options) in domain object
        # HOW: Type-hinted parameter expecting DataSource entity
        # BENEFIT: Type safety, domain validation, rich metadata support
        # TRADE-OFF: Domain object vs simple path string, but provides better abstraction

        clean: bool = True,
        # WHAT: Flag to enable/disable data cleaning step
        # WHY: Allow flexible pipeline configuration - sometimes raw data is needed
        # HOW: Boolean parameter with default True (cleaning is recommended)
        # BENEFIT: Flexibility to skip cleaning, faster processing when not needed
        # TRADE-OFF: Additional parameter vs always cleaning, but provides workflow control

        transform: bool = True,
        # WHAT: Flag to enable/disable data transformation step
        # WHY: Allow flexible pipeline configuration - transformations may vary by use case
        # HOW: Boolean parameter with default True (transformation is recommended)
        # BENEFIT: Flexibility to skip transformation, control over preprocessing steps
        # TRADE-OFF: Additional parameter vs always transforming, but enables customization

        validate: bool = True,
        # WHAT: Flag to enable/disable data validation step
        # WHY: Allow flexible pipeline configuration - validation may be expensive
        # HOW: Boolean parameter with default True (validation is recommended)
        # BENEFIT: Flexibility to skip validation, performance optimization option
        # TRADE-OFF: Additional parameter vs always validating, but balances quality vs speed

    ) -> ProcessedData:
        # WHAT: Return type annotation specifying ProcessedData entity
        # WHY: Type safety and IDE auto-completion for return value
        # HOW: Arrow notation indicating method returns ProcessedData instance
        # BENEFIT: Type checking, clear contract, IDE support
        # TRADE-OFF: Annotation overhead vs untyped return, but improves reliability

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
        # WHAT: Comprehensive docstring documenting execute method
        # WHY: Clear documentation of pipeline parameters and return value
        # HOW: Google-style docstring with Args and Returns sections
        # BENEFIT: IDE support, clear API documentation, maintainability
        # TRADE-OFF: Verbose vs brief comments, but essential for public API

        logger.info(f"Starting data ingestion for {source.path}")
        # WHAT: Log informational message about pipeline start
        # WHY: Observability - track pipeline execution and debug issues
        # HOW: Use logger.info with f-string including source path
        # BENEFIT: Troubleshooting, monitoring, audit trail
        # TRADE-OFF: I/O overhead vs silent execution, but critical for production systems

        # Read data
        # WHAT: Comment indicating the data reading step
        # WHY: Code organization - mark distinct pipeline phase
        # HOW: Single-line comment above related code block
        # BENEFIT: Improved readability, clear pipeline structure
        # TRADE-OFF: Additional comment vs self-documenting code, but helpful for workflow clarity

        reader = self.reader_factory.get_reader(source)
        # WHAT: Create appropriate data reader using factory pattern
        # WHY: Factory encapsulates reader selection logic based on source type
        # HOW: Call factory's get_reader method passing source entity
        # BENEFIT: Automatic reader selection (CSV/PDF/Text/Scanned), extensible to new types
        # TRADE-OFF: Factory indirection vs direct reader instantiation, but improves maintainability

        raw_data = reader.read(source)
        # WHAT: Read data from source into raw DataFrame
        # WHY: Get data into memory for processing
        # HOW: Call reader's read method which returns pandas DataFrame
        # BENEFIT: Data is now accessible for processing, reader handles file format complexity
        # TRADE-OFF: Memory usage (loads full dataset) vs streaming, but simpler for ML workflows

        # Create ProcessedData object
        # WHAT: Comment indicating creation of domain entity
        # WHY: Mark transition from raw data to domain model
        # HOW: Single-line comment above entity instantiation
        # BENEFIT: Clear pipeline step separation
        # TRADE-OFF: Additional comment vs self-documenting code, but marks important transition

        processed_data = ProcessedData(
            # WHAT: Instantiate ProcessedData domain entity
            # WHY: Wrap raw data in rich domain object with metadata and status tracking
            # HOW: Call ProcessedData constructor with initial data, source, and status
            # BENEFIT: Business logic encapsulation, metadata tracking, status management
            # TRADE-OFF: Domain object overhead vs simple dict, but provides structure

            data=raw_data,
            # WHAT: Set data attribute to raw DataFrame from reader
            # WHY: Initialize entity with the data to be processed
            # HOW: Pass raw_data as named parameter
            # BENEFIT: Data is now managed within domain entity
            # TRADE-OFF: Additional wrapping vs direct DataFrame, but enables domain logic

            source=source,
            # WHAT: Store reference to original data source
            # WHY: Maintain lineage - track where data came from
            # HOW: Pass source entity as named parameter
            # BENEFIT: Traceability, debugging, audit trail
            # TRADE-OFF: Additional memory (source reference) vs standalone data, but critical for lineage

            status=ProcessingStatus.IN_PROGRESS,
            # WHAT: Set initial processing status to IN_PROGRESS
            # WHY: Track pipeline state for monitoring and error handling
            # HOW: Use ProcessingStatus enum value
            # BENEFIT: Clear status tracking, type-safe state management
            # TRADE-OFF: Status overhead vs statusless processing, but enables monitoring

        )

        try:
            # WHAT: Begin try block for exception handling
            # WHY: Graceful error handling - catch processing failures and update status
            # HOW: Try-except block wrapping processing steps
            # BENEFIT: Robust error handling, status tracking on failure, no silent failures
            # TRADE-OFF: Error handling overhead vs uncaught exceptions, but critical for production

            # Clean data
            # WHAT: Comment marking the data cleaning step
            # WHY: Organize pipeline into distinct phases
            # HOW: Single-line comment above cleaning logic
            # BENEFIT: Clear code structure, easy to navigate
            # TRADE-OFF: Additional comment vs self-documenting code, but improves clarity

            if clean:
                # WHAT: Conditional execution of cleaning based on parameter
                # WHY: Allow skipping cleaning step when raw data is needed
                # HOW: Check clean boolean flag
                # BENEFIT: Flexible pipeline configuration, performance optimization
                # TRADE-OFF: Conditional logic vs always clean, but provides user control

                logger.info("Cleaning data...")
                # WHAT: Log cleaning step initiation
                # WHY: Observability - track which steps are executed
                # HOW: Info-level log message
                # BENEFIT: Debugging, monitoring, understanding pipeline flow
                # TRADE-OFF: Logging overhead vs silent execution, but aids troubleshooting

                processed_data.data = self.processor.clean(processed_data.data)
                # WHAT: Apply data cleaning and update entity's data attribute
                # WHY: Remove nulls, outliers, duplicates, invalid values
                # HOW: Call processor's clean method, reassign result to data attribute
                # BENEFIT: Higher quality data, remove noise, improve model performance
                # TRADE-OFF: Potential data loss (removed records) vs keeping all data, but improves quality

                processed_data.add_processing_step("cleaned")
                # WHAT: Record cleaning step in entity's processing history
                # WHY: Audit trail - track which transformations were applied
                # HOW: Call entity method to append "cleaned" to steps list
                # BENEFIT: Reproducibility, debugging, understanding data lineage
                # TRADE-OFF: Memory for history vs no tracking, but critical for ML workflows

            # Transform data
            # WHAT: Comment marking the transformation step
            # WHY: Organize pipeline phases clearly
            # HOW: Single-line comment above transformation logic
            # BENEFIT: Code readability, clear workflow structure
            # TRADE-OFF: Additional comment vs code alone, but improves navigation

            if transform:
                # WHAT: Conditional execution of transformation based on parameter
                # WHY: Allow skipping transformation when not needed
                # HOW: Check transform boolean flag
                # BENEFIT: Flexible pipeline, control over preprocessing
                # TRADE-OFF: Conditional complexity vs always transform, but enables customization

                logger.info("Transforming data...")
                # WHAT: Log transformation step initiation
                # WHY: Track pipeline execution progress
                # HOW: Info-level log message
                # BENEFIT: Observability, debugging, execution transparency
                # TRADE-OFF: Logging I/O vs silent processing, but improves monitoring

                processed_data.data = self.processor.transform(processed_data.data)
                # WHAT: Apply data transformations and update entity
                # WHY: Feature engineering, scaling, encoding, etc.
                # HOW: Call processor's transform method, reassign to data attribute
                # BENEFIT: ML-ready features, improved model performance
                # TRADE-OFF: Data modification (not reversible) vs keeping original, but necessary for ML

                processed_data.add_processing_step("transformed")
                # WHAT: Record transformation in processing history
                # WHY: Track applied transformations for reproducibility
                # HOW: Append "transformed" to entity's steps list
                # BENEFIT: Audit trail, debugging, pipeline transparency
                # TRADE-OFF: Memory for tracking vs no history, but essential for lineage

            # Validate data
            # WHAT: Comment marking the validation step
            # WHY: Organize validation phase separately
            # HOW: Single-line comment above validation logic
            # BENEFIT: Clear pipeline structure
            # TRADE-OFF: Comment overhead vs code only, but aids readability

            if validate:
                # WHAT: Conditional execution of validation based on parameter
                # WHY: Allow skipping validation for performance or when not needed
                # HOW: Check validate boolean flag
                # BENEFIT: Flexible pipeline, performance control
                # TRADE-OFF: Conditional logic vs always validate, but balances quality vs speed

                logger.info("Validating data...")
                # WHAT: Log validation step initiation
                # WHY: Track validation execution
                # HOW: Info-level log message
                # BENEFIT: Observability, debugging support
                # TRADE-OFF: Logging overhead vs silent validation, but improves tracking

                is_valid = self.processor.validate(processed_data.data)
                # WHAT: Validate data quality and store boolean result
                # WHY: Ensure data meets quality requirements before downstream use
                # HOW: Call processor's validate method returning True/False
                # BENEFIT: Early error detection, quality gates, prevent bad data propagation
                # TRADE-OFF: Validation time vs skipping checks, but critical for data quality

                processed_data.metadata["validation_passed"] = is_valid
                # WHAT: Store validation result in entity's metadata dictionary
                # WHY: Record validation outcome for downstream consumers
                # HOW: Set metadata key with boolean validation result
                # BENEFIT: Consumers can check validation status, audit trail
                # TRADE-OFF: Additional metadata vs ignoring result, but enables quality tracking

                processed_data.add_processing_step("validated")
                # WHAT: Record validation in processing history
                # WHY: Track that validation was performed
                # HOW: Append "validated" to steps list
                # BENEFIT: Complete audit trail of pipeline steps
                # TRADE-OFF: Memory for history vs no tracking, but important for transparency

            # Mark as completed
            # WHAT: Comment indicating status update step
            # WHY: Mark successful pipeline completion
            # HOW: Single-line comment above status update
            # BENEFIT: Clear code intent
            # TRADE-OFF: Comment overhead vs code alone, but improves clarity

            processed_data.mark_completed()
            # WHAT: Update entity status to COMPLETED
            # WHY: Signal successful pipeline execution
            # HOW: Call entity method to set status enum
            # BENEFIT: Status tracking, downstream consumers know data is ready
            # TRADE-OFF: Status management overhead vs no tracking, but enables monitoring

            logger.info("Data ingestion completed successfully")
            # WHAT: Log successful pipeline completion
            # WHY: Observability - confirm success for monitoring/debugging
            # HOW: Info-level log message
            # BENEFIT: Clear success indication, audit trail, troubleshooting
            # TRADE-OFF: Logging I/O vs silent success, but important for production

        except Exception as e:
            # WHAT: Catch any exception during processing
            # WHY: Graceful error handling - don't crash, update status
            # HOW: Except clause catching all Exception types
            # BENEFIT: Robust error handling, status tracking on failure
            # TRADE-OFF: Catching broad Exception vs specific exceptions, but ensures status update

            logger.error(f"Data ingestion failed: {e}")
            # WHAT: Log error with exception details
            # WHY: Record failure for debugging and monitoring
            # HOW: Error-level log with f-string including exception
            # BENEFIT: Troubleshooting, error tracking, alerting
            # TRADE-OFF: Logging overhead vs silent failure, but critical for debugging

            processed_data.mark_failed()
            # WHAT: Update entity status to FAILED
            # WHY: Signal pipeline failure to downstream consumers
            # HOW: Call entity method to set FAILED status
            # BENEFIT: Clear failure indication, prevent using invalid data
            # TRADE-OFF: Status update overhead vs no tracking, but essential for reliability

            raise
            # WHAT: Re-raise the caught exception
            # WHY: Allow caller to handle exception (don't swallow it)
            # HOW: Raise statement without arguments re-raises current exception
            # BENEFIT: Caller can catch and handle, don't hide errors, maintain stack trace
            # TRADE-OFF: Exception propagation vs returning error status, but transparent error handling

        return processed_data
        # WHAT: Return the processed data entity
        # WHY: Provide result to caller for downstream processing
        # HOW: Return ProcessedData instance with data, metadata, status, history
        # BENEFIT: Rich result object with data and metadata, clear API
        # TRADE-OFF: Domain object vs simple DataFrame, but provides comprehensive information
```

---

## Design Patterns Used

### 1. **Use Case Pattern** (Application Layer)
- **Purpose**: Encapsulates business workflow/application logic
- **Implementation**: `DataIngestionUseCase` class with `execute` method
- **Benefits**: Single responsibility, testable, reusable across different interfaces (CLI, API, UI)
- **Trade-offs**: Additional abstraction vs direct implementation

### 2. **Dependency Injection Pattern**
- **Purpose**: Invert control - depend on abstractions injected via constructor
- **Implementation**: Constructor accepts `reader_factory` and `processor` dependencies
- **Benefits**: Testability (inject mocks), flexibility (swap implementations), explicit dependencies
- **Trade-offs**: More verbose initialization vs creating dependencies internally

### 3. **Factory Pattern**
- **Purpose**: Encapsulate object creation logic
- **Implementation**: `DataReaderFactory` creates appropriate readers based on file type
- **Benefits**: Single point of reader creation, extensible to new types, hides creation complexity
- **Trade-offs**: Additional indirection vs direct instantiation

### 4. **Facade Pattern**
- **Purpose**: Provide simple interface to complex subsystem
- **Implementation**: `execute` method hides complexity of read → clean → transform → validate
- **Benefits**: Simple API for clients, coordinates multiple objects, clear workflow
- **Trade-offs**: Less control for advanced users vs exposing all details

### 5. **Chain of Responsibility**
- **Purpose**: Pass request through chain of processing steps
- **Implementation**: Sequential steps: clean → transform → validate
- **Benefits**: Flexible pipeline (enable/disable steps), easy to add new steps, single responsibility per step
- **Trade-offs**: Sequential processing (not parallel) vs complex orchestration

---

## Pros & Cons

### Pros ✅

1. **Clean Architecture**
   - Clear separation: domain entities, use case orchestration, infrastructure injection
   - Follows SOLID principles (SRP, DIP, OCP)

2. **Flexible Pipeline**
   - Configurable steps via boolean flags
   - Can skip cleaning, transformation, or validation as needed
   - Easy to add new processing steps

3. **Excellent Observability**
   - Comprehensive logging at each step
   - Status tracking (IN_PROGRESS → COMPLETED/FAILED)
   - Processing history (audit trail of applied steps)

4. **Robust Error Handling**
   - Try-except catches processing failures
   - Updates status to FAILED on error
   - Re-raises exception for caller handling

5. **Testability**
   - Dependencies injected via constructor (easy to mock)
   - Clear inputs and outputs
   - No hidden dependencies or global state

6. **Rich Domain Model**
   - `ProcessedData` entity with data, metadata, status, history
   - Type safety with type hints
   - Business logic encapsulated in entities

### Cons ❌

1. **Memory Usage**
   - Loads entire dataset into memory (not streaming)
   - May struggle with very large datasets (> available RAM)
   - Holds source reference (additional memory)

2. **Sequential Processing**
   - Steps executed sequentially (not parallel)
   - Cannot leverage multi-core for cleaning/transformation
   - Slower for large datasets

3. **Limited Granularity**
   - All-or-nothing flags (clean: bool vs clean_nulls, clean_outliers separately)
   - Cannot configure individual cleaning/transformation operations
   - Less control for advanced users

4. **Coupling to Processor Interface**
   - Tightly coupled to `IDataProcessor` contract (clean/transform/validate methods)
   - Changing processor interface requires updating use case
   - Cannot easily swap to different processing paradigm

5. **No Progress Tracking**
   - Logs steps but no percentage completion or ETA
   - Cannot track progress for long-running operations
   - No callback mechanism for UI updates

6. **Limited Metadata**
   - Only stores validation result in metadata
   - Could track more details (rows processed, errors encountered, duration)
   - No intermediate state persistence

---

## Usage Examples

### Example 1: Basic Data Ingestion
```python
from pathlib import Path
from src.domain.entities import DataSource
from src.infrastructure.data_readers.factory import DataReaderFactory
from src.infrastructure.processing.data_processor import DataProcessor
from src.application.use_cases.data_ingestion import DataIngestionUseCase

# Setup dependencies
reader_factory = DataReaderFactory()
processor = DataProcessor()

# Create use case
use_case = DataIngestionUseCase(reader_factory, processor)

# Execute with all steps
source = DataSource(path=Path("data/sales.csv"), data_type="csv")
result = use_case.execute(source)

print(f"Status: {result.status}")
print(f"Shape: {result.data.shape}")
print(f"Steps: {result.processing_steps}")
# Output:
# Status: ProcessingStatus.COMPLETED
# Shape: (1000, 10)
# Steps: ['cleaned', 'transformed', 'validated']
```

### Example 2: Skip Validation for Speed
```python
# Skip validation to improve performance
result = use_case.execute(
    source,
    clean=True,
    transform=True,
    validate=False  # Skip expensive validation
)

print(f"Steps: {result.processing_steps}")
# Output: ['cleaned', 'transformed']
```

### Example 3: Raw Data Only (No Processing)
```python
# Get raw data without any processing
result = use_case.execute(
    source,
    clean=False,
    transform=False,
    validate=False
)

# Only reading step was performed
print(f"Steps: {result.processing_steps}")
# Output: []
```

### Example 4: Error Handling
```python
try:
    source = DataSource(path=Path("data/corrupted.csv"), data_type="csv")
    result = use_case.execute(source)
except Exception as e:
    print(f"Pipeline failed: {e}")
    # Check status even on failure
    if result.status == ProcessingStatus.FAILED:
        print("Data marked as failed - safe to retry or alert")
```

### Example 5: Testing with Mocks
```python
from unittest.mock import Mock
import pandas as pd

# Create mock dependencies
mock_factory = Mock()
mock_processor = Mock()
mock_reader = Mock()

# Setup mock behaviors
mock_factory.get_reader.return_value = mock_reader
mock_reader.read.return_value = pd.DataFrame({"col": [1, 2, 3]})
mock_processor.clean.return_value = pd.DataFrame({"col": [1, 2]})
mock_processor.transform.return_value = pd.DataFrame({"col_transformed": [10, 20]})
mock_processor.validate.return_value = True

# Test use case with mocks
use_case = DataIngestionUseCase(mock_factory, mock_processor)
source = DataSource(path=Path("test.csv"), data_type="csv")
result = use_case.execute(source)

# Assert expectations
assert result.status == ProcessingStatus.COMPLETED
assert result.metadata["validation_passed"] == True
assert len(result.processing_steps) == 3
mock_processor.clean.assert_called_once()
```

---

## Key Takeaways

### What This Use Case Does
Orchestrates the complete data ingestion pipeline: reading → cleaning → transformation → validation, with configurable steps and comprehensive status tracking.

### Why This Architecture
- **Clean Architecture**: Separates use case (application layer) from domain entities and infrastructure
- **Testability**: Dependencies injected, easy to mock
- **Flexibility**: Configurable pipeline steps via flags

### How It Works
1. Accept `DataSource` entity and configuration flags
2. Use factory to get appropriate reader
3. Read raw data into DataFrame
4. Optionally clean, transform, validate
5. Track status and processing history
6. Return rich `ProcessedData` entity

### Benefits
- **Robust**: Comprehensive error handling with status tracking
- **Observable**: Extensive logging at each step
- **Reusable**: Can be called from CLI, API, web UI
- **Maintainable**: Clear structure, single responsibility

### Trade-offs
- **Memory**: Loads entire dataset (not streaming)
- **Speed**: Sequential processing (not parallel)
- **Granularity**: Boolean flags vs fine-grained configuration

---

## Related Files
- **Domain Entities**: `src/domain/entities.py` - `DataSource`, `ProcessedData`, `ProcessingStatus`
- **Repository Interface**: `src/domain/repositories.py` - `IDataProcessor`
- **Factory**: `src/infrastructure/data_readers/factory.py` - `DataReaderFactory`
- **Processor Implementation**: `src/infrastructure/processing/data_processor.py`
- **Readers**: `src/infrastructure/data_readers/` - CSV, PDF, Text, Scanned PDF readers

---

*This documentation provides comprehensive line-by-line annotations with WHAT/WHY/HOW/BENEFIT/TRADE-OFF analysis for each line of code, design patterns explanation, pros & cons analysis, and practical usage examples.*
