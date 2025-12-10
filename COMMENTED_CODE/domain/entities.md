# Domain Entities - Detailed Code Documentation

**File**: `src/domain/entities.py`  
**Purpose**: Define core business entities for the ML pipeline system  
**Layer**: Domain (Core Business Logic)  
**Dependencies**: None (Pure domain logic)

---

## Overview

This file contains the core business entities that represent the fundamental concepts in our machine learning pipeline. These entities are **framework-independent** and contain the essential business logic.

---

## Complete Code with Line-by-Line Comments

```python
"""Domain entities representing core business objects."""
# Module docstring - describes the purpose of this file
# WHY: Provides clear documentation at the module level
# WHAT: This module contains domain entities (business objects)
# HOW: Uses dataclasses and enums for clean, maintainable code

from dataclasses import dataclass, field
# WHAT: Import dataclass decorator and field function
# WHY: Reduces boilerplate code for creating classes
# HOW: Automatically generates __init__, __repr__, __eq__ methods
# BENEFIT: Less code, fewer bugs, more readable
# TRADE-OFF: Slightly less control than manual implementation

from datetime import datetime
# WHAT: Import datetime for timestamps
# WHY: Track when entities are created/modified
# HOW: Used with field(default_factory=datetime.now)
# BENEFIT: Automatic timestamping without manual intervention
# TRADE-OFF: Naive datetime (no timezone awareness by default)

from enum import Enum
# WHAT: Import Enum base class
# WHY: Create type-safe constants/categories
# HOW: Classes inherit from Enum
# BENEFIT: Prevents typos, enables IDE autocomplete, type checking
# TRADE-OFF: Slightly more verbose than string constants

from typing import Any, Dict, List, Optional
# WHAT: Import type hint utilities
# WHY: Enable static type checking and better IDE support
# HOW: Annotate function parameters and return types
# BENEFIT: Catches type errors early, self-documenting code
# TRADE-OFF: Not enforced at runtime (need mypy for checking)

import pandas as pd
# WHAT: Import pandas library
# WHY: Industry-standard data manipulation library
# HOW: Used for DataFrame type hints
# BENEFIT: Powerful data structures, wide ecosystem support
# TRADE-OFF: Heavy dependency, memory-intensive


class DataSourceType(Enum):
    """Types of data sources supported."""
    # WHAT: Enumeration of supported data source types
    # WHY: Type-safe way to specify data formats
    # HOW: Each constant maps to a string value
    # BENEFIT: Prevents invalid source types, easy to extend
    
    CSV = "csv"
    # WHAT: CSV file format
    # WHY: Most common data format in ML/data science
    # BENEFIT: Human-readable, universally supported
    # USE CASE: Structured tabular data
    
    TXT = "txt"
    # WHAT: Plain text file format
    # WHY: Support unstructured text data
    # BENEFIT: Simple, portable
    # USE CASE: Text documents, logs
    
    PDF = "pdf"
    # WHAT: PDF document format
    # WHY: Support structured PDF documents
    # BENEFIT: Preserves formatting
    # USE CASE: Reports, structured documents
    
    PDF_SCAN = "pdf_scan"
    # WHAT: Scanned PDF (requires OCR)
    # WHY: Support image-based PDFs
    # BENEFIT: Can extract text from scanned documents
    # USE CASE: Scanned forms, legacy documents
    # TRADE-OFF: Slower processing, requires Tesseract
    
    DATAFRAME = "dataframe"
    # WHAT: Direct pandas DataFrame
    # WHY: Support in-memory data
    # BENEFIT: No file I/O overhead
    # USE CASE: Already-loaded data, testing


class ProcessingStatus(Enum):
    """Status of data processing."""
    # WHAT: Enumeration of processing pipeline states
    # WHY: Track where data is in the processing workflow
    # HOW: Transitions: PENDING → IN_PROGRESS → COMPLETED/FAILED
    # BENEFIT: Clear state machine, prevents invalid states
    
    PENDING = "pending"
    # WHAT: Initial state before processing starts
    # WHY: Indicates data is queued but not started
    # USE CASE: Newly created ProcessedData objects
    
    IN_PROGRESS = "in_progress"
    # WHAT: Currently being processed
    # WHY: Indicates active processing
    # BENEFIT: Enables progress tracking
    # USE CASE: During data cleaning/transformation
    
    COMPLETED = "completed"
    # WHAT: Successfully finished processing
    # WHY: Indicates data is ready for use
    # USE CASE: After successful pipeline completion
    
    FAILED = "failed"
    # WHAT: Processing encountered errors
    # WHY: Indicates processing failure
    # BENEFIT: Explicit failure handling
    # USE CASE: When exceptions occur during processing


@dataclass
class DataSource:
    """Represents a data source."""
    # WHAT: Entity representing the origin of data
    # WHY: Encapsulates all information about where data comes from
    # HOW: Immutable-ish dataclass with metadata
    # BENEFIT: Single source of truth for data origin
    # PATTERN: Entity pattern from Domain-Driven Design
    
    source_type: DataSourceType
    # WHAT: Type of data source (CSV, PDF, etc.)
    # WHY: Determines which reader to use
    # HOW: Must be one of DataSourceType enum values
    # BENEFIT: Type-safe, prevents invalid source types
    # REQUIRED: Yes (no default value)
    
    path: str
    # WHAT: File path or data location
    # WHY: Specifies where to read data from
    # HOW: Can be file path, URL, or identifier
    # BENEFIT: Flexible, supports various locations
    # TRADE-OFF: String type doesn't validate existence
    # IMPROVEMENT: Could use Path type from pathlib
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    # WHAT: Additional metadata about the source
    # WHY: Store extra information without schema changes
    # HOW: Uses field(default_factory=dict) to avoid mutable default
    # BENEFIT: Extensible, no database migrations needed
    # TRADE-OFF: No schema validation, can be misused
    # USE CASE: File size, encoding, custom tags
    
    created_at: datetime = field(default_factory=datetime.now)
    # WHAT: Timestamp when source was created
    # WHY: Track when data source was registered
    # HOW: Auto-generated using datetime.now
    # BENEFIT: Automatic, no manual intervention
    # TRADE-OFF: Naive datetime (no timezone info)
    # IMPROVEMENT: Use datetime.now(timezone.utc)


@dataclass
class ProcessedData:
    """Represents processed data ready for analysis."""
    # WHAT: Entity for data after preprocessing pipeline
    # WHY: Wraps cleaned/transformed data with metadata
    # HOW: Contains DataFrame plus processing history
    # BENEFIT: Rich domain model, audit trail
    # PATTERN: Entity + Value Object pattern
    
    data: pd.DataFrame
    # WHAT: The actual processed data
    # WHY: Store the result of processing pipeline
    # HOW: Pandas DataFrame with cleaned/transformed data
    # BENEFIT: Powerful data structure, ML-ready
    # TRADE-OFF: Memory intensive, not serializable
    # USE CASE: Training data, test data
    
    source: DataSource
    # WHAT: Reference to original data source
    # WHY: Maintain data lineage/traceability
    # HOW: Stores the DataSource entity
    # BENEFIT: Know where data came from
    # USE CASE: Auditing, debugging, reproducibility
    
    processing_steps: List[str] = field(default_factory=list)
    # WHAT: History of transformations applied
    # WHY: Track what was done to the data
    # HOW: Append step names as processing occurs
    # BENEFIT: Audit trail, reproducibility, debugging
    # TRADE-OFF: String-based, no structured metadata
    # USE CASE: ["cleaned", "transformed", "validated"]
    
    status: ProcessingStatus = ProcessingStatus.PENDING
    # WHAT: Current processing state
    # WHY: Track pipeline progress
    # HOW: Enum value, defaults to PENDING
    # BENEFIT: Type-safe state management
    # USE CASE: UI progress indicators, error handling
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    # WHAT: Additional processing metadata
    # WHY: Store quality metrics, warnings, etc.
    # HOW: Dictionary with flexible schema
    # BENEFIT: Extensible without code changes
    # TRADE-OFF: Untyped, no validation
    # USE CASE: validation_passed=True, quality_score=0.95
    
    processed_at: Optional[datetime] = None
    # WHAT: When processing completed
    # WHY: Track processing duration
    # HOW: None until processing finishes
    # BENEFIT: Performance monitoring
    # USE CASE: Calculate processing time
    
    def mark_completed(self) -> None:
        """Mark data processing as completed."""
        # WHAT: Method to transition to completed state
        # WHY: Encapsulates state change logic
        # HOW: Sets status and timestamp
        # BENEFIT: Consistent state transitions
        # PATTERN: Command method
        
        self.status = ProcessingStatus.COMPLETED
        # WHAT: Set status to COMPLETED
        # WHY: Indicate successful processing
        # HOW: Direct enum assignment
        
        self.processed_at = datetime.now()
        # WHAT: Record completion timestamp
        # WHY: Track when processing finished
        # HOW: Auto-generate current time
    
    def mark_failed(self) -> None:
        """Mark data processing as failed."""
        # WHAT: Method to transition to failed state
        # WHY: Handle processing errors explicitly
        # HOW: Sets status and timestamp
        # TRADE-OFF: Doesn't capture error details
        # IMPROVEMENT: Add error parameter
        
        self.status = ProcessingStatus.FAILED
        # WHAT: Set status to FAILED
        # WHY: Indicate processing failure
        
        self.processed_at = datetime.now()
        # WHAT: Record failure timestamp
        # WHY: Track when failure occurred
    
    def add_processing_step(self, step: str) -> None:
        """Add a processing step to the history."""
        # WHAT: Append step to processing history
        # WHY: Build audit trail
        # HOW: Simple list append
        # BENEFIT: Chronological order preserved
        # TRADE-OFF: String-based, no metadata per step
        
        self.processing_steps.append(step)
        # WHAT: Add step name to list
        # WHY: Record what happened
        # USE CASE: "cleaned", "encoded_categoricals", "scaled"


@dataclass
class EDAReport:
    """Represents exploratory data analysis report."""
    # WHAT: Entity containing EDA results
    # WHY: Structure analysis findings
    # HOW: Aggregates statistics, correlations, insights
    # BENEFIT: Organized EDA output
    # USE CASE: Data understanding, quality assessment
    
    data_shape: tuple
    # WHAT: Dataset dimensions (rows, columns)
    # WHY: Quick overview of dataset size
    # HOW: Stores (n_rows, n_cols)
    # BENEFIT: Immediate size information
    # USE CASE: (1000, 50) = 1000 rows, 50 columns
    
    column_types: Dict[str, str]
    # WHAT: Column name → data type mapping
    # WHY: Schema documentation
    # HOW: {"col1": "int64", "col2": "float64"}
    # BENEFIT: Type information at a glance
    # USE CASE: Understand data types
    
    missing_values: Dict[str, int]
    # WHAT: Column name → missing count
    # WHY: Data quality insight
    # HOW: {"col1": 5, "col2": 0}
    # BENEFIT: Identify problematic columns
    # USE CASE: Decide imputation strategy
    
    statistics: Dict[str, Any]
    # WHAT: Statistical summaries
    # WHY: Descriptive statistics
    # HOW: Flexible dictionary structure
    # BENEFIT: Store any statistic
    # TRADE-OFF: Untyped, no validation
    # USE CASE: mean, median, std, min, max
    
    correlations: Optional[pd.DataFrame] = None
    # WHAT: Correlation matrix
    # WHY: Feature relationships
    # HOW: DataFrame of correlation coefficients
    # BENEFIT: Identify multicollinearity
    # TRADE-OFF: Optional, can be large
    # USE CASE: Feature selection
    
    visualizations: Dict[str, str] = field(default_factory=dict)
    # WHAT: Plot name → file path mapping
    # WHY: Reference to saved visualizations
    # HOW: {"histogram": "/path/to/hist.png"}
    # BENEFIT: Separate storage from metadata
    # USE CASE: Link to generated plots
    
    insights: List[str] = field(default_factory=list)
    # WHAT: Human-readable findings
    # WHY: Summarize key observations
    # HOW: List of insight strings
    # BENEFIT: Actionable information
    # TRADE-OFF: String-based, no structure
    # USE CASE: ["10% missing in col1", "High correlation detected"]
    
    generated_at: datetime = field(default_factory=datetime.now)
    # WHAT: Report generation timestamp
    # WHY: Track when analysis was done
    # HOW: Auto-generated timestamp
    # BENEFIT: Report versioning


@dataclass
class ModelConfig:
    """Configuration for ML models."""
    # WHAT: Value object for model parameters
    # WHY: Centralize model configuration
    # HOW: Stores model type and hyperparameters
    # BENEFIT: Reusable, version-controlled configs
    # PATTERN: Builder pattern configuration
    
    model_type: str
    # WHAT: Type of ML model
    # WHY: Determine which algorithm to use
    # HOW: String identifier
    # TRADE-OFF: Should be Enum for type safety
    # USE CASE: "random_forest", "logistic_regression"
    # IMPROVEMENT: Create ModelType enum
    
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    # WHAT: Model-specific hyperparameters
    # WHY: Configure algorithm behavior
    # HOW: Flexible dictionary
    # BENEFIT: Works with any model
    # TRADE-OFF: No validation, easy to mistype
    # USE CASE: {"n_estimators": 100, "max_depth": 10}
    
    target_column: Optional[str] = None
    # WHAT: Column to predict
    # WHY: Specify the label/target
    # HOW: Column name string
    # TRADE-OFF: None can cause runtime errors
    # USE CASE: "price", "class", "outcome"
    
    feature_columns: List[str] = field(default_factory=list)
    # WHAT: Features to use for training
    # WHY: Enable feature selection
    # HOW: List of column names (empty = use all)
    # BENEFIT: Flexible feature selection
    # USE CASE: ["age", "income", "score"]
    
    test_size: float = 0.2
    # WHAT: Train/test split ratio
    # WHY: Reserve data for testing
    # HOW: Fraction of data for test set
    # BENEFIT: Sensible default (80/20 split)
    # TRADE-OFF: No validation (should be 0-1)
    # USE CASE: 0.2 = 20% test, 80% train
    
    random_state: int = 42
    # WHAT: Random seed for reproducibility
    # WHY: Ensure consistent results
    # HOW: Fixed integer seed
    # BENEFIT: Reproducible experiments
    # WHY 42: Convention (Hitchhiker's Guide reference)


@dataclass
class TrainedModel:
    """Represents a trained machine learning model."""
    # WHAT: Entity for trained ML models
    # WHY: Wrap model with metadata
    # HOW: Stores model object + config + metrics
    # BENEFIT: Complete model information
    # PATTERN: Rich domain entity
    
    model: Any
    # WHAT: The actual trained model object
    # WHY: Store sklearn/other ML model
    # HOW: Any type (flexible)
    # BENEFIT: Works with any ML framework
    # TRADE-OFF: No type safety
    # USE CASE: RandomForestClassifier instance
    
    config: ModelConfig
    # WHAT: Configuration used for training
    # WHY: Reproducibility
    # HOW: Stores ModelConfig entity
    # BENEFIT: Know how model was trained
    # USE CASE: Retrain with same config
    
    metrics: Dict[str, float] = field(default_factory=dict)
    # WHAT: Performance metrics
    # WHY: Track model quality
    # HOW: Dictionary of metric names and values
    # BENEFIT: Flexible metric storage
    # USE CASE: {"accuracy": 0.95, "f1": 0.93}
    
    feature_importance: Optional[Dict[str, float]] = None
    # WHAT: Feature importance scores
    # WHY: Model interpretability
    # HOW: Feature name → importance mapping
    # BENEFIT: Understand model decisions
    # TRADE-OFF: Not all models support this
    # USE CASE: {"age": 0.3, "income": 0.5}
    
    training_data_shape: Optional[tuple] = None
    # WHAT: Shape of training data
    # WHY: Validate new data compatibility
    # HOW: (n_samples, n_features)
    # BENEFIT: Catch shape mismatches
    # USE CASE: (800, 10) = 800 samples, 10 features
    
    trained_at: datetime = field(default_factory=datetime.now)
    # WHAT: Training timestamp
    # WHY: Model versioning
    # HOW: Auto-generated timestamp
    # BENEFIT: Track model age
    
    model_path: Optional[str] = None
    # WHAT: Path where model is saved
    # WHY: Persistence tracking
    # HOW: File path string
    # TRADE-OFF: Should use Path type
    # USE CASE: "models/rf_20231210.pkl"


@dataclass
class Prediction:
    """Represents model predictions."""
    # WHAT: Entity for prediction results
    # WHY: Structure prediction output
    # HOW: Wraps predictions with confidence/metadata
    # BENEFIT: Rich prediction information
    # USE CASE: Model inference results
    
    predictions: pd.Series
    # WHAT: The predicted values
    # WHY: Store model output
    # HOW: Pandas Series (preserves index)
    # BENEFIT: Index alignment with input
    # TRADE-OFF: Not serializable
    # USE CASE: [0, 1, 1, 0] for classification
    
    probabilities: Optional[pd.DataFrame] = None
    # WHAT: Class probabilities
    # WHY: Uncertainty quantification
    # HOW: DataFrame with probability per class
    # BENEFIT: Full probability distribution
    # TRADE-OFF: Only for classifiers, memory intensive
    # USE CASE: [[0.8, 0.2], [0.3, 0.7]]
    
    model_used: str = ""
    # WHAT: Name/type of model used
    # WHY: Traceability
    # HOW: String identifier
    # TRADE-OFF: Empty string default is weak
    # USE CASE: "random_forest", "logistic_regression"
    
    confidence_scores: Optional[pd.Series] = None
    # WHAT: Confidence in each prediction
    # WHY: Uncertainty quantification
    # HOW: Series of confidence values
    # BENEFIT: Risk assessment
    # USE CASE: [0.9, 0.6, 0.95, 0.7]
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    # WHAT: Additional prediction metadata
    # WHY: Store extra information
    # HOW: Flexible dictionary
    # BENEFIT: Extensible
    # USE CASE: prediction_time, batch_id
    
    predicted_at: datetime = field(default_factory=datetime.now)
    # WHAT: Prediction timestamp
    # WHY: Audit trail
    # HOW: Auto-generated timestamp
    # BENEFIT: Track when predictions were made
```

---

## Design Patterns Used

### 1. **Entity Pattern** (Domain-Driven Design)
- **What**: Objects with identity that persist over time
- **Why**: Represent core business concepts
- **Examples**: `DataSource`, `ProcessedData`, `TrainedModel`

### 2. **Value Object Pattern**
- **What**: Immutable objects without identity
- **Why**: Represent descriptive aspects
- **Examples**: `ModelConfig`, `EDAReport`

### 3. **Enum Pattern**
- **What**: Type-safe constants
- **Why**: Prevent invalid values
- **Examples**: `DataSourceType`, `ProcessingStatus`

---

## Key Benefits

✅ **Type Safety**: Extensive use of type hints and enums  
✅ **Traceability**: Timestamps and audit trails everywhere  
✅ **Flexibility**: Metadata dictionaries for extension  
✅ **Clean Code**: Dataclasses reduce boilerplate by ~60%  
✅ **Domain Focus**: Pure business logic, no infrastructure

---

## Areas for Improvement

⚠️ **Immutability**: Use `frozen=True` for immutable entities  
⚠️ **Path Handling**: Replace `str` with `Path` objects  
⚠️ **Timezone Awareness**: Use `datetime.now(timezone.utc)`  
⚠️ **Validation**: Add `__post_init__` validation methods  
⚠️ **Type Safety**: Convert string constants to Enums

---

## Usage Examples

```python
# Create a data source
source = DataSource(
    source_type=DataSourceType.CSV,
    path="data/sales.csv",
    metadata={"encoding": "utf-8"}
)

# Create processed data
processed = ProcessedData(
    data=df,
    source=source,
    status=ProcessingStatus.IN_PROGRESS
)

# Track processing steps
processed.add_processing_step("cleaned")
processed.add_processing_step("transformed")
processed.mark_completed()

# Create model config
config = ModelConfig(
    model_type="random_forest",
    target_column="price",
    hyperparameters={"n_estimators": 100}
)
```

---

## Dependencies

- `dataclasses`: Standard library (Python 3.7+)
- `datetime`: Standard library
- `enum`: Standard library
- `typing`: Standard library
- `pandas`: External (data manipulation)

---

**Total Lines**: 115  
**Complexity**: Medium  
**Maintainability**: High  
**Test Coverage**: Should be 100% (pure logic)
