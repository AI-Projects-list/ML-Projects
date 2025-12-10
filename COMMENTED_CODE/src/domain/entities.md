# entities.py - Complete Line-by-Line Documentation

**Source**: `src/domain/entities.py`  
**Purpose**: Core domain entities representing business concepts  
**Layer**: Domain (innermost layer - pure business logic)  
**Lines**: 120  
**Patterns**: Entity Pattern, Enum Pattern, DataClass Pattern

---

## File Overview

This file defines the **heart of the business logic** - the core entities that represent real-world concepts in our ML pipeline. These are **not** database models or DTOs; they are rich domain objects with behavior.

**Key Principle**: Domain entities should be framework-independent and contain business logic.

---

## Complete Code with Comprehensive Comments

```python
"""Domain entities representing core business objects."""
# MODULE DOCSTRING
# WHAT: This module contains all domain entities
# WHY: Central definition of business concepts
# BENEFIT: Single source of truth for domain models
# PATTERN: Domain-Driven Design (DDD)

from dataclasses import dataclass, field
# WHAT: Python's dataclass decorator for automatic __init__, __repr__, etc.
# WHY: Reduces boilerplate code for data-holding classes
# HOW: @dataclass decorator generates special methods automatically
# BENEFIT: Less code, immutable options, type hints
# TRADE-OFF: Less control than manual __init__
# ALTERNATIVE: NamedTuple (immutable), attrs library, regular class

from datetime import datetime
# WHAT: Python's datetime for timestamp handling
# WHY: Track when entities are created/modified
# BENEFIT: Standard library, timezone-aware, widely supported
# TRADE-OFF: Mutable (use pendulum or arrow for immutable)

from enum import Enum
# WHAT: Python's Enum base class
# WHY: Type-safe constants (better than strings or ints)
# BENEFIT: Prevents typos, IDE autocomplete, type checking
# TRADE-OFF: More verbose than plain strings
# EXAMPLE: DataSourceType.CSV vs "csv" (typo-proof)

from typing import Any, Dict, List, Optional
# WHAT: Type hint utilities
# WHY: Static type checking, documentation, IDE support
# BENEFIT: Catch type errors before runtime
# TRADE-OFF: Python runtime doesn't enforce types (use mypy)
# NOTE: Dict[str, Any] = dictionary with string keys, any values

import pandas as pd
# WHAT: pandas library for DataFrame type hints
# WHY: DataFrames are the standard data structure in ML
# BENEFIT: Rich API, NumPy integration, widespread use
# TRADE-OFF: Heavy dependency in domain layer (normally avoid)
# JUSTIFICATION: DataFrame is de facto standard, acceptable compromise


class DataSourceType(Enum):
    """Types of data sources supported."""
    # WHAT: Enum defining supported file formats
    # WHY: Type-safe file type identification
    # PATTERN: Enum Pattern
    # BENEFIT: Prevents invalid values ("cvs" typo caught at runtime)
    # USE CASE: Factory pattern to select appropriate reader
    
    CSV = "csv"
    # WHAT: Comma-separated values format
    # WHY: Most common tabular data format
    # USE CASE: pd.read_csv()
    # BENEFIT: Universal support, human-readable
    
    TXT = "txt"
    # WHAT: Plain text files
    # WHY: Unstructured text data, logs, documents
    # USE CASE: Text analysis, NLP tasks
    # BENEFIT: Simple, universal format
    
    PDF = "pdf"
    # WHAT: Adobe PDF documents
    # WHY: Common business document format
    # USE CASE: Extract tables from PDF reports
    # TRADE-OFF: Complex parsing, may lose structure
    
    PDF_SCAN = "pdf_scan"
    # WHAT: Scanned PDF images
    # WHY: Legacy documents, scanned forms
    # USE CASE: OCR (Optical Character Recognition)
    # TRADE-OFF: Accuracy issues, computationally expensive
    # REQUIRES: tesseract OCR engine
    
    DATAFRAME = "dataframe"
    # WHAT: Already-loaded pandas DataFrame
    # WHY: In-memory data already processed
    # USE CASE: Skip file reading, work with existing data
    # BENEFIT: Fastest option, no I/O


class ProcessingStatus(Enum):
    """Status of data processing."""
    # WHAT: Enum tracking processing pipeline state
    # WHY: Monitor workflow progress, handle failures
    # PATTERN: State Pattern
    # BENEFIT: Clear state transitions, error handling
    # USE CASE: UI progress indicators, retry logic
    
    PENDING = "pending"
    # WHAT: Initial state, not yet started
    # TRANSITION: → IN_PROGRESS when processing begins
    
    IN_PROGRESS = "in_progress"
    # WHAT: Currently being processed
    # TRANSITION: → COMPLETED on success, FAILED on error
    # USE CASE: Show progress bar, prevent concurrent processing
    
    COMPLETED = "completed"
    # WHAT: Successfully processed
    # TERMINAL STATE: No further transitions
    # USE CASE: Enable downstream operations
    
    FAILED = "failed"
    # WHAT: Processing failed with error
    # TERMINAL STATE: May transition back to PENDING for retry
    # USE CASE: Error notification, retry logic


@dataclass
class DataSource:
    """Represents a data source."""
    # WHAT: Entity representing where data comes from
    # WHY: Track data lineage, metadata, provenance
    # PATTERN: Entity Pattern (has identity via path)
    # RESPONSIBILITY: Hold metadata about data origin
    # NOT RESPONSIBLE FOR: Reading data (that's IDataReader's job)
    
    source_type: DataSourceType
    # WHAT: Type of data source (CSV, PDF, etc.)
    # WHY: Determine which reader to use
    # TYPE: Enum (type-safe)
    # REQUIRED: Yes (no default)
    # USE CASE: DataReaderFactory.get_reader(source)
    
    path: str
    # WHAT: File path or URL to data
    # WHY: Location of the actual data
    # TYPE: str (not Path) for serialization compatibility
    # TRADE-OFF: str less type-safe than pathlib.Path
    # VALIDATION: Should check file exists (not done here)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    # WHAT: Additional key-value metadata
    # WHY: Flexible storage for format-specific options
    # TYPE: Dict with string keys, any values
    # DEFAULT: Empty dict (field(default_factory=dict) not dict())
    # WHY default_factory: Avoid mutable default argument bug
    # EXAMPLES: {"encoding": "utf-8", "separator": ";"}
    # BENEFIT: Extensible without changing class
    
    created_at: datetime = field(default_factory=datetime.now)
    # WHAT: Timestamp when this entity was created
    # WHY: Track data lineage, audit trail
    # DEFAULT: Current time when object created
    # WHY datetime.now: Callable (default_factory), not value
    # TRADE-OFF: Uses local timezone (better: datetime.utcnow)
    # USE CASE: Debug when data was ingested


@dataclass
class ProcessedData:
    """Represents processed data ready for analysis."""
    # WHAT: Entity wrapping DataFrame with processing metadata
    # WHY: Rich domain model vs anemic DataFrame
    # PATTERN: Entity Pattern + Wrapper Pattern
    # BENEFIT: Track processing history, status, metadata
    # TRADE-OFF: More complex than plain DataFrame
    
    data: pd.DataFrame
    # WHAT: The actual data (pandas DataFrame)
    # WHY: Standard ML data structure
    # TYPE: DataFrame (rows × columns)
    # REQUIRED: Yes
    # TRADE-OFF: Mutable (DataFrame can be changed)
    # BETTER: Make DataFrame immutable (not practical)
    
    source: DataSource
    # WHAT: Reference to original data source
    # WHY: Data lineage, traceability
    # TYPE: DataSource entity
    # BENEFIT: Know where data came from
    # USE CASE: Debugging, audit trail
    
    processing_steps: List[str] = field(default_factory=list)
    # WHAT: Ordered list of processing steps applied
    # WHY: Reproducibility, debugging, audit trail
    # TYPE: List of strings
    # DEFAULT: Empty list
    # EXAMPLES: ["cleaned", "transformed", "validated"]
    # BENEFIT: Complete processing history
    # USE CASE: Debug pipeline, reproduce results
    
    status: ProcessingStatus = ProcessingStatus.PENDING
    # WHAT: Current processing status
    # WHY: Track pipeline state
    # TYPE: ProcessingStatus enum
    # DEFAULT: PENDING (not yet started)
    # TRANSITIONS: PENDING → IN_PROGRESS → COMPLETED/FAILED
    # USE CASE: Workflow management, error handling
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    # WHAT: Flexible key-value storage
    # WHY: Store processing-specific information
    # EXAMPLES: {"validation_passed": True, "quality_score": 0.85}
    # BENEFIT: Extensible without class changes
    # TRADE-OFF: Type-unsafe (any value type)
    
    processed_at: Optional[datetime] = None
    # WHAT: Timestamp when processing completed
    # WHY: Track processing time, performance metrics
    # TYPE: Optional (None initially)
    # VALUE: Set when mark_completed() called
    # USE CASE: Calculate processing duration
    
    def mark_completed(self) -> None:
        """Mark data processing as completed."""
        # WHAT: Method to mark successful completion
        # WHY: Encapsulate state transition logic
        # PATTERN: State Pattern
        # BENEFIT: Single place for completion logic
        # SIDE EFFECTS: Modifies status and processed_at
        
        self.status = ProcessingStatus.COMPLETED
        # WHAT: Set status to COMPLETED
        # WHY: Indicate success
        # ALLOWS: Downstream operations to proceed
        
        self.processed_at = datetime.now()
        # WHAT: Record completion timestamp
        # WHY: Audit trail, performance tracking
        # TRADE-OFF: Local timezone (better: UTC)
    
    def mark_failed(self) -> None:
        """Mark data processing as failed."""
        # WHAT: Method to mark failure
        # WHY: Encapsulate failure handling
        # PATTERN: State Pattern
        # USE CASE: Exception handling in use cases
        
        self.status = ProcessingStatus.FAILED
        # WHAT: Set status to FAILED
        # WHY: Prevent downstream operations
        # ALLOWS: Retry logic to detect failures
        
        self.processed_at = datetime.now()
        # WHAT: Record failure timestamp
        # WHY: Track when failure occurred
        # USE CASE: Debugging, retry delays
    
    def add_processing_step(self, step: str) -> None:
        """Add a processing step to the history."""
        # WHAT: Append step to processing history
        # WHY: Build complete audit trail
        # PATTERN: Event Sourcing (lite version)
        # BENEFIT: Full processing history
        
        self.processing_steps.append(step)
        # WHAT: Add step name to list
        # WHY: Track what was done
        # EXAMPLES: "cleaned", "transformed", "encoded"
        # USE CASE: Reproduce processing, debugging


@dataclass
class EDAReport:
    """Represents exploratory data analysis report."""
    # WHAT: Entity holding EDA results
    # WHY: Structured analysis output
    # PATTERN: Value Object (could be frozen)
    # BENEFIT: Rich report vs scattered variables
    
    data_shape: tuple
    # WHAT: (rows, columns) of analyzed data
    # WHY: Quick data size check
    # TYPE: tuple (immutable)
    # EXAMPLE: (1000, 50) = 1000 rows, 50 columns
    
    column_types: Dict[str, str]
    # WHAT: Mapping column names to data types
    # WHY: Understand data structure
    # EXAMPLE: {"age": "int64", "name": "object"}
    # USE CASE: Type validation, casting decisions
    
    missing_values: Dict[str, int]
    # WHAT: Count of missing values per column
    # WHY: Data quality assessment
    # EXAMPLE: {"age": 5, "name": 0}
    # USE CASE: Decide on imputation strategy
    
    statistics: Dict[str, Any]
    # WHAT: Descriptive statistics
    # WHY: Understand data distribution
    # EXAMPLES: mean, median, std, min, max, quartiles
    # TYPE: Any (could be numbers, dicts, arrays)
    
    correlations: Optional[pd.DataFrame] = None
    # WHAT: Correlation matrix between features
    # WHY: Detect multicollinearity, feature selection
    # TYPE: Optional DataFrame (None if not computed)
    # USE CASE: Feature engineering, model insights
    
    visualizations: Dict[str, str] = field(default_factory=dict)
    # WHAT: Mapping plot names to file paths
    # WHY: Link report to visualization files
    # EXAMPLE: {"histogram": "outputs/hist.png"}
    # TRADE-OFF: Stores paths not images (less portable)
    
    insights: List[str] = field(default_factory=list)
    # WHAT: Generated insights from analysis
    # WHY: Actionable findings
    # EXAMPLES: "Age has 5% missing values", "High correlation between A and B"
    # BENEFIT: Automated insight generation
    
    generated_at: datetime = field(default_factory=datetime.now)
    # WHAT: Report generation timestamp
    # WHY: Track when analysis was performed
    # USE CASE: Versioning, staleness detection


@dataclass
class ModelConfig:
    """Configuration for ML models."""
    # WHAT: Entity defining model training configuration
    # WHY: Reproducible, versioned model training
    # PATTERN: Configuration Object
    # BENEFIT: Separate config from code
    
    model_type: str
    # WHAT: Type of ML model
    # WHY: Select algorithm
    # EXAMPLES: "random_forest", "logistic_regression"
    # TRADE-OFF: str not enum (more flexible but less type-safe)
    # VALIDATION: Should validate against supported types
    
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    # WHAT: Model-specific hyperparameters
    # WHY: Customize model behavior
    # EXAMPLES: {"n_estimators": 100, "max_depth": 10}
    # BENEFIT: Flexible across different models
    # USE CASE: Hyperparameter tuning
    
    target_column: Optional[str] = None
    # WHAT: Name of target/label column
    # WHY: Identify what to predict
    # TYPE: Optional (None for unsupervised learning)
    # EXAMPLE: "price", "churn", "category"
    
    feature_columns: List[str] = field(default_factory=list)
    # WHAT: Explicit list of feature columns
    # WHY: Feature selection
    # DEFAULT: Empty list = use all except target
    # USE CASE: Manual feature selection
    
    test_size: float = 0.2
    # WHAT: Proportion of data for testing
    # WHY: Train/test split ratio
    # TYPE: float between 0 and 1
    # DEFAULT: 0.2 = 20% test, 80% train
    # COMMON VALUES: 0.2, 0.25, 0.3
    
    random_state: int = 42
    # WHAT: Random seed for reproducibility
    # WHY: Reproducible splits and training
    # DEFAULT: 42 (common convention)
    # BENEFIT: Same results across runs
    # TRADE-OFF: May not reflect production variance


@dataclass
class TrainedModel:
    """Represents a trained machine learning model."""
    # WHAT: Entity wrapping trained model with metadata
    # WHY: Rich model object vs bare sklearn model
    # PATTERN: Entity Pattern
    # BENEFIT: Metadata, metrics, provenance
    
    model: Any
    # WHAT: The actual trained model object
    # WHY: The prediction engine
    # TYPE: Any (sklearn models, custom models, etc.)
    # EXAMPLES: RandomForestClassifier, LinearRegression
    # TRADE-OFF: Any = no type safety
    
    config: ModelConfig
    # WHAT: Configuration used for training
    # WHY: Reproducibility
    # TYPE: ModelConfig entity
    # BENEFIT: Know how model was trained
    
    metrics: Dict[str, float] = field(default_factory=dict)
    # WHAT: Evaluation metrics
    # WHY: Model performance tracking
    # EXAMPLES: {"accuracy": 0.95, "precision": 0.92, "recall": 0.89}
    # USE CASE: Model comparison, monitoring
    
    feature_importance: Optional[Dict[str, float]] = None
    # WHAT: Feature importance scores
    # WHY: Model interpretability
    # TYPE: Optional dict (None if not available)
    # EXAMPLE: {"age": 0.3, "income": 0.5}
    # USE CASE: Feature engineering insights
    
    training_data_shape: Optional[tuple] = None
    # WHAT: Shape of training data
    # WHY: Track model capacity
    # EXAMPLE: (8000, 30) = 8000 samples, 30 features
    # USE CASE: Detect train/inference shape mismatches
    
    trained_at: datetime = field(default_factory=datetime.now)
    # WHAT: Training completion timestamp
    # WHY: Model versioning, staleness detection
    # USE CASE: Model registry, deployment tracking
    
    model_path: Optional[str] = None
    # WHAT: Path where model is saved
    # WHY: Persistence tracking
    # TYPE: Optional (None if not saved)
    # USE CASE: Model loading, deployment


@dataclass
class Prediction:
    """Represents model predictions."""
    # WHAT: Entity wrapping prediction results
    # WHY: Rich prediction object vs bare array
    # PATTERN: Value Object (immutable would be better)
    # BENEFIT: Metadata, confidence, provenance
    
    predictions: pd.Series
    # WHAT: The predicted values
    # WHY: Main output of model.predict()
    # TYPE: pandas Series (1D array with index)
    # EXAMPLES: [0, 1, 1, 0] for classification
    
    probabilities: Optional[pd.DataFrame] = None
    # WHAT: Prediction probabilities (for classifiers)
    # WHY: Confidence estimation
    # TYPE: Optional DataFrame (None for regression)
    # SHAPE: (n_samples, n_classes)
    # EXAMPLE: [[0.8, 0.2], [0.3, 0.7]] = class probabilities
    
    model_used: str = ""
    # WHAT: Name of model that made predictions
    # WHY: Traceability
    # TYPE: str
    # EXAMPLE: "random_forest_v1.2"
    # USE CASE: Multi-model comparison, A/B testing
    
    confidence_scores: Optional[pd.Series] = None
    # WHAT: Confidence score per prediction
    # WHY: Uncertainty quantification
    # TYPE: Optional Series (None if not available)
    # CALCULATION: max(probabilities) for classifiers
    # USE CASE: Filter low-confidence predictions
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    # WHAT: Additional prediction metadata
    # WHY: Extensibility
    # EXAMPLES: {"model_version": "1.2", "preprocessing": "standard"}
    
    predicted_at: datetime = field(default_factory=datetime.now)
    # WHAT: Prediction timestamp
    # WHY: Track when predictions were made
    # USE CASE: Monitoring, debugging, cache invalidation
```

---

## Design Patterns Identified

### 1. **Entity Pattern** (DDD)
- **Classes**: DataSource, ProcessedData, TrainedModel
- **Characteristic**: Have identity, mutable, business logic
- **Benefit**: Rich domain models

### 2. **Value Object Pattern** (DDD)
- **Classes**: EDAReport, Prediction, ModelConfig
- **Characteristic**: Defined by values, immutable (should be frozen)
- **Benefit**: Thread-safe, cacheable

### 3. **Enum Pattern**
- **Classes**: DataSourceType, ProcessingStatus
- **Benefit**: Type-safe constants

### 4. **State Pattern**
- **Methods**: mark_completed(), mark_failed()
- **Benefit**: Encapsulated state transitions

---

## Pros & Cons

### ✅ Pros

1. **Type Safety**: Enums prevent typos, type hints catch errors
2. **Rich Models**: Entities have behavior (mark_completed, add_processing_step)
3. **Auditability**: Timestamps, processing steps, metadata
4. **Flexibility**: Dict metadata allows extensibility
5. **Clean Code**: Dataclasses reduce boilerplate
6. **Separation**: Domain entities independent of frameworks

### ❌ Cons & Trade-offs

1. **Mutability**: Most entities mutable (ProcessedData.data can change)
   - **Fix**: Use frozen=True for value objects
2. **Pandas Dependency**: Domain depends on pandas
   - **Trade-off**: Acceptable (DataFrame is standard)
3. **String Types**: model_type is str not enum
   - **Fix**: Create ModelType enum
4. **No Validation**: No field validators (path exists, test_size range)
   - **Fix**: Add Pydantic validators or __post_init__
5. **Timezone**: datetime.now() uses local timezone
   - **Fix**: Use datetime.utcnow()
6. **Type Safety**: Many `Any` types
   - **Fix**: More specific types where possible

---

## Improvements

```python
# Better: Frozen value objects
@dataclass(frozen=True)
class EDAReport:
    ...

# Better: Model type enum
class ModelType(Enum):
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"

# Better: Field validation
@dataclass
class ModelConfig:
    test_size: float = 0.2
    
    def __post_init__(self):
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")

# Better: UTC timestamps
processed_at: datetime = field(default_factory=datetime.utcnow)
```

---

**Lines**: 120  
**Entities**: 7  
**Enums**: 2  
**Methods**: 3  
**Complexity**: Low-Medium  
**Dependencies**: pandas, dataclasses, datetime, enum, typing
