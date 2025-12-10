# Domain Repositories - Detailed Code Documentation

**File**: `src/domain/repositories.py`  
**Purpose**: Define interface contracts for infrastructure implementations  
**Layer**: Domain (Ports in Hexagonal Architecture)  
**Pattern**: Repository Pattern + Dependency Inversion Principle

---

## Overview

This file defines **interfaces (contracts)** that infrastructure layer must implement. This is the foundation of **Hexagonal Architecture** where these are the **PORTS** that allow different **ADAPTERS** to be plugged in.

**Key Concept**: The domain defines WHAT it needs, infrastructure provides HOW.

---

## Complete Code with Line-by-Line Comments

```python
"""Repository interfaces (ports) for the domain layer."""
# WHAT: Module defining abstract interfaces
# WHY: Dependency Inversion - domain doesn't depend on infrastructure
# HOW: Abstract Base Classes with @abstractmethod
# PATTERN: Hexagonal Architecture - these are "Ports"

from abc import ABC, abstractmethod
# WHAT: Import Abstract Base Class support
# WHY: Enforce interface contracts
# HOW: Classes inherit from ABC, methods decorated with @abstractmethod
# BENEFIT: Cannot instantiate interfaces, must implement all methods
# ENFORCES: Contract compliance at runtime

from pathlib import Path
# WHAT: Modern path handling
# WHY: Platform-independent file paths
# BENEFIT: Better than strings, object-oriented API
# USE CASE: Cross-platform file operations

from typing import List, Optional
# WHAT: Type hint utilities
# WHY: Type safety and documentation
# BENEFIT: IDE autocomplete, type checking

import pandas as pd
# WHAT: Import pandas for DataFrame type hints
# WHY: Standard data structure in ML/Data Science
# TRADE-OFF: Heavy dependency in domain layer
# JUSTIFICATION: DataFrame is de facto standard

from src.domain.entities import (
    DataSource,
    EDAReport,
    ModelConfig,
    Prediction,
    ProcessedData,
    TrainedModel,
)
# WHAT: Import domain entities
# WHY: Use rich domain types in interfaces
# BENEFIT: Type-safe contracts
# DEPENDENCY: domain → domain (clean)


class IDataReader(ABC):
    """Interface for reading data from various sources."""
    # WHAT: Abstract interface for data readers
    # WHY: Support multiple file formats without changing domain
    # PATTERN: Strategy Pattern + Interface Segregation
    # BENEFIT: Easy to add new readers (CSV, JSON, XML, databases)
    
    # WHY 'I' prefix?
    # Convention from C#/Java for interfaces
    # TRADE-OFF: Some prefer no prefix (Python convention)
    # ALTERNATIVE: Use Protocol from typing for structural subtyping
    
    @abstractmethod
    def can_read(self, source: DataSource) -> bool:
        """Check if this reader can handle the given source."""
        # WHAT: Capability checking method
        # WHY: Chain of Responsibility pattern
        # HOW: Each reader checks if it can handle source type
        # BENEFIT: Runtime source type selection
        
        # PATTERN: Chain of Responsibility
        # Factory asks each reader: can_read(source)?
        # First reader that returns True is selected
        
        # ALTERNATIVE APPROACH:
        # Use type registry: {DataSourceType.CSV: CSVReader}
        # TRADE-OFF: Less flexible, but more explicit
        pass
    
    @abstractmethod
    def read(self, source: DataSource) -> pd.DataFrame:
        """Read data from the source."""
        # WHAT: Main read operation
        # WHY: Convert various formats to standard DataFrame
        # RETURN: pandas DataFrame (standard format)
        
        # BENEFIT: Uniform interface for all readers
        # TRADE-OFF: Loads entire file into memory
        # IMPROVEMENT: Add streaming support
        # SIGNATURE: Could add chunk_size parameter
        
        # USE CASE:
        # csv_reader.read(source) → DataFrame
        # pdf_reader.read(source) → DataFrame
        # Both return same type!
        pass


class IDataProcessor(ABC):
    """Interface for data processing operations."""
    # WHAT: Interface for data cleaning/transformation
    # WHY: Separate processing logic from domain
    # PATTERN: Template Method (clean → transform → validate)
    
    # WHY three separate methods?
    # - Single Responsibility Principle
    # - Flexible pipeline (skip steps)
    # - Easy to test separately
    
    @abstractmethod
    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean the data."""
        # WHAT: Data cleaning operation
        # WHY: Handle missing values, duplicates, outliers
        # RETURN: Cleaned DataFrame
        
        # RESPONSIBILITIES:
        # - Missing value imputation
        # - Duplicate removal
        # - Outlier handling
        # - Column dropping (low quality)
        
        # BENEFIT: Returns new DataFrame (functional style)
        # TRADE-OFF: No configuration in signature
        # IMPROVEMENT: Add CleaningConfig parameter
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        # WHAT: Data transformation operation
        # WHY: Encode categoricals, scale numerics, engineer features
        # RETURN: Transformed DataFrame
        
        # RESPONSIBILITIES:
        # - Categorical encoding
        # - Feature scaling
        # - Datetime feature extraction
        # - Feature engineering
        
        # TRADE-OFF: No parameters for transformation type
        # IMPROVEMENT: Add TransformConfig parameter
        pass
    
    @abstractmethod
    def validate(self, data: pd.DataFrame) -> bool:
        """Validate the data quality."""
        # WHAT: Quality validation
        # WHY: Ensure data meets quality standards
        # RETURN: Boolean (pass/fail)
        
        # RESPONSIBILITIES:
        # - Completeness check
        # - Consistency check
        # - Validity check
        
        # TRADE-OFF: Boolean doesn't provide details
        # IMPROVEMENT: Return ValidationReport entity
        # BETTER SIGNATURE:
        # def validate(self, data) -> ValidationReport
        pass


class IEDAAnalyzer(ABC):
    """Interface for exploratory data analysis."""
    # WHAT: Interface for EDA operations
    # WHY: Separate analysis from domain logic
    # PATTERN: Strategy Pattern
    
    @abstractmethod
    def analyze(self, data: ProcessedData) -> EDAReport:
        """Perform exploratory data analysis."""
        # WHAT: Main analysis method
        # WHY: Generate statistical insights
        # PARAMETER: ProcessedData (not just DataFrame)
        # BENEFIT: Access to metadata, processing history
        # RETURN: Rich EDAReport entity
        
        # RESPONSIBILITIES:
        # - Descriptive statistics
        # - Correlation analysis
        # - Outlier detection
        # - Distribution analysis
        # - Generate insights
        
        # TRADE-OFF: No configuration options
        # IMPROVEMENT: Add EDAConfig parameter
        pass
    
    @abstractmethod
    def generate_visualizations(
        self, data: ProcessedData, output_dir: Path
    ) -> List[str]:
        """Generate visualization plots."""
        # WHAT: Create and save visualizations
        # WHY: Separate from analyze() for flexibility
        # PARAMETER: output_dir (where to save plots)
        # RETURN: List of file paths
        
        # RESPONSIBILITIES:
        # - Histogram plots
        # - Correlation heatmap
        # - Boxplots for outliers
        # - Save to files
        
        # BENEFIT: Can analyze without generating plots
        # TRADE-OFF: Side effect (file I/O)
        # IMPROVEMENT: Add plot_types parameter
        # SIGNATURE:
        # def generate_visualizations(
        #     self, data, output_dir, plot_types: List[PlotType]
        # ) -> List[str]
        pass


class IModelTrainer(ABC):
    """Interface for model training."""
    # WHAT: Interface for ML model training
    # WHY: Support multiple ML algorithms
    # PATTERN: Strategy Pattern + Template Method
    
    @abstractmethod
    def train(self, data: ProcessedData, config: ModelConfig) -> TrainedModel:
        """Train a machine learning model."""
        # WHAT: Main training method
        # WHY: Train model on data
        # PARAMETER: ProcessedData (rich context)
        # PARAMETER: ModelConfig (model type + hyperparameters)
        # RETURN: TrainedModel (model + metadata)
        
        # RESPONSIBILITIES:
        # - Feature/target split
        # - Train/test split
        # - Model instantiation
        # - Model training
        # - Metric calculation
        # - Feature importance extraction
        
        # BENEFIT: Rich input/output types
        # TRADE-OFF: No callbacks for progress
        # IMPROVEMENT: Add callbacks parameter
        pass
    
    @abstractmethod
    def evaluate(self, model: TrainedModel, test_data: pd.DataFrame) -> dict:
        """Evaluate model performance."""
        # WHAT: Evaluate trained model
        # WHY: Reusable on different datasets
        # RETURN: Dictionary of metrics
        
        # RESPONSIBILITIES:
        # - Generate predictions
        # - Calculate metrics
        # - Return results
        
        # TRADE-OFF: Returns dict (not typed)
        # IMPROVEMENT: Return MetricsReport entity
        # BETTER SIGNATURE:
        # def evaluate(self, model, test_data) -> MetricsReport
        pass


class IPredictor(ABC):
    """Interface for making predictions."""
    # WHAT: Interface for model inference
    # WHY: Separate training from prediction
    # PATTERN: Strategy Pattern
    
    @abstractmethod
    def predict(self, model: TrainedModel, data: pd.DataFrame) -> Prediction:
        """Make predictions using the trained model."""
        # WHAT: Generate predictions
        # WHY: Inference on new data
        # PARAMETER: TrainedModel (model + metadata)
        # PARAMETER: DataFrame (features)
        # RETURN: Rich Prediction entity
        
        # RESPONSIBILITIES:
        # - Feature preparation
        # - Model prediction
        # - Probability calculation (if classifier)
        # - Confidence score calculation
        
        # BENEFIT: Rich return type with confidence
        # TRADE-OFF: No batch size configuration
        # IMPROVEMENT: Add batch_size parameter for large datasets
        pass


class IModelRepository(ABC):
    """Interface for model persistence."""
    # WHAT: Interface for model storage
    # WHY: Abstract persistence mechanism
    # PATTERN: Repository Pattern
    
    # BENEFIT:
    # - Can swap storage (pickle, joblib, ONNX, database)
    # - Easy to test (mock repository)
    # - Versioning support
    
    @abstractmethod
    def save(self, model: TrainedModel, path: Path) -> None:
        """Save a trained model."""
        # WHAT: Persist model to disk
        # WHY: Reuse trained models
        # PARAMETER: TrainedModel (entire entity)
        # PARAMETER: Path (where to save)
        # RETURN: None
        
        # RESPONSIBILITIES:
        # - Serialize model object
        # - Serialize metadata
        # - Write to file
        
        # TRADE-OFF: No return value (success/fail indicator)
        # IMPROVEMENT: Return bool or raise specific exception
        # TRADE-OFF: No versioning support
        # IMPROVEMENT: Add version parameter
        pass
    
    @abstractmethod
    def load(self, path: Path) -> TrainedModel:
        """Load a trained model."""
        # WHAT: Load model from disk
        # WHY: Reuse saved models
        # PARAMETER: Path (where to load from)
        # RETURN: TrainedModel entity
        
        # RESPONSIBILITIES:
        # - Read from file
        # - Deserialize model object
        # - Deserialize metadata
        # - Reconstruct TrainedModel entity
        
        # TRADE-OFF: No lazy loading
        # IMPROVEMENT: Add lazy loading option
        # TRADE-OFF: Exception on missing file
        # IMPROVEMENT: Return Optional[TrainedModel]
        pass
    
    @abstractmethod
    def list_models(self, directory: Path) -> List[str]:
        """List all available models."""
        # WHAT: Model discovery
        # WHY: Find saved models
        # PARAMETER: Directory to search
        # RETURN: List of model paths
        
        # RESPONSIBILITIES:
        # - Scan directory
        # - Find model files
        # - Return paths
        
        # TRADE-OFF: Returns strings not Path objects
        # IMPROVEMENT: Return List[Path]
        # TRADE-OFF: No filtering options
        # IMPROVEMENT: Add filter parameter (by date, type, etc.)
        pass


class IDataRepository(ABC):
    """Interface for data persistence."""
    # WHAT: Interface for data storage
    # WHY: Abstract data persistence
    # PATTERN: Repository Pattern
    
    # BENEFIT:
    # - Can swap storage (pickle, parquet, database)
    # - Consistent with IModelRepository
    
    @abstractmethod
    def save(self, data: ProcessedData, path: Path) -> None:
        """Save processed data."""
        # WHAT: Persist processed data
        # WHY: Reuse preprocessed data
        # PARAMETER: ProcessedData (entity with metadata)
        # PARAMETER: Path (where to save)
        
        # RESPONSIBILITIES:
        # - Serialize DataFrame
        # - Serialize metadata
        # - Write to file
        
        # BENEFIT: Preserves processing history
        # TRADE-OFF: No compression options
        # IMPROVEMENT: Add compression parameter
        pass
    
    @abstractmethod
    def load(self, path: Path) -> ProcessedData:
        """Load processed data."""
        # WHAT: Load processed data
        # WHY: Reuse saved data
        # RETURN: ProcessedData entity
        
        # RESPONSIBILITIES:
        # - Read from file
        # - Deserialize DataFrame
        # - Deserialize metadata
        # - Reconstruct ProcessedData entity
        
        # TRADE-OFF: Loads entire dataset into memory
        # IMPROVEMENT: Add chunking support
        pass
```

---

## Design Patterns Used

### 1. **Repository Pattern**
- **WHAT**: Abstract data access
- **WHY**: Separate domain from storage
- **EXAMPLES**: `IModelRepository`, `IDataRepository`

### 2. **Strategy Pattern**
- **WHAT**: Interchangeable algorithms
- **WHY**: Multiple implementations
- **EXAMPLES**: `IDataReader`, `IModelTrainer`

### 3. **Dependency Inversion Principle** (SOLID)
- **WHAT**: Depend on abstractions
- **WHY**: Loose coupling
- **HOW**: Use case depends on `IDataProcessor`, not `DataProcessor`

### 4. **Interface Segregation Principle** (SOLID)
- **WHAT**: Small focused interfaces
- **WHY**: Clients only depend on what they use
- **EXAMPLES**: Separate interfaces for reading, processing, analyzing

---

## Key Benefits

✅ **Testability**: Easy to mock for unit tests  
✅ **Flexibility**: Swap implementations without changing domain  
✅ **Framework Independence**: No infrastructure dependencies  
✅ **Clear Contracts**: Explicit interface definitions  
✅ **Plugin Architecture**: Add new readers/trainers easily

---

## Usage Example

```python
# Interface (domain layer)
class IDataReader(ABC):
    @abstractmethod
    def read(self, source: DataSource) -> pd.DataFrame:
        pass

# Implementation (infrastructure layer)
class CSVDataReader(IDataReader):
    def read(self, source: DataSource) -> pd.DataFrame:
        return pd.read_csv(source.path)

# Use case (application layer)
class DataIngestionUseCase:
    def __init__(self, reader: IDataReader):  # Depends on interface!
        self.reader = reader
    
    def execute(self, source: DataSource):
        return self.reader.read(source)

# Testing (test layer)
class MockReader(IDataReader):
    def read(self, source: DataSource) -> pd.DataFrame:
        return pd.DataFrame({"test": [1, 2, 3]})

use_case = DataIngestionUseCase(MockReader())  # Easy to test!
```

---

**Total Lines**: 120  
**Interfaces**: 7  
**Methods**: 14  
**Complexity**: Low  
**Coupling**: Zero (interfaces have no dependencies)
