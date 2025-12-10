# repositories.py - Complete Line-by-Line Documentation

**Source**: `src/domain/repositories.py`  
**Purpose**: Define interface contracts (Ports in Hexagonal Architecture)  
**Layer**: Domain  
**Lines**: 120  
**Patterns**: Repository Pattern, Interface Segregation, Dependency Inversion

---

## Complete Annotated Code

```python
"""Repository interfaces (ports) for the domain layer."""
# WHAT: Domain layer interfaces
# WHY: Dependency Inversion - domain defines contracts, infrastructure implements
# PATTERN: Hexagonal Architecture "Ports"
# BENEFIT: Domain independent of infrastructure

from abc import ABC, abstractmethod
# WHAT: Abstract Base Class support
# WHY: Create interfaces that must be implemented
# HOW: Inherit ABC, use @abstractmethod decorator
# BENEFIT: Cannot instantiate, enforces contract
# PYTHON NOTE: Unlike Java/C#, Python doesn't have native interfaces

from pathlib import Path
# WHAT: Modern path handling
# WHY: Platform-independent file operations
# BENEFIT: Object-oriented API vs string paths
# USE CASE: Cross-platform file handling

from typing import List, Optional
# WHAT: Type hints
# WHY: Type safety, documentation, IDE support
# BENEFIT: Catch errors early

import pandas as pd
# WHAT: pandas for DataFrame type hints
# WHY: Standard data structure in data science
# TRADE-OFF: Domain depends on pandas (heavy dependency)
# JUSTIFICATION: DataFrame is industry standard

from src.domain.entities import (
    DataSource,
    EDAReport,
    ModelConfig,
    Prediction,
    ProcessedData,
    TrainedModel,
)
# WHAT: Import domain entities
# WHY: Use rich types in interfaces
# DEPENDENCY: domain → domain (clean)
# BENEFIT: Type-safe contracts


class IDataReader(ABC):
    """Interface for reading data from various sources."""
    # WHAT: Interface for data readers
    # WHY: Support multiple formats (CSV, PDF, TXT)
    # PATTERN: Strategy Pattern + Interface
    # NAMING: 'I' prefix (convention from C#/Java)
    # ALTERNATIVE: Use Protocol from typing
    
    @abstractmethod
    def can_read(self, source: DataSource) -> bool:
        """Check if this reader can handle the given source."""
        # WHAT: Capability checking method
        # WHY: Chain of Responsibility pattern
        # HOW: Each reader returns True if it can handle source
        # BENEFIT: Runtime reader selection
        # USAGE: factory asks readers until one says True
        pass
    
    @abstractmethod
    def read(self, source: DataSource) -> pd.DataFrame:
        """Read data from the source."""
        # WHAT: Main read operation
        # WHY: Convert various formats to standard DataFrame
        # RETURN: pandas DataFrame
        # TRADE-OFF: Loads entire file (no streaming)
        # IMPROVEMENT: Add chunk_size parameter
        pass


class IDataProcessor(ABC):
    """Interface for data processing operations."""
    # WHAT: Interface for data cleaning/transformation
    # WHY: Separate processing logic from domain
    # PATTERN: Template Method potential
    
    @abstractmethod
    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean the data."""
        # WHAT: Data cleaning operation
        # RESPONSIBILITIES:
        #   - Handle missing values
        #   - Remove duplicates
        #   - Fix outliers
        # TRADE-OFF: No configuration in signature
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        # WHAT: Data transformation
        # RESPONSIBILITIES:
        #   - Encode categoricals
        #   - Scale numerics
        #   - Engineer features
        pass
    
    @abstractmethod
    def validate(self, data: pd.DataFrame) -> bool:
        """Validate the data quality."""
        # WHAT: Quality validation
        # RETURN: Boolean (pass/fail)
        # TRADE-OFF: Boolean doesn't provide details
        # BETTER: Return ValidationReport
        pass


class IEDAAnalyzer(ABC):
    """Interface for exploratory data analysis."""
    # WHAT: Interface for EDA operations
    # WHY: Separate analysis from domain
    
    @abstractmethod
    def analyze(self, data: ProcessedData) -> EDAReport:
        """Perform exploratory data analysis."""
        # WHAT: Generate statistical insights
        # PARAMETER: ProcessedData (not raw DataFrame)
        # BENEFIT: Access to metadata
        # RETURN: Rich EDAReport entity
        pass
    
    @abstractmethod
    def generate_visualizations(
        self, data: ProcessedData, output_dir: Path
    ) -> List[str]:
        """Generate visualization plots."""
        # WHAT: Create and save plots
        # WHY: Separate from analyze() for flexibility
        # RETURN: List of file paths
        # SIDE EFFECT: Writes files to disk
        pass


class IModelTrainer(ABC):
    """Interface for model training."""
    # WHAT: Interface for ML training
    # WHY: Support multiple algorithms
    
    @abstractmethod
    def train(self, data: ProcessedData, config: ModelConfig) -> TrainedModel:
        """Train a machine learning model."""
        # WHAT: Train model on data
        # PARAMETER: Rich ProcessedData + ModelConfig
        # RETURN: TrainedModel with metrics
        # RESPONSIBILITIES:
        #   - Feature/target split
        #   - Train/test split
        #   - Model instantiation
        #   - Training
        #   - Evaluation
        pass
    
    @abstractmethod
    def evaluate(self, model: TrainedModel, test_data: pd.DataFrame) -> dict:
        """Evaluate model performance."""
        # WHAT: Evaluate on new data
        # RETURN: Dictionary of metrics
        # TRADE-OFF: Dict not typed
        # BETTER: Return MetricsReport entity
        pass


class IPredictor(ABC):
    """Interface for making predictions."""
    # WHAT: Interface for inference
    # WHY: Separate training from prediction
    
    @abstractmethod
    def predict(self, model: TrainedModel, data: pd.DataFrame) -> Prediction:
        """Make predictions using the trained model."""
        # WHAT: Generate predictions
        # PARAMETER: TrainedModel + features
        # RETURN: Rich Prediction entity with confidence
        pass


class IModelRepository(ABC):
    """Interface for model persistence."""
    # WHAT: Interface for model storage
    # WHY: Abstract persistence mechanism
    # PATTERN: Repository Pattern
    
    @abstractmethod
    def save(self, model: TrainedModel, path: Path) -> None:
        """Save a trained model."""
        # WHAT: Persist model to disk
        # TRADE-OFF: No return value
        # BETTER: Return bool or Path
        pass
    
    @abstractmethod
    def load(self, path: Path) -> TrainedModel:
        """Load a trained model."""
        # WHAT: Load from disk
        # RETURN: TrainedModel entity
        # TRADE-OFF: Exception on missing file
        # BETTER: Return Optional[TrainedModel]
        pass
    
    @abstractmethod
    def list_models(self, directory: Path) -> List[str]:
        """List all available models."""
        # WHAT: Model discovery
        # RETURN: List of paths
        # TRADE-OFF: Returns strings not Path objects
        pass


class IDataRepository(ABC):
    """Interface for data persistence."""
    # WHAT: Interface for data storage
    # WHY: Abstract data persistence
    
    @abstractmethod
    def save(self, data: ProcessedData, path: Path) -> None:
        """Save processed data."""
        # WHAT: Persist processed data
        # BENEFIT: Reuse preprocessed data
        pass
    
    @abstractmethod
    def load(self, path: Path) -> ProcessedData:
        """Load processed data."""
        # WHAT: Load saved data
        # RETURN: ProcessedData with metadata
        pass
```

---

## Design Patterns

### 1. **Repository Pattern**
- **Interfaces**: IModelRepository, IDataRepository
- **Benefit**: Abstract storage mechanism
- **Allows**: Swap pickle for database

### 2. **Strategy Pattern**
- **Interfaces**: IDataReader, IModelTrainer
- **Benefit**: Interchangeable algorithms

### 3. **Dependency Inversion** (SOLID)
- **Principle**: Depend on abstractions
- **Benefit**: Domain independent of infrastructure

### 4. **Interface Segregation** (SOLID)
- **Principle**: Small, focused interfaces
- **Benefit**: Clients depend only on what they use

---

## Pros & Cons

### ✅ Pros

1. **Clean Architecture**: Domain defines ports
2. **Testability**: Easy to mock
3. **Flexibility**: Swap implementations
4. **Framework Independence**: No infrastructure dependencies

### ❌ Cons

1. **Boolean Returns**: validate() returns bool (not rich result)
2. **Dict Returns**: evaluate() returns dict (not typed)
3. **No Streaming**: read() loads entire file
4. **String Returns**: list_models() returns strings not Paths

---

**Lines**: 120  
**Interfaces**: 7  
**Methods**: 14  
**Dependencies**: 0 on infrastructure
