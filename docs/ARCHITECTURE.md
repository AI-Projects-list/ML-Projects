# Architecture Overview

This document provides a detailed overview of the ML Ollama architecture.

## Table of Contents

1. [Architecture Principles](#architecture-principles)
2. [Layer Details](#layer-details)
3. [Dependency Flow](#dependency-flow)
4. [Key Components](#key-components)
5. [Design Patterns](#design-patterns)

## Architecture Principles

### Clean Architecture

The project follows Clean Architecture principles:

- **Independence of Frameworks**: Business logic doesn't depend on external frameworks
- **Testability**: Business logic can be tested without UI, database, or external services
- **Independence of UI**: UI can change without changing business logic
- **Independence of Database**: Can swap data sources without affecting business logic
- **Independence of External Services**: Business rules don't know about the outside world

### Hexagonal Architecture (Ports & Adapters)

- **Ports**: Interfaces defined in the domain layer
- **Adapters**: Implementations in the infrastructure layer
- **Core**: Business logic in domain and application layers

## Layer Details

### 1. Domain Layer (Core)

**Purpose**: Contains business logic and rules. No external dependencies.

**Components**:
- `entities.py`: Core business objects (DataSource, ProcessedData, TrainedModel, etc.)
- `value_objects.py`: Immutable values (ColumnSchema, DataQualityMetrics)
- `repositories.py`: Port interfaces (IDataReader, IDataProcessor, IModelTrainer, etc.)

**Rules**:
- ✅ Can reference other domain objects
- ❌ Cannot reference application, infrastructure, or presentation layers
- ❌ No framework dependencies

### 2. Application Layer

**Purpose**: Contains use cases and orchestrates business workflows.

**Components**:
- `use_cases/data_ingestion.py`: Data loading and preprocessing workflow
- `use_cases/eda.py`: Exploratory data analysis workflow
- `use_cases/model_training.py`: Model training workflow
- `use_cases/prediction.py`: Prediction workflow
- `use_cases/ml_pipeline.py`: End-to-end pipeline orchestration

**Rules**:
- ✅ Can reference domain layer
- ✅ Can reference repository interfaces
- ❌ Cannot reference infrastructure implementations directly
- ❌ Cannot reference presentation layer

### 3. Infrastructure Layer

**Purpose**: Implements external interfaces and adapters.

**Components**:

#### Data Readers
- `csv_reader.py`: CSV file reader
- `text_reader.py`: Text file reader
- `pdf_reader.py`: PDF document reader
- `scanned_pdf_reader.py`: OCR-based PDF reader
- `factory.py`: Reader factory

#### Processing
- `data_processor.py`: Data cleaning and transformation
- `eda_analyzer.py`: Exploratory data analysis

#### ML
- `model_trainer.py`: Model training implementation
- `predictor.py`: Prediction service
- `model_repository.py`: Model persistence

#### Configuration
- `settings.py`: Configuration management
- `logging.py`: Logging setup
- `container.py`: Dependency injection

**Rules**:
- ✅ Can reference domain layer (ports)
- ✅ Can reference application layer
- ✅ Can use external frameworks
- ❌ Should not reference presentation layer

### 4. Presentation Layer

**Purpose**: User interfaces and input/output handling.

**Components**:
- `cli.py`: Command-line interface using Typer

**Rules**:
- ✅ Can reference all other layers
- ✅ Handles user interaction
- ❌ Should not contain business logic

## Dependency Flow

```
┌─────────────────────────────────────────┐
│         Presentation Layer              │
│            (CLI, API)                   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│       Application Layer                 │
│         (Use Cases)                     │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│       Domain Layer (Core)               │
│  (Entities, Value Objects, Interfaces)  │
└──────────────▲──────────────────────────┘
               │
               │ (implements)
               │
┌──────────────┴──────────────────────────┐
│     Infrastructure Layer                │
│  (Concrete Implementations, Adapters)   │
└─────────────────────────────────────────┘
```

**Key Rule**: Dependencies point inward. Inner layers never depend on outer layers.

## Key Components

### Data Flow

```
1. User Input (CLI/API)
   │
   ▼
2. Use Case (Application Layer)
   │
   ▼
3. Repository Interface (Domain Layer)
   │
   ▼
4. Concrete Implementation (Infrastructure Layer)
   │
   ▼
5. External System (File, Database, API)
```

### Pipeline Execution

```
DataSource → DataReader → RawData
    │
    ▼
DataProcessor → ProcessedData
    │
    ▼
EDAAnalyzer → EDAReport
    │
    ▼
ModelTrainer → TrainedModel
    │
    ▼
Predictor → Predictions
```

## Design Patterns

### 1. Repository Pattern

**Purpose**: Abstract data access

**Example**:
```python
# Domain (interface)
class IModelRepository(ABC):
    @abstractmethod
    def save(self, model: TrainedModel, path: Path) -> None:
        pass

# Infrastructure (implementation)
class ModelRepository(IModelRepository):
    def save(self, model: TrainedModel, path: Path) -> None:
        # Actual implementation
        pass
```

### 2. Factory Pattern

**Purpose**: Create objects without specifying exact class

**Example**:
```python
class DataReaderFactory:
    def get_reader(self, source: DataSource) -> IDataReader:
        for reader in self.readers:
            if reader.can_read(source):
                return reader
        raise ValueError("No suitable reader")
```

### 3. Strategy Pattern

**Purpose**: Pluggable algorithms

**Example**:
```python
# Different readers for different formats
csv_reader = CSVDataReader()
pdf_reader = PDFDataReader()
ocr_reader = ScannedPDFDataReader()
```

### 4. Dependency Injection

**Purpose**: Loose coupling, testability

**Example**:
```python
class Container:
    def __init__(self, settings: Settings):
        self.settings = settings
    
    @property
    def ml_pipeline_use_case(self) -> MLPipelineUseCase:
        return MLPipelineUseCase(
            data_ingestion=self.data_ingestion_use_case,
            eda=self.eda_use_case,
            # ... inject dependencies
        )
```

### 5. Use Case Pattern

**Purpose**: Encapsulate business workflows

**Example**:
```python
class DataIngestionUseCase:
    def execute(self, source: DataSource) -> ProcessedData:
        # Orchestrate the workflow
        reader = self.reader_factory.get_reader(source)
        data = reader.read(source)
        cleaned = self.processor.clean(data)
        return ProcessedData(data=cleaned, source=source)
```

## Benefits

### Scalability
- Easy to add new data sources
- Easy to add new models
- Easy to add new processing steps

### Maintainability
- Clear separation of concerns
- Each layer has specific responsibility
- Changes in one layer don't affect others

### Testability
- Business logic isolated from external dependencies
- Mock interfaces for testing
- Test each layer independently

### Flexibility
- Swap implementations without changing core logic
- Support multiple UIs (CLI, API, Web)
- Change data storage without affecting business rules

## Extension Points

### Add New Data Format

1. Create reader implementing `IDataReader`
2. Register with `DataReaderFactory`
3. No changes to domain or application layers

### Add New ML Model

1. Add model to `ModelTrainer.SUPPORTED_MODELS`
2. Optionally add custom config
3. No changes to domain or use cases

### Add New Use Case

1. Create new use case class in `application/use_cases/`
2. Inject required repositories
3. Wire in container

### Add New UI

1. Create new presentation layer (e.g., REST API)
2. Use existing use cases
3. No changes to business logic

---

This architecture ensures the project remains maintainable and scalable as it grows.
