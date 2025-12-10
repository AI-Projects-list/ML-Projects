# Model Training Use Case - Comprehensive Documentation

## File Information
- **Source File**: `src/application/use_cases/model_training.py`
- **Purpose**: Orchestrates the complete model training workflow with persistence
- **Layer**: Application Layer (Use Cases)
- **Pattern**: Use Case Pattern, Repository Pattern, Strategy Pattern

## Complete Annotated Code

```python
"""Use case for model training."""
# WHAT: Module-level docstring documenting the model training use case
# WHY: Provides clear documentation for developers and tools
# HOW: Python docstring convention with triple quotes
# BENEFIT: Improved discoverability, IDE support, documentation generation
# TRADE-OFF: Brief description - could expand to describe training workflow

from pathlib import Path
# WHAT: Import Path class for file system path manipulation
# WHY: Modern, object-oriented approach to handling file paths
# HOW: Import from pathlib standard library
# BENEFIT: Cross-platform compatibility, immutable paths, rich path operations
# TRADE-OFF: Additional import vs string paths, but benefits outweigh costs

from typing import Optional
# WHAT: Import Optional type hint for nullable parameters
# WHY: Type safety for parameters that can be None
# HOW: Import from typing standard library module
# BENEFIT: Clear API documentation, IDE auto-completion, type checking
# TRADE-OFF: Import overhead vs untyped, but improves code quality

from loguru import logger
# WHAT: Import pre-configured logger from loguru library
# WHY: Structured logging with beautiful formatting and automatic context
# HOW: Import logger singleton configured in infrastructure layer
# BENEFIT: Beautiful console output, automatic exception catching, structured logs
# TRADE-OFF: External dependency vs standard logging, but superior developer experience

from src.domain.entities import ModelConfig, ProcessedData, TrainedModel
# WHAT: Import domain entities for model training workflow
# WHY: Use domain models to encapsulate business data and enforce rules
# HOW: Import ModelConfig (configuration), ProcessedData (input), TrainedModel (output)
# BENEFIT: Type safety, business logic encapsulation, clear data contracts
# TRADE-OFF: More classes vs simple dicts, but enforces domain integrity

from src.domain.repositories import IModelRepository, IModelTrainer
# WHAT: Import interfaces (abstract base classes) for training and persistence
# WHY: Dependency inversion - depend on abstractions not implementations
# HOW: Import IModelRepository (persistence) and IModelTrainer (training) protocols
# BENEFIT: Testability (easy to mock), flexibility (swap implementations), loose coupling
# TRADE-OFF: Additional abstraction layer vs concrete dependencies, but enables SOLID


class ModelTrainingUseCase:
    # WHAT: Use case class orchestrating model training workflow
    # WHY: Application layer pattern to coordinate training and persistence
    # HOW: Class encapsulating model training business workflow
    # BENEFIT: Single responsibility (training only), testable, reusable
    # TRADE-OFF: Additional class vs procedural function, but better organization

    """Handles the model training workflow."""
    # WHAT: Class-level docstring describing responsibility
    # WHY: Documents class purpose for developers and tools
    # HOW: Concise single-line docstring
    # BENEFIT: Clear understanding of purpose, IDE support
    # TRADE-OFF: Brief description - could expand to describe workflow steps
    
    def __init__(
        # WHAT: Constructor method signature using multi-line parameter layout
        # WHY: Dependency injection pattern for testability and flexibility
        # HOW: Accept dependencies as constructor parameters
        # BENEFIT: Testable (inject mocks), flexible (swap implementations)
        # TRADE-OFF: More verbose than creating dependencies internally, but enables testing

        self,
        # WHAT: Reference to the instance being initialized
        # WHY: Python requirement for instance methods
        # HOW: First parameter of instance methods by convention
        # BENEFIT: Access to instance attributes and methods
        # TRADE-OFF: None - required by Python

        trainer: IModelTrainer,
        # WHAT: Injected dependency for model training
        # WHY: Interface dependency for training different model types
        # HOW: Type-hinted parameter expecting IModelTrainer implementation
        # BENEFIT: Testable (inject mock), swappable trainers (sklearn, xgboost, neural networks)
        # TRADE-OFF: Abstraction overhead vs concrete trainer, but enables flexibility

        repository: IModelRepository,
        # WHAT: Injected dependency for model persistence
        # WHY: Repository pattern for saving/loading trained models
        # HOW: Type-hinted parameter expecting IModelRepository implementation
        # BENEFIT: Testable (inject mock), flexible storage (filesystem, cloud, database)
        # TRADE-OFF: Additional abstraction vs direct file I/O, but enables clean architecture

    ):
        """
        Initialize model training use case.
        
        Args:
            trainer: Model trainer implementation
            repository: Model repository for persistence
        """
        # WHAT: Constructor docstring documenting parameters
        # WHY: Clear documentation for dependency injection requirements
        # HOW: Google-style docstring with Args section
        # BENEFIT: IDE auto-complete, clear API documentation
        # TRADE-OFF: Verbose documentation vs brief comment, but improves usability

        self.trainer = trainer
        # WHAT: Store trainer as instance attribute
        # WHY: Make trainer available throughout use case lifecycle
        # HOW: Assign injected trainer to instance variable
        # BENEFIT: Access trainer in execute method without passing as parameter
        # TRADE-OFF: State management (mutable instance) vs stateless function, but appropriate

        self.repository = repository
        # WHAT: Store repository as instance attribute
        # WHY: Make repository available for model persistence
        # HOW: Assign injected repository to instance variable
        # BENEFIT: Access repository in execute method, reuse across calls
        # TRADE-OFF: Holds reference to repository vs creating on-demand, but enables reuse
    
    def execute(
        # WHAT: Main execution method signature for training workflow
        # WHY: Execute is standard naming for use case entry points
        # HOW: Public method coordinating training and persistence
        # BENEFIT: Clear entry point, standardized interface
        # TRADE-OFF: Generic name vs specific like "train_model", but consistent pattern

        self,
        # WHAT: Instance reference for accessing trainer and repository
        # WHY: Required for Python instance methods
        # HOW: Access self.trainer and self.repository within method
        # BENEFIT: Access to injected dependencies
        # TRADE-OFF: None - required by Python

        data: ProcessedData,
        # WHAT: Domain entity containing preprocessed training data
        # WHY: Use rich domain object with data, metadata, and processing history
        # HOW: Type-hinted parameter expecting ProcessedData entity
        # BENEFIT: Type safety, access to metadata and lineage, validation
        # TRADE-OFF: Domain object vs simple DataFrame, but provides comprehensive context

        config: ModelConfig,
        # WHAT: Domain entity containing model configuration
        # WHY: Encapsulate model type, hyperparameters, and training settings
        # HOW: Type-hinted parameter expecting ModelConfig entity
        # BENEFIT: Type safety, validation, clear configuration structure
        # TRADE-OFF: Domain object vs simple dict, but enforces configuration integrity

        save_model: bool = True,
        # WHAT: Flag to enable/disable model persistence
        # WHY: Allow flexible workflow - sometimes don't want to save (e.g., experiments)
        # HOW: Boolean parameter with default True (saving is recommended)
        # BENEFIT: Flexibility to skip saving, useful for quick experiments
        # TRADE-OFF: Additional parameter vs always saving, but provides control

        model_path: Optional[Path] = None,
        # WHAT: Optional file path for saving trained model
        # WHY: Allow caller to specify custom save location
        # HOW: Optional type hint (Path or None) with default None
        # BENEFIT: Flexibility in save location, None triggers default behavior
        # TRADE-OFF: Union type complexity vs always requiring Path, but improves usability

    ) -> TrainedModel:
        # WHAT: Return type annotation specifying TrainedModel entity
        # WHY: Type safety and IDE auto-completion for return value
        # HOW: Arrow notation indicating method returns TrainedModel instance
        # BENEFIT: Type checking, clear contract, IDE support
        # TRADE-OFF: Annotation overhead vs untyped, but improves reliability

        """
        Execute the model training workflow.

        Args:
            data: Processed data for training
            config: Model configuration
            save_model: Whether to save the trained model
            model_path: Path for saving the model

        Returns:
            Trained model with evaluation metrics
        """
        # WHAT: Comprehensive docstring documenting execute method
        # WHY: Clear documentation of training parameters and return value
        # HOW: Google-style docstring with Args and Returns sections
        # BENEFIT: IDE support, clear API documentation, maintainability
        # TRADE-OFF: Verbose vs brief comments, but essential for public API

        logger.info(f"Starting model training workflow for {config.model_type}")
        # WHAT: Log informational message about training start with model type
        # WHY: Observability - track which model is being trained
        # HOW: Use logger.info with f-string including model type from config
        # BENEFIT: Troubleshooting, monitoring, audit trail, know what's training
        # TRADE-OFF: I/O overhead vs silent execution, but critical for production
        
        # Train model
        # WHAT: Comment indicating the training step
        # WHY: Mark distinct workflow phase
        # HOW: Single-line comment above training code
        # BENEFIT: Code organization, clear workflow structure
        # TRADE-OFF: Additional comment vs self-documenting code, but aids readability

        trained_model = self.trainer.train(data, config)
        # WHAT: Train model using trainer implementation
        # WHY: Core functionality - fit model to data using specified configuration
        # HOW: Call trainer's train method passing data and config
        # BENEFIT: Model is trained and ready for predictions, metrics calculated
        # TRADE-OFF: Training time (can be long) vs skipping, but essential for ML

        # Save model if requested
        # WHAT: Comment indicating conditional persistence step
        # WHY: Mark model saving logic separately
        # HOW: Single-line comment above conditional block
        # BENEFIT: Clear code structure, easy to navigate
        # TRADE-OFF: Comment overhead vs code only, but improves readability

        if save_model:
            # WHAT: Conditional execution of model persistence
            # WHY: Allow skipping persistence for experiments or temporary models
            # HOW: Check save_model boolean flag
            # BENEFIT: Flexible workflow, avoid unnecessary I/O for experiments
            # TRADE-OFF: Conditional logic vs always save, but provides control

            if model_path is None:
                # WHAT: Check if save path was provided
                # WHY: Provide default path when caller doesn't specify
                # HOW: Test if model_path is None
                # BENEFIT: Sensible default behavior, caller doesn't need to know structure
                # TRADE-OFF: Hardcoded default vs requiring caller to specify, but improves usability

                model_path = Path(f"models/{config.model_type}_model.pkl")
                # WHAT: Set default model save path based on model type
                # WHY: Organized naming convention when not specified
                # HOW: Create Path with models directory and model_type from config
                # BENEFIT: Consistent file organization, descriptive filenames
                # TRADE-OFF: Hardcoded directory structure vs configuration, but reasonable default
            
            logger.info(f"Saving model to {model_path}")
            # WHAT: Log model save operation with path
            # WHY: Observability - inform user where model is being saved
            # HOW: Info-level log with f-string including model_path
            # BENEFIT: User knows where to find model, troubleshooting
            # TRADE-OFF: Logging I/O vs silent save, but helpful for users

            self.repository.save(trained_model, model_path)
            # WHAT: Persist trained model using repository
            # WHY: Save model for later use (prediction, deployment, sharing)
            # HOW: Call repository's save method with model and path
            # BENEFIT: Model preserved for reuse, deployment, version control
            # TRADE-OFF: Disk space and I/O time vs not saving, but essential for production

        logger.info("Model training workflow completed")
        # WHAT: Log successful workflow completion
        # WHY: Observability - confirm success for monitoring/debugging
        # HOW: Info-level log message
        # BENEFIT: Clear success indication, audit trail
        # TRADE-OFF: Logging I/O vs silent success, but important for production

        logger.info(f"Model metrics: {trained_model.metrics}")
        # WHAT: Log model evaluation metrics
        # WHY: Provide immediate feedback on model performance
        # HOW: Info-level log with f-string showing metrics dict
        # BENEFIT: Quick performance assessment, no need to inspect model object
        # TRADE-OFF: Additional logging vs minimal output, but very useful context

        return trained_model
        # WHAT: Return the trained model entity
        # WHY: Provide trained model to caller for predictions or evaluation
        # HOW: Return TrainedModel instance with model, config, and metrics
        # BENEFIT: Rich result object with model and metadata, clear API
        # TRADE-OFF: Domain object vs raw model, but provides comprehensive information
```

---

## Design Patterns Used

### 1. **Use Case Pattern** (Application Layer)
- **Purpose**: Encapsulates business workflow for model training
- **Implementation**: `ModelTrainingUseCase` class with `execute` method
- **Benefits**: Single responsibility, testable, reusable across interfaces
- **Trade-offs**: Additional abstraction vs direct implementation

### 2. **Dependency Injection Pattern**
- **Purpose**: Invert control - depend on abstractions (trainer, repository)
- **Implementation**: Constructor accepts `trainer` and `repository` dependencies
- **Benefits**: Testability (inject mocks), flexibility (swap implementations)
- **Trade-offs**: More verbose initialization vs creating dependencies internally

### 3. **Repository Pattern**
- **Purpose**: Abstract data persistence logic
- **Implementation**: `IModelRepository` interface for save/load operations
- **Benefits**: Flexible storage (filesystem, cloud, database), testable, clean separation
- **Trade-offs**: Additional abstraction layer vs direct file I/O

### 4. **Strategy Pattern**
- **Purpose**: Allow different training strategies via IModelTrainer interface
- **Implementation**: Trainer interface enables swapping training implementations
- **Benefits**: Support multiple ML libraries (sklearn, xgboost, tensorflow), runtime selection
- **Trade-offs**: Additional abstraction vs hardcoded trainer

### 5. **Default Parameter Pattern**
- **Purpose**: Provide sensible defaults for optional parameters
- **Implementation**: `save_model=True`, `model_path=None` with default assignment
- **Benefits**: Simpler API for common cases, flexibility for advanced use
- **Trade-offs**: Hidden defaults vs explicit parameters

---

## Pros & Cons

### Pros ✅

1. **Simple & Clean API**
   - Minimal required parameters (data and config)
   - Clear entry point with `execute` method
   - Sensible defaults for persistence

2. **Flexible Persistence**
   - Optional saving via flag
   - Custom save path support with default
   - Easy to skip for experiments

3. **Strong Separation of Concerns**
   - Use case orchestrates workflow
   - Trainer performs actual training
   - Repository handles persistence

4. **Excellent Observability**
   - Logging at workflow start/end
   - Logs model type and metrics
   - Logs save path for traceability

5. **Testability**
   - Trainer and repository injected (easy to mock)
   - Clear inputs (ProcessedData, ModelConfig) and outputs (TrainedModel)
   - No hidden dependencies

6. **Rich Return Value**
   - TrainedModel entity with model, config, and metrics
   - Immediate access to performance metrics
   - Complete model metadata

### Cons ❌

1. **No Training Callbacks**
   - Cannot track training progress
   - No intermediate checkpoints
   - No early stopping hooks

2. **No Error Handling**
   - No try-except around training
   - No handling of training failures
   - Exceptions propagate to caller

3. **Limited Validation**
   - No data validation before training
   - No config validation
   - Assumes data is ready for training

4. **No Model Versioning**
   - Single save path (no automatic versioning)
   - Can overwrite existing models
   - No model registry integration

5. **Directory Creation Not Guaranteed**
   - Assumes models directory exists
   - May fail if directory doesn't exist
   - No explicit directory creation

6. **Single Training Run**
   - No support for cross-validation
   - No hyperparameter tuning loops
   - One-shot training only

---

## Usage Examples

### Example 1: Basic Model Training with Save
```python
from pathlib import Path
from src.domain.entities import ModelConfig, ProcessedData
from src.infrastructure.ml.model_trainer import ModelTrainer
from src.infrastructure.ml.model_repository import ModelRepository
from src.application.use_cases.model_training import ModelTrainingUseCase

# Setup dependencies
trainer = ModelTrainer()
repository = ModelRepository()
use_case = ModelTrainingUseCase(trainer, repository)

# Configure model
config = ModelConfig(
    model_type="random_forest",
    hyperparameters={"n_estimators": 100, "max_depth": 10}
)

# Train and save model
trained_model = use_case.execute(processed_data, config)

print(f"Metrics: {trained_model.metrics}")
print(f"Model saved to: models/random_forest_model.pkl")
# Output:
# Metrics: {'accuracy': 0.95, 'f1_score': 0.93}
# Model saved to: models/random_forest_model.pkl
```

### Example 2: Training Without Saving (Experiments)
```python
# Quick experiment - don't save model
config = ModelConfig(
    model_type="logistic_regression",
    hyperparameters={"C": 1.0, "penalty": "l2"}
)

trained_model = use_case.execute(
    processed_data,
    config,
    save_model=False  # Skip persistence for experiment
)

print(f"Experiment metrics: {trained_model.metrics}")
# Model not saved - good for quick tests
```

### Example 3: Custom Save Path
```python
# Save to custom location with versioning
custom_path = Path("models/v2.0/random_forest_2024_01_15.pkl")

trained_model = use_case.execute(
    processed_data,
    config,
    save_model=True,
    model_path=custom_path
)

print(f"Model saved to: {custom_path}")
```

### Example 4: Different Model Types
```python
# Train multiple model types
models = ["random_forest", "gradient_boosting", "logistic_regression"]

for model_type in models:
    config = ModelConfig(model_type=model_type)
    trained_model = use_case.execute(processed_data, config)
    
    print(f"{model_type} - Accuracy: {trained_model.metrics['accuracy']}")
# Output:
# random_forest - Accuracy: 0.95
# gradient_boosting - Accuracy: 0.96
# logistic_regression - Accuracy: 0.92
```

### Example 5: Testing with Mocks
```python
from unittest.mock import Mock
from src.domain.entities import TrainedModel

# Create mock dependencies
mock_trainer = Mock()
mock_repository = Mock()
mock_model = TrainedModel(
    model=Mock(),
    config=config,
    metrics={"accuracy": 0.99}
)
mock_trainer.train.return_value = mock_model

# Test use case with mocks
use_case = ModelTrainingUseCase(mock_trainer, mock_repository)
result = use_case.execute(processed_data, config)

# Assert expectations
assert result.metrics["accuracy"] == 0.99
mock_trainer.train.assert_called_once_with(processed_data, config)
mock_repository.save.assert_called_once()
```

### Example 6: Integration with Full Pipeline
```python
from src.application.use_cases.data_ingestion import DataIngestionUseCase
from src.application.use_cases.eda import EDAUseCase

# Complete workflow: ingest → EDA → train
# 1. Ingest data
ingestion = DataIngestionUseCase(reader_factory, processor)
processed_data = ingestion.execute(source)

# 2. Perform EDA
eda = EDAUseCase(analyzer)
eda_report = eda.execute(processed_data)

# 3. Train model based on EDA insights
config = ModelConfig(model_type="random_forest")
training = ModelTrainingUseCase(trainer, repository)
trained_model = training.execute(processed_data, config)

print(f"Pipeline complete - Accuracy: {trained_model.metrics['accuracy']}")
```

---

## Key Takeaways

### What This Use Case Does
Orchestrates model training workflow: train model → optionally save to disk, returning trained model with metrics.

### Why This Architecture
- **Clean Architecture**: Separates use case (orchestration) from trainer (training) and repository (persistence)
- **Testability**: Dependencies injected, easy to mock
- **Flexibility**: Optional saving, custom paths, swappable trainers

### How It Works
1. Accept `ProcessedData`, `ModelConfig`, and optional flags
2. Call trainer to train model with data and configuration
3. Optionally save trained model to specified or default path
4. Log metrics for immediate feedback
5. Return rich `TrainedModel` entity with model and metadata

### Benefits
- **Simple API**: Minimal required parameters, sensible defaults
- **Observable**: Logging at key workflow points with metrics
- **Reusable**: Can be called from CLI, API, notebooks
- **Persistent**: Model saved for deployment and reuse

### Trade-offs
- **No Progress**: Cannot track long-running training
- **No Validation**: Assumes data is ready for training
- **Single Run**: No cross-validation or hyperparameter tuning

---

## Related Files
- **Domain Entities**: `src/domain/entities.py` - `ModelConfig`, `ProcessedData`, `TrainedModel`
- **Repository Interfaces**: `src/domain/repositories.py` - `IModelTrainer`, `IModelRepository`
- **Trainer Implementation**: `src/infrastructure/ml/model_trainer.py`
- **Repository Implementation**: `src/infrastructure/ml/model_repository.py`
- **Data Ingestion**: `src/application/use_cases/data_ingestion.py` - Produces ProcessedData input

---

*This documentation provides comprehensive line-by-line annotations with WHAT/WHY/HOW/BENEFIT/TRADE-OFF analysis for each line of code, design patterns explanation, pros & cons analysis, and practical usage examples.*
