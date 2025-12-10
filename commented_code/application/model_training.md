# Model Training Use Case - Detailed Code Documentation

**File**: `src/application/use_cases/model_training.py`  
**Purpose**: Orchestrate model training and persistence workflow  
**Layer**: Application  
**Pattern**: Orchestration + Repository Pattern

---

## Key Code with Commentary

```python
class ModelTrainingUseCase:
    """Handles the model training workflow."""
    
    def __init__(
        self,
        trainer: IModelTrainer,
        repository: IModelRepository,
    ):
        # WHAT: DI with two dependencies
        # WHY: Trainer for training, repository for persistence
        # PATTERN: Constructor Injection
        # BENEFIT: Separately testable components
        self.trainer = trainer
        self.repository = repository
    
    def execute(
        self,
        data: ProcessedData,
        config: ModelConfig,
        save_model: bool = True,
        model_path: Optional[Path] = None,
    ) -> TrainedModel:
        # WHAT: Train and optionally save model
        # PARAMETERS:
        #   - data: ProcessedData (features + target)
        #   - config: ModelConfig (model_type + hyperparameters)
        #   - save_model: Boolean (optional persistence)
        #   - model_path: Optional custom path
        # RETURN: TrainedModel (model + metrics + metadata)
        
        logger.info(f"Starting model training workflow for {config.model_type}")
        
        # Train model
        trained_model = self.trainer.train(data, config)
        # WHAT: Delegate training to trainer
        # WHY: Separation of concerns
        # HOW: Trainer handles:
        #   - Feature/target split
        #   - Train/test split
        #   - Model instantiation
        #   - Training
        #   - Evaluation
        # RETURN: TrainedModel entity
        
        # Save model if requested
        if save_model:
            # WHAT: Optional persistence
            # WHY: Sometimes we just want to train (experiments)
            # USE CASE: Skip saving for quick tests
            
            if model_path is None:
                model_path = Path(f"models/{config.model_type}_model.pkl")
            # WHAT: Generate default path
            # WHY: Convenient default naming
            # FORMAT: models/linear_regression_model.pkl
            # TRADE-OFF: Could overwrite existing models
            # IMPROVEMENT: Add timestamp or version
            
            logger.info(f"Saving model to {model_path}")
            self.repository.save(trained_model, model_path)
            # WHAT: Persist to disk
            # WHY: Reuse trained models
            # DELEGATES: Repository handles serialization
        
        logger.info("Model training workflow completed")
        logger.info(f"Model metrics: {trained_model.metrics}")
        # WHAT: Log completion and metrics
        # WHY: Observability
        # SHOWS: accuracy, precision, recall, etc.
        
        return trained_model
```

---

## Pros & Cons

### ✅ Pros
- **Optional Saving**: Flexible for experiments
- **Default Path Generation**: Convenient naming
- **Metric Logging**: Immediate feedback
- **Clean Separation**: Training vs persistence

### ❌ Cons
- **Overwrite Risk**: Default path could replace existing models
- **No Versioning**: Can't track model versions
- **No Model Registry**: Manual path management
- **Hardcoded Directory**: "models/" not configurable

---

## Improvements

```python
# Better approach:
model_path = Path(f"models/{config.model_type}_{timestamp}.pkl")

# Or use model registry:
registry.save(trained_model, version="v1.0.0", tags=["production"])
```

---

**Total Lines**: 60  
**Complexity**: Low  
**Dependencies**: 2 (trainer, repository)
