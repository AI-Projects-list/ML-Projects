# ML Pipeline Use Case - Detailed Code Documentation

**File**: `src/application/use_cases/ml_pipeline.py`  
**Purpose**: Orchestrate complete end-to-end ML workflow  
**Layer**: Application  
**Pattern**: Facade Pattern + Orchestration

---

## Key Insights

```python
class MLPipelineUseCase:
    """Orchestrates the complete end-to-end ML pipeline."""
    # WHAT: Master orchestrator
    # WHY: Combine all use cases into single workflow
    # PATTERN: Facade Pattern
    # BENEFIT: Simple interface to complex subsystem
    
    def __init__(
        self,
        data_ingestion: DataIngestionUseCase,
        eda: EDAUseCase,
        model_training: ModelTrainingUseCase,
        prediction: PredictionUseCase,
    ):
        # WHAT: Inject ALL use cases
        # WHY: Compose complete workflow
        # PATTERN: Composition over inheritance
        # BENEFIT: Reuse existing use cases
        # COUNT: 4 dependencies (high coupling)
        # TRADE-OFF: Many dependencies vs duplication
        
        self.data_ingestion = data_ingestion
        self.eda = eda
        self.model_training = model_training
        self.prediction = prediction
    
    def execute(
        self,
        source: DataSource,
        model_config: ModelConfig,
        perform_eda: bool = True,
        eda_output_dir: Optional[Path] = None,
        model_output_path: Optional[Path] = None,
    ) -> dict:
        # WHAT: Execute 4-step pipeline
        # STEPS:
        #   1. Data Ingestion
        #   2. EDA (optional)
        #   3. Model Training
        #   4. Prediction
        # RETURN: Dictionary with all results
        # TRADE-OFF: Dict vs custom PipelineResult entity
        
        logger.info("=" * 60)
        logger.info("Starting End-to-End ML Pipeline")
        logger.info("=" * 60)
        # WHAT: Visual separator in logs
        # WHY: Clear pipeline start
        # BENEFIT: Easy to spot in log files
        
        results = {}
        # WHAT: Accumulator for results
        # WHY: Return all artifacts
        # KEYS: processed_data, eda_report, trained_model, predictions
        
        # Step 1: Data Ingestion
        logger.info("\n[1/4] Data Ingestion & Preprocessing")
        # WHAT: Step indicator
        # WHY: Show progress
        # FORMAT: [step/total] description
        
        logger.info("-" * 60)
        processed_data = self.data_ingestion.execute(source)
        results["processed_data"] = processed_data
        logger.info(f"✓ Processed data shape: {processed_data.data.shape}")
        # WHAT: Success indicator with details
        # WHY: Immediate feedback
        # SYMBOL: ✓ (checkmark)
        
        # Step 2: EDA (optional)
        if perform_eda:
            logger.info("\n[2/4] Exploratory Data Analysis")
            logger.info("-" * 60)
            eda_report = self.eda.execute(
                processed_data,
                generate_plots=True,
                output_dir=eda_output_dir,
            )
            results["eda_report"] = eda_report
            logger.info(f"✓ Generated {len(eda_report.insights)} insights")
            logger.info(f"✓ Created {len(eda_report.visualizations)} visualizations")
        else:
            logger.info("\n[2/4] Exploratory Data Analysis (SKIPPED)")
            # WHAT: Indicate skipped step
            # WHY: Show what didn't run
            # BENEFIT: Complete pipeline trace
        
        # Step 3: Model Training
        logger.info("\n[3/4] Model Training")
        logger.info("-" * 60)
        trained_model = self.model_training.execute(
            processed_data,
            model_config,
            save_model=True,
            model_path=model_output_path,
        )
        results["trained_model"] = trained_model
        logger.info(f"✓ Model trained: {model_config.model_type}")
        logger.info(f"✓ Metrics: {trained_model.metrics}")
        
        # Step 4: Generate predictions on training data (as example)
        logger.info("\n[4/4] Generating Predictions")
        logger.info("-" * 60)
        prediction = self.prediction.execute(
            processed_data.data,
            model=trained_model,
        )
        # NOTE: Predicts on training data
        # WHY: Example/demo purposes
        # PRODUCTION: Should use separate test data
        # TRADE-OFF: Not a real-world use case
        
        results["predictions"] = prediction
        logger.info(f"✓ Generated {len(prediction.predictions)} predictions")
        
        logger.info("\n" + "=" * 60)
        logger.info("ML Pipeline Completed Successfully!")
        logger.info("=" * 60)
        
        return results
        # WHAT: Return all artifacts
        # FORMAT: Dictionary
        # KEYS: processed_data, eda_report, trained_model, predictions
        # TRADE-OFF: Dict vs typed PipelineResult dataclass
```

---

## Design Pattern: Facade

This class is a **FACADE** over the complex subsystem:
- **Complex Subsystem**: 4 use cases with multiple dependencies
- **Simple Interface**: Single `execute()` method
- **Benefit**: Hide complexity from end users

---

## Pros & Cons

### ✅ Pros
- **One-Stop Shop**: Complete pipeline in one call
- **Excellent Logging**: Step indicators, progress, results
- **Flexible**: Optional EDA, custom paths
- **Composition**: Reuses existing use cases
- **Visual Feedback**: ✓ checkmarks, separators

### ❌ Cons
- **Dict Return**: Not typed (should be PipelineResult entity)
- **High Coupling**: Depends on 4 use cases
- **Training Data Prediction**: Step 4 not useful in production
- **No Error Recovery**: Failure stops entire pipeline
- **No Checkpointing**: Can't resume from failure

---

## Improvements

```python
@dataclass
class PipelineResult:
    """Typed pipeline result."""
    processed_data: ProcessedData
    eda_report: Optional[EDAReport]
    trained_model: TrainedModel
    predictions: Prediction

# Better return type:
def execute(...) -> PipelineResult:
    ...
    return PipelineResult(
        processed_data=processed_data,
        eda_report=eda_report,
        trained_model=trained_model,
        predictions=prediction,
    )

# Add checkpointing:
def execute(..., checkpoint_dir: Optional[Path] = None):
    if checkpoint_dir:
        save_checkpoint(step=1, data=processed_data)
```

---

**Total Lines**: 115  
**Complexity**: Medium  
**Dependencies**: 4 (high coupling)  
**Pattern**: Facade  
**Use Case**: End-user simplified interface
