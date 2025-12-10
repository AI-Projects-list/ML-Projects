# ML Pipeline Use Case - Comprehensive Documentation

## File Information
- **Source File**: `src/application/use_cases/ml_pipeline.py`
- **Purpose**: Orchestrates end-to-end ML pipeline from data ingestion to predictions
- **Layer**: Application Layer (Use Cases)
- **Pattern**: Facade Pattern, Composite Pattern, Orchestrator Pattern

## Complete Annotated Code

```python
"""End-to-end ML pipeline use case."""
# WHAT: Module docstring for ML pipeline orchestrator
# WHY: Documents complete workflow coordination
# HOW: Python docstring convention
# BENEFIT: Clear module purpose
# TRADE-OFF: Brief - could detail pipeline steps

from pathlib import Path
# WHAT: Import Path for file operations
# WHY: Modern path handling
# HOW: Standard library pathlib
# BENEFIT: Cross-platform, immutable paths
# TRADE-OFF: Import vs string paths

from typing import Optional
# WHAT: Import Optional type hint
# WHY: Type safety for nullable parameters
# HOW: Typing module
# BENEFIT: Clear optional parameters
# TRADE-OFF: Annotation overhead

from loguru import logger
# WHAT: Import logger
# WHY: Comprehensive pipeline logging
# HOW: Loguru singleton
# BENEFIT: Beautiful output, observability
# TRADE-OFF: External dependency

from src.domain.entities import DataSource, ModelConfig, Prediction
# WHAT: Import domain entities for pipeline I/O
# WHY: Type-safe pipeline contracts
# HOW: Import input (DataSource, ModelConfig) and output (Prediction)
# BENEFIT: Strong typing, validation
# TRADE-OFF: Multiple imports

from src.application.use_cases.data_ingestion import DataIngestionUseCase
# WHAT: Import data ingestion use case
# WHY: First pipeline step - load and preprocess data
# HOW: Import from application layer
# BENEFIT: Reusable component, tested independently
# TRADE-OFF: Dependency on another use case

from src.application.use_cases.eda import EDAUseCase
# WHAT: Import EDA use case
# WHY: Second pipeline step - analyze data
# HOW: Import from application layer
# BENEFIT: Optional analysis step
# TRADE-OFF: Additional dependency

from src.application.use_cases.model_training import ModelTrainingUseCase
# WHAT: Import training use case
# WHY: Third pipeline step - train model
# HOW: Import from application layer
# BENEFIT: Reusable training logic
# TRADE-OFF: Pipeline coupling

from src.application.use_cases.prediction import PredictionUseCase
# WHAT: Import prediction use case
# WHY: Fourth pipeline step - generate predictions
# HOW: Import from application layer
# BENEFIT: Consistent prediction interface
# TRADE-OFF: Dependency chain


class MLPipelineUseCase:
    # WHAT: Orchestrator class for complete ML workflow
    # WHY: Coordinate multiple use cases into pipeline
    # HOW: Facade pattern over individual use cases
    # BENEFIT: Single entry point for full workflow
    # TRADE-OFF: Complexity in one class

    """Orchestrates the complete end-to-end ML pipeline."""
    # WHAT: Class docstring
    # WHY: Document orchestration purpose
    # HOW: Single-line description
    # BENEFIT: Clear responsibility
    # TRADE-OFF: Could detail 4-step workflow

    def __init__(
        # WHAT: Constructor accepting all use case dependencies
        # WHY: Dependency injection for testability
        # HOW: Accept 4 use cases as parameters
        # BENEFIT: Testable, flexible composition
        # TRADE-OFF: Many dependencies to inject

        self,
        # WHAT: Instance reference
        # WHY: Python requirement
        # HOW: Standard self parameter
        # BENEFIT: Access to attributes
        # TRADE-OFF: None - required

        data_ingestion: DataIngestionUseCase,
        # WHAT: Injected data ingestion use case
        # WHY: Step 1 - load and preprocess
        # HOW: Use case dependency
        # BENEFIT: Reusable, tested component
        # TRADE-OFF: Coupling to use case

        eda: EDAUseCase,
        # WHAT: Injected EDA use case
        # WHY: Step 2 - analyze data
        # HOW: Use case dependency
        # BENEFIT: Optional analysis
        # TRADE-OFF: Additional dependency

        model_training: ModelTrainingUseCase,
        # WHAT: Injected training use case
        # WHY: Step 3 - train model
        # HOW: Use case dependency
        # BENEFIT: Consistent training
        # TRADE-OFF: Pipeline coupling

        prediction: PredictionUseCase,
        # WHAT: Injected prediction use case
        # WHY: Step 4 - generate predictions
        # HOW: Use case dependency
        # BENEFIT: Reusable predictions
        # TRADE-OFF: Full pipeline dependency

    ):
        """
        Initialize ML pipeline use case.

        Args:
            data_ingestion: Data ingestion use case
            eda: EDA use case
            model_training: Model training use case
            prediction: Prediction use case
        """
        # WHAT: Constructor docstring
        # WHY: Document 4 dependencies
        # HOW: Google-style Args section
        # BENEFIT: Clear initialization requirements
        # TRADE-OFF: Verbose documentation

        self.data_ingestion = data_ingestion
        # WHAT: Store ingestion use case
        # WHY: Access in execute method
        # HOW: Instance attribute
        # BENEFIT: Pipeline step 1
        # TRADE-OFF: State management

        self.eda = eda
        # WHAT: Store EDA use case
        # WHY: Optional analysis step
        # HOW: Instance attribute
        # BENEFIT: Pipeline step 2
        # TRADE-OFF: Memory reference

        self.model_training = model_training
        # WHAT: Store training use case
        # WHY: Model training step
        # HOW: Instance attribute
        # BENEFIT: Pipeline step 3
        # TRADE-OFF: Coupling

        self.prediction = prediction
        # WHAT: Store prediction use case
        # WHY: Final prediction step
        # HOW: Instance attribute
        # BENEFIT: Pipeline step 4
        # TRADE-OFF: Dependency chain

    def execute(
        # WHAT: Main pipeline execution method
        # WHY: Orchestrate all 4 steps
        # HOW: Sequential execution with results passing
        # BENEFIT: Complete workflow automation
        # TRADE-OFF: Long method, many responsibilities

        self,
        # WHAT: Instance reference
        # WHY: Access use case dependencies
        # HOW: Standard parameter
        # BENEFIT: Access to all 4 use cases
        # TRADE-OFF: None - required

        source: DataSource,
        # WHAT: Input data source
        # WHY: Starting point for pipeline
        # HOW: Domain entity parameter
        # BENEFIT: Type-safe input
        # TRADE-OFF: Required parameter

        model_config: ModelConfig,
        # WHAT: Model configuration
        # WHY: Specify model type and hyperparameters
        # HOW: Domain entity parameter
        # BENEFIT: Type-safe config
        # TRADE-OFF: Must be provided

        perform_eda: bool = True,
        # WHAT: Flag to enable/disable EDA
        # WHY: Optional analysis step
        # HOW: Boolean with True default
        # BENEFIT: Flexible pipeline
        # TRADE-OFF: Conditional logic

        eda_output_dir: Optional[Path] = None,
        # WHAT: Optional EDA output directory
        # WHY: Customize visualization location
        # HOW: Optional Path parameter
        # BENEFIT: Flexible output
        # TRADE-OFF: Additional parameter

        model_output_path: Optional[Path] = None,
        # WHAT: Optional model save path
        # WHY: Customize model location
        # HOW: Optional Path parameter
        # BENEFIT: Flexible persistence
        # TRADE-OFF: Many optional parameters

    ) -> dict:
        # WHAT: Return dictionary of all results
        # WHY: Provide access to all pipeline outputs
        # HOW: Dict return type
        # BENEFIT: Comprehensive results
        # TRADE-OFF: Untyped dict vs custom entity

        """
        Execute the complete ML pipeline.

        Args:
            source: Data source to process
            model_config: Model configuration
            perform_eda: Whether to perform EDA
            eda_output_dir: Directory for EDA outputs
            model_output_path: Path for saving trained model

        Returns:
            Dictionary containing all pipeline results
        """
        # WHAT: Method docstring
        # WHY: Document complete pipeline
        # HOW: Args and Returns sections
        # BENEFIT: Clear API documentation
        # TRADE-OFF: Verbose

        logger.info("=" * 60)
        # WHAT: Log visual separator
        # WHY: Clear pipeline start
        # HOW: Repeated equals signs
        # BENEFIT: Visual distinction in logs
        # TRADE-OFF: Cosmetic logging

        logger.info("Starting End-to-End ML Pipeline")
        # WHAT: Log pipeline initiation
        # WHY: Track workflow start
        # HOW: Info-level message
        # BENEFIT: Observability
        # TRADE-OFF: Logging overhead

        logger.info("=" * 60)
        # WHAT: Closing separator
        # WHY: Visual framing
        # HOW: Repeated equals
        # BENEFIT: Clear section
        # TRADE-OFF: Extra log line

        results = {}
        # WHAT: Initialize results dictionary
        # WHY: Accumulate all pipeline outputs
        # HOW: Empty dict
        # BENEFIT: Collect all results
        # TRADE-OFF: Mutable state

        # Step 1: Data Ingestion
        # WHAT: Comment marking step 1
        # WHY: Clear pipeline phase
        # HOW: Inline comment
        # BENEFIT: Code organization
        # TRADE-OFF: Additional comment

        logger.info("\n[1/4] Data Ingestion & Preprocessing")
        # WHAT: Log step 1 with numbering
        # WHY: Track pipeline progress
        # HOW: Formatted step indicator
        # BENEFIT: Progress tracking
        # TRADE-OFF: Hardcoded step count

        logger.info("-" * 60)
        # WHAT: Visual separator
        # WHY: Distinguish steps
        # HOW: Repeated dashes
        # BENEFIT: Log readability
        # TRADE-OFF: Cosmetic overhead

        processed_data = self.data_ingestion.execute(source)
        # WHAT: Execute data ingestion
        # WHY: Load and preprocess data
        # HOW: Call ingestion use case
        # BENEFIT: Clean, processed data
        # TRADE-OFF: Time for I/O and processing

        results["processed_data"] = processed_data
        # WHAT: Store processed data in results
        # WHY: Make available for caller
        # HOW: Dict assignment
        # BENEFIT: Access to intermediate results
        # TRADE-OFF: Memory for multiple results

        logger.info(f"✓ Processed data shape: {processed_data.data.shape}")
        # WHAT: Log data dimensions with checkmark
        # WHY: Confirm success and provide stats
        # HOW: Unicode checkmark, f-string with shape
        # BENEFIT: Visual success indicator, data stats
        # TRADE-OFF: Unicode may not display everywhere
        
        # Step 2: EDA (optional)
        # WHAT: Comment for optional step 2
        # WHY: Mark conditional analysis
        # HOW: Inline comment
        # BENEFIT: Clear step separation
        # TRADE-OFF: Additional comment

        if perform_eda:
            # WHAT: Conditional EDA execution
            # WHY: Optional analysis step
            # HOW: Boolean flag check
            # BENEFIT: Flexible pipeline
            # TRADE-OFF: Conditional complexity

            logger.info("\n[2/4] Exploratory Data Analysis")
            # WHAT: Log step 2
            # WHY: Track EDA execution
            # HOW: Step indicator
            # BENEFIT: Progress visibility
            # TRADE-OFF: Logging overhead

            logger.info("-" * 60)
            # WHAT: Step separator
            # WHY: Visual organization
            # HOW: Dashes
            # BENEFIT: Log readability
            # TRADE-OFF: Extra line

            eda_report = self.eda.execute(
                processed_data,
                generate_plots=True,
                output_dir=eda_output_dir,
            )
            # WHAT: Execute EDA with visualizations
            # WHY: Analyze data, generate insights
            # HOW: Call EDA use case with parameters
            # BENEFIT: Data understanding, insights
            # TRADE-OFF: Analysis time

            results["eda_report"] = eda_report
            # WHAT: Store EDA report
            # WHY: Make insights available
            # HOW: Dict assignment
            # BENEFIT: Access to analysis
            # TRADE-OFF: Memory for report

            logger.info(f"✓ Generated {len(eda_report.insights)} insights")
            # WHAT: Log insights count
            # WHY: Feedback on analysis depth
            # HOW: Count insights list
            # BENEFIT: Quality indicator
            # TRADE-OFF: Logging overhead

            logger.info(f"✓ Created {len(eda_report.visualizations)} visualizations")
            # WHAT: Log visualization count
            # WHY: Confirm plot generation
            # HOW: Count visualizations dict
            # BENEFIT: Output confirmation
            # TRADE-OFF: Additional log

        else:
            # WHAT: Else block for skipped EDA
            # WHY: Log when EDA is skipped
            # HOW: Else clause
            # BENEFIT: Clear execution path
            # TRADE-OFF: Extra code

            logger.info("\n[2/4] Exploratory Data Analysis (SKIPPED)")
            # WHAT: Log skipped EDA
            # WHY: Track pipeline path
            # HOW: SKIPPED indicator
            # BENEFIT: Clear execution trace
            # TRADE-OFF: Logging for non-action

            logger.info("-" * 60)
            # WHAT: Separator even when skipped
            # WHY: Consistent formatting
            # HOW: Dashes
            # BENEFIT: Log consistency
            # TRADE-OFF: Extra line

        # Step 3: Model Training
        # WHAT: Comment for step 3
        # WHY: Mark training phase
        # HOW: Inline comment
        # BENEFIT: Code organization
        # TRADE-OFF: Additional comment

        logger.info("\n[3/4] Model Training")
        # WHAT: Log step 3
        # WHY: Track training start
        # HOW: Step indicator
        # BENEFIT: Progress tracking
        # TRADE-OFF: Logging overhead

        logger.info("-" * 60)
        # WHAT: Step separator
        # WHY: Visual organization
        # HOW: Dashes
        # BENEFIT: Readability
        # TRADE-OFF: Extra line

        trained_model = self.model_training.execute(
            processed_data,
            model_config,
            save_model=True,
            model_path=model_output_path,
        )
        # WHAT: Train and save model
        # WHY: Create ML model
        # HOW: Call training use case
        # BENEFIT: Trained model ready for use
        # TRADE-OFF: Training time

        results["trained_model"] = trained_model
        # WHAT: Store trained model
        # WHY: Make available to caller
        # HOW: Dict assignment
        # BENEFIT: Access to model
        # TRADE-OFF: Memory reference

        logger.info(f"✓ Model trained: {model_config.model_type}")
        # WHAT: Log model type
        # WHY: Confirm training success
        # HOW: Checkmark with model type
        # BENEFIT: Success confirmation
        # TRADE-OFF: Logging overhead

        logger.info(f"✓ Metrics: {trained_model.metrics}")
        # WHAT: Log performance metrics
        # WHY: Immediate performance feedback
        # HOW: Display metrics dict
        # BENEFIT: Quality assessment
        # TRADE-OFF: Verbose logging

        # Step 4: Generate predictions on training data (as example)
        # WHAT: Comment for step 4
        # WHY: Explain prediction purpose
        # HOW: Inline comment
        # BENEFIT: Clarifies demo nature
        # TRADE-OFF: Additional comment

        logger.info("\n[4/4] Generating Predictions")
        # WHAT: Log step 4
        # WHY: Track prediction phase
        # HOW: Step indicator
        # BENEFIT: Progress tracking
        # TRADE-OFF: Logging overhead

        logger.info("-" * 60)
        # WHAT: Step separator
        # WHY: Visual organization
        # HOW: Dashes
        # BENEFIT: Readability
        # TRADE-OFF: Extra line

        prediction = self.prediction.execute(
            processed_data.data,
            model=trained_model,
        )
        # WHAT: Generate predictions
        # WHY: Demonstrate model usage
        # HOW: Call prediction use case with training data
        # BENEFIT: Example predictions
        # TRADE-OFF: Predictions on training data (not ideal)

        results["predictions"] = prediction
        # WHAT: Store predictions
        # WHY: Make available to caller
        # HOW: Dict assignment
        # BENEFIT: Access to predictions
        # TRADE-OFF: Memory for results

        logger.info(f"✓ Generated {len(prediction.predictions)} predictions")
        # WHAT: Log prediction count
        # WHY: Confirm prediction completion
        # HOW: Count predictions
        # BENEFIT: Output confirmation
        # TRADE-OFF: Logging overhead
        
        logger.info("\n" + "=" * 60)
        # WHAT: Pipeline completion separator
        # WHY: Visual completion indicator
        # HOW: Newline plus equals
        # BENEFIT: Clear ending
        # TRADE-OFF: Cosmetic logging

        logger.info("ML Pipeline Completed Successfully!")
        # WHAT: Log successful completion
        # WHY: Confirm entire pipeline success
        # HOW: Success message
        # BENEFIT: Clear outcome
        # TRADE-OFF: Logging overhead

        logger.info("=" * 60)
        # WHAT: Final separator
        # WHY: Visual framing
        # HOW: Equals signs
        # BENEFIT: Clean log section
        # TRADE-OFF: Extra line

        return results
        # WHAT: Return all pipeline results
        # WHY: Provide comprehensive outputs
        # HOW: Return dict with 4 results
        # BENEFIT: Access to all intermediate and final results
        # TRADE-OFF: Large dict, untyped structure
```

---

## Design Patterns

### 1. **Facade Pattern**
- **Purpose**: Simplify complex multi-step workflow
- **Benefits**: Single interface for complete pipeline
- **Trade-offs**: Hides flexibility of individual steps

### 2. **Composite Pattern**
- **Purpose**: Compose multiple use cases
- **Benefits**: Reusable components, modular design
- **Trade-offs**: Dependency chain

### 3. **Orchestrator Pattern**
- **Purpose**: Coordinate workflow across use cases
- **Benefits**: Centralized pipeline logic
- **Trade-offs**: Orchestrator complexity

### 4. **Dependency Injection**
- **Purpose**: Inject all 4 use case dependencies
- **Benefits**: Testable, flexible composition
- **Trade-offs**: Many dependencies

---

## Pros & Cons

### Pros ✅
1. **Complete Automation** - End-to-end workflow in one call
2. **Comprehensive Logging** - Detailed progress tracking
3. **Flexible** - Optional EDA, custom paths
4. **Reusable Components** - Composed of tested use cases
5. **All Results Returned** - Access to intermediate outputs
6. **Visual Progress** - Clear step indicators in logs

### Cons ❌
1. **Long Method** - Execute method has many responsibilities
2. **Hardcoded Steps** - Fixed 4-step workflow
3. **Training Data Predictions** - Step 4 uses training data (not ideal)
4. **Untyped Results** - Returns dict instead of typed entity
5. **No Error Recovery** - If step fails, entire pipeline fails
6. **No Parallel Execution** - Sequential only

---

## Usage Examples

### Example 1: Complete Pipeline
```python
# Full end-to-end workflow
pipeline = MLPipelineUseCase(ingestion, eda, training, prediction)
results = pipeline.execute(
    source=DataSource(path=Path("data.csv"), data_type="csv"),
    model_config=ModelConfig(model_type="random_forest")
)
print(f"Model accuracy: {results['trained_model'].metrics['accuracy']}")
```

### Example 2: Skip EDA
```python
# Fast pipeline without analysis
results = pipeline.execute(
    source=source,
    model_config=config,
    perform_eda=False  # Skip EDA for speed
)
```

### Example 3: Custom Paths
```python
# Custom output locations
results = pipeline.execute(
    source=source,
    model_config=config,
    eda_output_dir=Path("reports/eda"),
    model_output_path=Path("models/production/model.pkl")
)
```

---

## Related Files
- **Use Cases**: data_ingestion.py, eda.py, model_training.py, prediction.py
- **Domain**: entities.py (DataSource, ModelConfig, Prediction)
- **DI Container**: container.py (MLPipelineUseCase composition)
