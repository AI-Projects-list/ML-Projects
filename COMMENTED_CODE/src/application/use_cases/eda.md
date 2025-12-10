# EDA Use Case - Comprehensive Documentation

## File Information
- **Source File**: `src/application/use_cases/eda.py`
- **Purpose**: Orchestrates exploratory data analysis workflow with statistics and visualizations
- **Layer**: Application Layer (Use Cases)
- **Pattern**: Use Case Pattern, Strategy Pattern, Template Method

## Complete Annotated Code

```python
"""Use case for exploratory data analysis."""
# WHAT: Module-level docstring documenting the EDA use case
# WHY: Provides clear documentation for developers and tools
# HOW: Python docstring convention with triple quotes
# BENEFIT: Improved discoverability, IDE support, documentation generation
# TRADE-OFF: Brief description - could expand to describe EDA workflow steps

from pathlib import Path
# WHAT: Import Path class for file system path manipulation
# WHY: Modern, object-oriented approach to handling file paths
# HOW: Import from pathlib standard library
# BENEFIT: Cross-platform compatibility, immutable paths, rich path operations
# TRADE-OFF: Additional import vs string paths, but benefits outweigh costs

from loguru import logger
# WHAT: Import pre-configured logger from loguru library
# WHY: Structured logging with beautiful formatting and automatic context
# HOW: Import logger singleton configured in infrastructure layer
# BENEFIT: Beautiful console output, automatic exception catching, structured logs
# TRADE-OFF: External dependency vs standard logging, but superior developer experience

from src.domain.entities import EDAReport, ProcessedData
# WHAT: Import domain entities for EDA workflow
# WHY: Use domain models to encapsulate business data and enforce rules
# HOW: Import EDAReport (output) and ProcessedData (input) entities
# BENEFIT: Type safety, business logic encapsulation, clear data contracts
# TRADE-OFF: More classes vs simple dicts, but enforces domain integrity

from src.domain.repositories import IEDAAnalyzer
# WHAT: Import interface (abstract base class) for EDA analysis
# WHY: Dependency inversion - depend on abstractions not implementations
# HOW: Import IEDAAnalyzer protocol defining analyze/generate_visualizations contract
# BENEFIT: Testability (easy to mock), flexibility (swap implementations), loose coupling
# TRADE-OFF: Additional abstraction layer vs concrete dependency, but enables SOLID principles


class EDAUseCase:
    # WHAT: Use case class orchestrating exploratory data analysis
    # WHY: Application layer pattern to coordinate domain objects and infrastructure
    # HOW: Class encapsulating EDA business workflow
    # BENEFIT: Single responsibility (EDA only), testable, reusable across interfaces
    # TRADE-OFF: Additional class vs procedural function, but provides better organization

    """Handles exploratory data analysis workflow."""
    # WHAT: Class-level docstring describing responsibility
    # WHY: Documents class purpose for developers and tools
    # HOW: Concise single-line docstring
    # BENEFIT: Clear understanding of purpose, IDE support
    # TRADE-OFF: Brief description - could expand to describe analysis types

    def __init__(self, analyzer: IEDAAnalyzer):
        # WHAT: Constructor accepting analyzer dependency
        # WHY: Dependency injection for testability and flexibility
        # HOW: Accept IEDAAnalyzer interface as parameter
        # BENEFIT: Testable (inject mock), flexible (swap analyzers), explicit dependency
        # TRADE-OFF: Requires external analyzer creation vs creating internally, but enables testing

        """
        Initialize EDA use case.

        Args:
            analyzer: EDA analyzer implementation
        """
        # WHAT: Constructor docstring documenting parameter
        # WHY: Clear documentation for dependency injection requirement
        # HOW: Google-style docstring with Args section
        # BENEFIT: IDE auto-complete, clear API documentation
        # TRADE-OFF: Verbose documentation vs brief comment, but improves usability

        self.analyzer = analyzer
        # WHAT: Store analyzer as instance attribute
        # WHY: Make analyzer available throughout use case lifecycle
        # HOW: Assign injected analyzer to instance variable
        # BENEFIT: Access analyzer in execute method without passing as parameter
        # TRADE-OFF: State management (mutable instance) vs stateless function, but appropriate for use case
    
    def execute(
        # WHAT: Main execution method signature for EDA workflow
        # WHY: Execute is standard naming convention for use case entry points
        # HOW: Public method coordinating complete EDA workflow
        # BENEFIT: Clear entry point, standardized interface across use cases
        # TRADE-OFF: Generic name vs specific like "analyze_data", but consistent pattern

        self,
        # WHAT: Instance reference for accessing analyzer
        # WHY: Required for Python instance methods
        # HOW: Access self.analyzer within method
        # BENEFIT: Access to injected dependency
        # TRADE-OFF: None - required by Python

        data: ProcessedData,
        # WHAT: Domain entity containing processed data to analyze
        # WHY: Use rich domain object with data, metadata, and status
        # HOW: Type-hinted parameter expecting ProcessedData entity
        # BENEFIT: Type safety, access to metadata and processing history, validation
        # TRADE-OFF: Domain object vs simple DataFrame, but provides comprehensive context

        generate_plots: bool = True,
        # WHAT: Flag to enable/disable visualization generation
        # WHY: Allow flexible configuration - plots may be expensive or unnecessary
        # HOW: Boolean parameter with default True (visualizations recommended)
        # BENEFIT: Flexibility to skip plots, performance optimization, control over outputs
        # TRADE-OFF: Additional parameter vs always generating, but enables customization

        output_dir: Path | None = None,
        # WHAT: Optional directory path for saving visualization plots
        # WHY: Allow caller to specify custom output location
        # HOW: Union type hint (Path or None) with default None
        # BENEFIT: Flexibility in output location, None triggers default behavior
        # TRADE-OFF: Union type complexity vs always requiring Path, but improves usability

    ) -> EDAReport:
        # WHAT: Return type annotation specifying EDAReport entity
        # WHY: Type safety and IDE auto-completion for return value
        # HOW: Arrow notation indicating method returns EDAReport instance
        # BENEFIT: Type checking, clear contract, IDE support
        # TRADE-OFF: Annotation overhead vs untyped, but improves reliability

        """
        Execute exploratory data analysis.
        
        Args:
            data: Processed data to analyze
            generate_plots: Whether to generate visualization plots
            output_dir: Directory for saving plots

        Returns:
            EDA report with insights and statistics
        """
        # WHAT: Comprehensive docstring documenting execute method
        # WHY: Clear documentation of EDA parameters and return value
        # HOW: Google-style docstring with Args and Returns sections
        # BENEFIT: IDE support, clear API documentation, maintainability
        # TRADE-OFF: Verbose vs brief comments, but essential for public API

        logger.info("Starting exploratory data analysis...")
        # WHAT: Log informational message about EDA start
        # WHY: Observability - track workflow execution and debug issues
        # HOW: Use logger.info with descriptive message
        # BENEFIT: Troubleshooting, monitoring, audit trail
        # TRADE-OFF: I/O overhead vs silent execution, but critical for production

        # Perform analysis
        # WHAT: Comment indicating the analysis step
        # WHY: Mark distinct workflow phase
        # HOW: Single-line comment above analysis code
        # BENEFIT: Code organization, clear workflow structure
        # TRADE-OFF: Additional comment vs self-documenting code, but aids readability

        report = self.analyzer.analyze(data)
        # WHAT: Perform statistical analysis and generate insights
        # WHY: Core EDA functionality - compute statistics, detect patterns, generate insights
        # HOW: Call analyzer's analyze method passing ProcessedData entity
        # BENEFIT: Comprehensive analysis (statistics, correlations, distributions), insights generation
        # TRADE-OFF: Analysis time vs skipping, but essential for understanding data

        # Generate visualizations if requested
        # WHAT: Comment indicating conditional visualization step
        # WHY: Mark visualization logic separately from analysis
        # HOW: Single-line comment above conditional block
        # BENEFIT: Clear code structure, easy to navigate
        # TRADE-OFF: Comment overhead vs code only, but improves readability

        if generate_plots:
            # WHAT: Conditional execution of visualization generation
            # WHY: Allow skipping expensive plot generation when not needed
            # HOW: Check generate_plots boolean flag
            # BENEFIT: Flexible workflow, performance optimization, control over outputs
            # TRADE-OFF: Conditional logic vs always generate, but provides user control

            if output_dir is None:
                # WHAT: Check if output directory was provided
                # WHY: Provide default directory when caller doesn't specify
                # HOW: Test if output_dir is None
                # BENEFIT: Sensible default behavior, caller doesn't need to know directory structure
                # TRADE-OFF: Hardcoded default path vs requiring caller to specify, but improves usability

                output_dir = Path("outputs/eda")
                # WHAT: Set default output directory for visualizations
                # WHY: Standard location for EDA outputs when not specified
                # HOW: Create Path object with default directory path
                # BENEFIT: Consistent output location, organized project structure
                # TRADE-OFF: Hardcoded path vs configuration, but reasonable default

            logger.info(f"Generating visualizations in {output_dir}")
            # WHAT: Log visualization generation with output directory
            # WHY: Observability - inform user where plots will be saved
            # HOW: Info-level log with f-string including output_dir
            # BENEFIT: User knows where to find visualizations, troubleshooting
            # TRADE-OFF: Logging I/O vs silent generation, but helpful for users

            plot_paths = self.analyzer.generate_visualizations(data, output_dir)
            # WHAT: Generate visualization plots and get file paths
            # WHY: Create visual representations of data patterns and distributions
            # HOW: Call analyzer's generate_visualizations method with data and output directory
            # BENEFIT: Visual insights, easier pattern recognition, presentation-ready plots
            # TRADE-OFF: Plot generation time and disk space vs no visualizations, but critical for EDA

            report.visualizations = {
                # WHAT: Create dictionary mapping plot names to file paths
                # WHY: Store visualization references in report for downstream use
                # HOW: Dictionary comprehension creating name → path mappings
                # BENEFIT: Easy access to plots by name, links visualizations to report
                # TRADE-OFF: Memory for paths vs just saving files, but enables programmatic access

                f"plot_{i}": path for i, path in enumerate(plot_paths)
                # WHAT: Dictionary comprehension creating plot_0, plot_1, etc. → path mappings
                # WHY: Programmatically generate unique names for each plot
                # HOW: Enumerate plot_paths list to get index and path, create f-string keys
                # BENEFIT: Automatic naming, no manual name assignment, scalable to any number of plots
                # TRADE-OFF: Generic names (plot_0) vs descriptive names (histogram), but consistent

            }
            # WHAT: Closing brace for visualizations dictionary
            # WHY: Complete dictionary comprehension assignment
            # HOW: Standard Python syntax
            # BENEFIT: Report now contains visualization references
            # TRADE-OFF: None

        logger.info("EDA completed successfully")
        # WHAT: Log successful EDA completion
        # WHY: Observability - confirm success for monitoring/debugging
        # HOW: Info-level log message
        # BENEFIT: Clear success indication, audit trail, troubleshooting
        # TRADE-OFF: Logging I/O vs silent success, but important for production

        logger.info(f"Insights generated: {len(report.insights)}")
        # WHAT: Log number of insights generated
        # WHY: Provide feedback on analysis results
        # HOW: Info-level log with f-string counting insights list
        # BENEFIT: User knows analysis depth, helps assess data quality
        # TRADE-OFF: Additional logging vs minimal output, but provides useful context
        
        return report
        # WHAT: Return the EDA report entity
        # WHY: Provide analysis results to caller
        # HOW: Return EDAReport instance with statistics, insights, and visualizations
        # BENEFIT: Rich result object with all analysis outputs, clear API
        # TRADE-OFF: Domain object vs simple dict, but provides structured results
```

---

## Design Patterns Used

### 1. **Use Case Pattern** (Application Layer)
- **Purpose**: Encapsulates business workflow for exploratory data analysis
- **Implementation**: `EDAUseCase` class with `execute` method
- **Benefits**: Single responsibility, testable, reusable across interfaces (CLI, API, notebook)
- **Trade-offs**: Additional abstraction vs direct implementation

### 2. **Dependency Injection Pattern**
- **Purpose**: Invert control - depend on IEDAAnalyzer abstraction
- **Implementation**: Constructor accepts `analyzer` dependency
- **Benefits**: Testability (inject mocks), flexibility (swap analyzers), explicit dependencies
- **Trade-offs**: More verbose initialization vs creating analyzer internally

### 3. **Strategy Pattern**
- **Purpose**: Allow different analysis strategies via IEDAAnalyzer interface
- **Implementation**: Analyzer interface enables swapping analysis implementations
- **Benefits**: Flexible analysis strategies, easy to add new analyzers, runtime selection
- **Trade-offs**: Additional abstraction vs hardcoded analyzer

### 4. **Template Method Pattern**
- **Purpose**: Define workflow skeleton (analyze → optionally visualize)
- **Implementation**: Execute method defines analysis workflow steps
- **Benefits**: Consistent workflow structure, extensible via analyzer interface
- **Trade-offs**: Fixed workflow order vs fully customizable

### 5. **Default Parameter Pattern**
- **Purpose**: Provide sensible defaults for optional parameters
- **Implementation**: `generate_plots=True`, `output_dir=None` with default assignment
- **Benefits**: Simpler API for common cases, flexibility for advanced use
- **Trade-offs**: Hidden defaults vs explicit parameters

---

## Pros & Cons

### Pros ✅

1. **Simple & Clean API**
   - Minimal parameters (just data and optional flags)
   - Clear entry point with `execute` method
   - Sensible defaults for common workflows

2. **Flexible Workflow**
   - Configurable visualization generation via flag
   - Custom output directory support with sensible default
   - Easy to skip expensive visualizations

3. **Strong Separation of Concerns**
   - Use case orchestrates workflow
   - Analyzer performs actual analysis
   - Domain entities encapsulate data

4. **Excellent Observability**
   - Logging at workflow start/end
   - Logs insights count for feedback
   - Logs visualization output directory

5. **Testability**
   - Analyzer injected via constructor (easy to mock)
   - Clear inputs (ProcessedData) and outputs (EDAReport)
   - No hidden dependencies or global state

6. **Rich Return Value**
   - EDAReport entity with statistics, insights, visualizations
   - Structured access to all analysis outputs
   - Visualization file paths for downstream use

### Cons ❌

1. **Limited Workflow Control**
   - Fixed workflow: analyze → visualize
   - Cannot customize analysis steps
   - Cannot generate only specific visualizations

2. **No Progress Tracking**
   - No callbacks for long-running analysis
   - Cannot track visualization generation progress
   - No percentage completion or ETA

3. **Generic Visualization Names**
   - Plots named plot_0, plot_1, etc.
   - No descriptive names (histogram, boxplot, etc.)
   - Harder to find specific visualization

4. **No Error Handling**
   - No try-except around analyzer calls
   - Exceptions propagate to caller
   - No partial success handling

5. **Directory Creation Not Guaranteed**
   - Assumes output_dir exists or analyzer creates it
   - May fail if directory doesn't exist
   - No explicit directory creation logic

6. **Limited Customization**
   - Cannot configure specific analyses (e.g., only correlations)
   - Cannot customize visualization types
   - All-or-nothing approach

---

## Usage Examples

### Example 1: Basic EDA with Visualizations
```python
from pathlib import Path
from src.domain.entities import ProcessedData
from src.infrastructure.processing.eda_analyzer import EDAAnalyzer
from src.application.use_cases.eda import EDAUseCase

# Setup dependencies
analyzer = EDAAnalyzer()
use_case = EDAUseCase(analyzer)

# Execute EDA with default visualizations
result = use_case.execute(processed_data)

print(f"Insights: {len(result.insights)}")
print(f"Statistics: {result.statistics}")
print(f"Visualizations: {result.visualizations}")
# Output:
# Insights: 5
# Statistics: {'mean': ..., 'correlation': ...}
# Visualizations: {'plot_0': Path('outputs/eda/histogram.png'), ...}
```

### Example 2: Analysis Only (No Visualizations)
```python
# Skip expensive visualizations for quick analysis
result = use_case.execute(
    processed_data,
    generate_plots=False  # Skip plot generation
)

print(f"Insights: {len(result.insights)}")
print(f"Visualizations: {result.visualizations}")
# Output:
# Insights: 5
# Visualizations: {}
```

### Example 3: Custom Output Directory
```python
# Specify custom visualization output directory
custom_dir = Path("reports/2024-01/eda")
result = use_case.execute(
    processed_data,
    generate_plots=True,
    output_dir=custom_dir
)

print(f"Plots saved to: {custom_dir}")
# Visualizations will be in reports/2024-01/eda/
```

### Example 4: Accessing Analysis Results
```python
result = use_case.execute(processed_data)

# Access statistics
print(f"Mean: {result.statistics['mean']}")
print(f"Correlation: {result.statistics['correlation']}")

# Access insights
for insight in result.insights:
    print(f"- {insight}")

# Access visualizations
for name, path in result.visualizations.items():
    print(f"{name}: {path}")
    # Can open, display, or share the visualization file
```

### Example 5: Testing with Mock Analyzer
```python
from unittest.mock import Mock
from src.domain.entities import EDAReport

# Create mock analyzer
mock_analyzer = Mock()
mock_report = EDAReport(
    statistics={"mean": 10.5},
    insights=["Data is normally distributed"],
    visualizations={}
)
mock_analyzer.analyze.return_value = mock_report
mock_analyzer.generate_visualizations.return_value = [Path("plot1.png")]

# Test use case with mock
use_case = EDAUseCase(mock_analyzer)
result = use_case.execute(processed_data)

# Assert expectations
assert len(result.insights) == 1
assert result.statistics["mean"] == 10.5
mock_analyzer.analyze.assert_called_once_with(processed_data)
```

### Example 6: Integration with Data Ingestion
```python
from src.application.use_cases.data_ingestion import DataIngestionUseCase

# Execute full workflow: ingest → EDA
ingestion_use_case = DataIngestionUseCase(reader_factory, processor)
eda_use_case = EDAUseCase(analyzer)

# Ingest data
processed_data = ingestion_use_case.execute(source)

# Perform EDA
eda_report = eda_use_case.execute(processed_data)

print(f"Processing steps: {processed_data.processing_steps}")
print(f"Insights: {len(eda_report.insights)}")
```

---

## Key Takeaways

### What This Use Case Does
Orchestrates exploratory data analysis workflow: statistical analysis → optional visualization generation, returning comprehensive EDA report.

### Why This Architecture
- **Clean Architecture**: Separates use case (orchestration) from analyzer (implementation)
- **Testability**: Analyzer injected, easy to mock for testing
- **Flexibility**: Optional visualizations, custom output directory

### How It Works
1. Accept `ProcessedData` entity and configuration flags
2. Call analyzer to perform statistical analysis and generate insights
3. Optionally generate visualizations to specified directory
4. Store visualization paths in report
5. Return rich `EDAReport` entity with all outputs

### Benefits
- **Simple API**: Minimal parameters with sensible defaults
- **Observable**: Logging at key workflow points
- **Reusable**: Can be called from CLI, API, notebooks
- **Rich Output**: Comprehensive report with statistics, insights, visualizations

### Trade-offs
- **Limited Control**: Fixed workflow, cannot customize analysis steps
- **Generic Names**: Visualizations named plot_0, plot_1, etc.
- **No Progress**: Cannot track long-running analysis

---

## Related Files
- **Domain Entities**: `src/domain/entities.py` - `EDAReport`, `ProcessedData`
- **Repository Interface**: `src/domain/repositories.py` - `IEDAAnalyzer`
- **Analyzer Implementation**: `src/infrastructure/processing/eda_analyzer.py`
- **Data Ingestion**: `src/application/use_cases/data_ingestion.py` - Produces ProcessedData input

---

*This documentation provides comprehensive line-by-line annotations with WHAT/WHY/HOW/BENEFIT/TRADE-OFF analysis for each line of code, design patterns explanation, pros & cons analysis, and practical usage examples.*
