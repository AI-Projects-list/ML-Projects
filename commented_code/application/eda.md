# EDA Use Case - Detailed Code Documentation

**File**: `src/application/use_cases/eda.py`  
**Purpose**: Orchestrate exploratory data analysis workflow  
**Layer**: Application  
**Pattern**: Simple Orchestration + Optional Feature Pattern

---

## Key Code Sections with Commentary

```python
class EDAUseCase:
    """Handles exploratory data analysis workflow."""
    # WHAT: EDA orchestration use case
    # WHY: Separate analysis orchestration from implementation
    # RESPONSIBILITY: Coordinate analyzer, handle plot generation
    
    def __init__(self, analyzer: IEDAAnalyzer):
        # WHAT: DI constructor
        # WHY: Inject analyzer implementation
        # BENEFIT: Testable with mocks
        # GOOD: Depends on interface (IEDAAnalyzer)
        self.analyzer = analyzer
    
    def execute(
        self,
        data: ProcessedData,
        generate_plots: bool = True,
        output_dir: Path | None = None,
    ) -> EDAReport:
        # WHAT: Execute EDA workflow
        # WHY: Analyze + optionally generate plots
        # PARAMETERS:
        #   - data: ProcessedData (rich entity with metadata)
        #   - generate_plots: Boolean flag (optional feature)
        #   - output_dir: Where to save plots (None = default)
        # RETURN: EDAReport entity
        
        logger.info("Starting exploratory data analysis...")
        
        # Perform analysis
        report = self.analyzer.analyze(data)
        # WHAT: Delegate analysis to analyzer
        # WHY: Separation of concerns
        # RETURN: EDAReport with statistics, correlations, insights
        
        # Generate visualizations if requested
        if generate_plots:
            # WHAT: Optional plot generation
            # WHY: Sometimes we only need stats, not plots
            # USE CASE: Quick analysis vs full report
            
            if output_dir is None:
                output_dir = Path("outputs/eda")
            # WHAT: Default output directory
            # WHY: Sensible default if not provided
            # TRADE-OFF: Hardcoded path
            
            logger.info(f"Generating visualizations in {output_dir}")
            plot_paths = self.analyzer.generate_visualizations(data, output_dir)
            # WHAT: Generate and save plots
            # WHY: Visual insights
            # RETURN: List of file paths
            # SIDE EFFECT: Creates files on disk
            
            report.visualizations = {
                f"plot_{i}": path for i, path in enumerate(plot_paths)
            }
            # WHAT: Store plot paths in report
            # WHY: Link report to visualizations
            # FORMAT: Dictionary with keys like "plot_0", "plot_1"
            # TRADE-OFF: Generic keys, not descriptive
        
        logger.info("EDA completed successfully")
        logger.info(f"Insights generated: {len(report.insights)}")
        
        return report
```

---

## Pros & Cons

### ✅ Pros
- **Simple**: Minimal orchestration logic
- **Flexible**: Optional plot generation
- **Clean**: Delegates to analyzer
- **Default Handling**: Sensible default for output_dir

### ❌ Cons
- **Hardcoded Path**: "outputs/eda" default
- **Generic Plot Names**: "plot_0" not descriptive
- **No Plot Selection**: Can't choose which plots to generate
- **Side Effects**: File I/O not explicit in signature

---

**Total Lines**: 50  
**Complexity**: Very Low  
**Dependencies**: 1 (analyzer)
