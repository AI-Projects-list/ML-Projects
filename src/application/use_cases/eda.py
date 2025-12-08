"""Use case for exploratory data analysis."""

from pathlib import Path

from loguru import logger

from src.domain.entities import EDAReport, ProcessedData
from src.domain.repositories import IEDAAnalyzer


class EDAUseCase:
    """Handles exploratory data analysis workflow."""
    
    def __init__(self, analyzer: IEDAAnalyzer):
        """
        Initialize EDA use case.
        
        Args:
            analyzer: EDA analyzer implementation
        """
        self.analyzer = analyzer
    
    def execute(
        self,
        data: ProcessedData,
        generate_plots: bool = True,
        output_dir: Path | None = None,
    ) -> EDAReport:
        """
        Execute exploratory data analysis.
        
        Args:
            data: Processed data to analyze
            generate_plots: Whether to generate visualization plots
            output_dir: Directory for saving plots
            
        Returns:
            EDA report with insights and statistics
        """
        logger.info("Starting exploratory data analysis...")
        
        # Perform analysis
        report = self.analyzer.analyze(data)
        
        # Generate visualizations if requested
        if generate_plots:
            if output_dir is None:
                output_dir = Path("outputs/eda")
            
            logger.info(f"Generating visualizations in {output_dir}")
            plot_paths = self.analyzer.generate_visualizations(data, output_dir)
            report.visualizations = {
                f"plot_{i}": path for i, path in enumerate(plot_paths)
            }
        
        logger.info("EDA completed successfully")
        logger.info(f"Insights generated: {len(report.insights)}")
        
        return report
