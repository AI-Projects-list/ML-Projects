"""Example: EDA only workflow."""

from pathlib import Path

from loguru import logger

from src.domain.entities import DataSource, DataSourceType
from src.infrastructure.config.container import Container
from src.infrastructure.config.logging import setup_logging
from src.infrastructure.config.settings import get_settings


def main() -> None:
    """Run example EDA workflow."""
    # Initialize
    settings = get_settings()
    setup_logging(settings)
    container = Container(settings)
    
    logger.info("Starting EDA example")
    
    # Ingest data
    source = DataSource(
        source_type=DataSourceType.CSV,
        path="data/raw/sample_data.csv",
    )
    
    ingestion = container.data_ingestion_use_case
    processed_data = ingestion.execute(source)
    
    # Perform EDA
    eda = container.eda_use_case
    report = eda.execute(
        data=processed_data,
        generate_plots=True,
        output_dir=Path("outputs/eda_example"),
    )
    
    # Display insights
    logger.info("\n" + "=" * 60)
    logger.info("EDA INSIGHTS")
    logger.info("=" * 60)
    
    for i, insight in enumerate(report.insights, 1):
        logger.info(f"{i}. {insight}")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"Data shape: {report.data_shape}")
    logger.info(f"Column types: {len(report.column_types)}")
    logger.info(f"Visualizations generated: {len(report.visualizations)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
