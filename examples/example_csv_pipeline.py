"""Example: Complete ML pipeline with CSV data."""

from pathlib import Path

from loguru import logger

from src.domain.entities import DataSource, DataSourceType, ModelConfig
from src.infrastructure.config.container import Container
from src.infrastructure.config.logging import setup_logging
from src.infrastructure.config.settings import get_settings


def main() -> None:
    """Run example ML pipeline."""
    # Initialize
    settings = get_settings()
    setup_logging(settings)
    container = Container(settings)
    
    logger.info("Starting example ML pipeline")
    
    # Create sample data source
    # NOTE: Replace with your actual data file
    data_path = "data/raw/sample_data.csv"
    
    source = DataSource(
        source_type=DataSourceType.CSV,
        path=data_path,
        metadata={
            "encoding": "utf-8",
        },
    )
    
    # Configure model
    # NOTE: Set your target column name
    model_config = ModelConfig(
        model_type="random_forest",
        target_column="target",  # Replace with your target column
        hyperparameters={
            "n_estimators": 100,
            "max_depth": 10,
        },
        test_size=0.2,
        random_state=42,
    )
    
    # Execute pipeline
    pipeline = container.ml_pipeline_use_case
    
    results = pipeline.execute(
        source=source,
        model_config=model_config,
        perform_eda=True,
        eda_output_dir=Path("outputs/eda"),
        model_output_path=Path("models/example_model.pkl"),
    )
    
    # Access results
    processed_data = results["processed_data"]
    eda_report = results["eda_report"]
    trained_model = results["trained_model"]
    predictions = results["predictions"]
    
    logger.info(f"Pipeline completed successfully!")
    logger.info(f"Data shape: {processed_data.data.shape}")
    logger.info(f"Model metrics: {trained_model.metrics}")
    logger.info(f"Predictions: {len(predictions.predictions)}")


if __name__ == "__main__":
    main()
