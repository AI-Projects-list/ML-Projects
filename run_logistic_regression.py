"""Logistic Regression Model - Complete Pipeline."""

from pathlib import Path
from loguru import logger

from src.domain.entities import DataSource, DataSourceType, ModelConfig
from src.infrastructure.config.container import Container
from src.infrastructure.config.logging import setup_logging
from src.infrastructure.config.settings import get_settings


def main() -> None:
    """Run Logistic Regression pipeline."""
    # Initialize
    settings = get_settings()
    setup_logging(settings)
    container = Container(settings)
    
    logger.info("=" * 60)
    logger.info("LOGISTIC REGRESSION MODEL PIPELINE")
    logger.info("=" * 60)
    
    # Data source - use classification dataset
    source = DataSource(
        source_type=DataSourceType.CSV,
        path="data/raw/sample_classification.csv",
        metadata={"encoding": "utf-8"},
    )
    
    # Model configuration
    model_config = ModelConfig(
        model_type="logistic_regression",
        target_column="target",  # Classification target
        hyperparameters={
            "max_iter": 1000,
            "solver": "lbfgs",
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
        eda_output_dir=Path("outputs/logistic_regression/eda"),
        model_output_path=Path("models/logistic_regression_model.pkl"),
    )
    
    # Display results
    processed_data = results["processed_data"]
    trained_model = results["trained_model"]
    predictions = results["predictions"]
    
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Model Type: Logistic Regression")
    logger.info(f"Data Shape: {processed_data.data.shape}")
    logger.info(f"Model Metrics: {trained_model.metrics}")
    logger.info(f"Total Predictions: {len(predictions.predictions)}")
    logger.info(f"Model saved to: models/logistic_regression_model.pkl")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
