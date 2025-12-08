"""Gradient Boosting Model - Complete Pipeline."""

from pathlib import Path
from loguru import logger

from src.domain.entities import DataSource, DataSourceType, ModelConfig
from src.infrastructure.config.container import Container
from src.infrastructure.config.logging import setup_logging
from src.infrastructure.config.settings import get_settings


def main() -> None:
    """Run Gradient Boosting pipeline."""
    # Initialize
    settings = get_settings()
    setup_logging(settings)
    container = Container(settings)
    
    logger.info("=" * 60)
    logger.info("GRADIENT BOOSTING MODEL PIPELINE")
    logger.info("=" * 60)
    
    # Data source - use classification dataset
    source = DataSource(
        source_type=DataSourceType.CSV,
        path="data/raw/sample_classification.csv",
        metadata={"encoding": "utf-8"},
    )
    
    # Model configuration
    model_config = ModelConfig(
        model_type="gradient_boosting",
        target_column="target",  # Classification target
        hyperparameters={
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5,
            "min_samples_split": 2,
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
        eda_output_dir=Path("outputs/gradient_boosting/eda"),
        model_output_path=Path("models/gradient_boosting_model.pkl"),
    )
    
    # Display results
    processed_data = results["processed_data"]
    trained_model = results["trained_model"]
    predictions = results["predictions"]
    
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Model Type: Gradient Boosting")
    logger.info(f"Data Shape: {processed_data.data.shape}")
    logger.info(f"Model Metrics: {trained_model.metrics}")
    logger.info(f"Total Predictions: {len(predictions.predictions)}")
    logger.info(f"Feature Importance: {trained_model.feature_importance}")
    logger.info(f"Model saved to: models/gradient_boosting_model.pkl")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
