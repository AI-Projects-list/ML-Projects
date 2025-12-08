"""
Complete ML Pipeline - Linear Regression
=========================================
This script runs the entire ML pipeline for Linear Regression from start to finish.
"""

from pathlib import Path
from loguru import logger
import pandas as pd

from src.domain.entities import DataSource, DataSourceType, ModelConfig
from src.infrastructure.config.container import Container
from src.infrastructure.config.logging import setup_logging
from src.infrastructure.config.settings import get_settings


def run_linear_regression_pipeline():
    """Execute complete Linear Regression pipeline."""
    
    # ========== INITIALIZATION ==========
    print("\n" + "="*70)
    print("LINEAR REGRESSION - COMPLETE ML PIPELINE")
    print("="*70 + "\n")
    
    settings = get_settings()
    setup_logging(settings)
    container = Container(settings)
    
    # ========== STEP 1: DATA PREPARATION ==========
    logger.info("STEP 1: Data Preparation")
    logger.info("-" * 70)
    
    # Generate sample data if needed
    from scripts.generate_sample_data import generate_regression_dataset
    data_path = Path("data/raw/sample_regression.csv")
    if not data_path.exists():
        logger.info("Generating sample regression dataset...")
        df = generate_regression_dataset()
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)
        logger.info(f"Sample data created: {data_path}")
    
    # Create data source
    source = DataSource(
        source_type=DataSourceType.CSV,
        path=str(data_path),
        metadata={"encoding": "utf-8"},
    )
    
    # ========== STEP 2: DATA INGESTION & PREPROCESSING ==========
    logger.info("\nSTEP 2: Data Ingestion & Preprocessing")
    logger.info("-" * 70)
    
    ingestion_use_case = container.data_ingestion_use_case
    processed_data = ingestion_use_case.execute(
        source=source,
        clean=True,
        transform=True
    )
    
    logger.info(f"✓ Data processed: {processed_data.data.shape}")
    logger.info(f"✓ Processing steps: {', '.join(processed_data.processing_steps)}")
    
    # Save processed data
    processed_path = Path("data/processed/linear_regression_data.pkl")
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    container.data_repository.save(processed_data, processed_path)
    logger.info(f"✓ Processed data saved: {processed_path}")
    
    # ========== STEP 3: EXPLORATORY DATA ANALYSIS ==========
    logger.info("\nSTEP 3: Exploratory Data Analysis")
    logger.info("-" * 70)
    
    eda_use_case = container.eda_use_case
    eda_output_dir = Path("outputs/linear_regression/eda")
    eda_output_dir.mkdir(parents=True, exist_ok=True)
    
    eda_report = eda_use_case.execute(
        data=processed_data,
        output_dir=eda_output_dir
    )
    
    logger.info(f"✓ EDA completed: {len(eda_report.insights)} insights generated")
    logger.info("Top Insights:")
    for i, insight in enumerate(eda_report.insights[:3], 1):
        logger.info(f"  {i}. {insight}")
    
    # ========== STEP 4: MODEL TRAINING ==========
    logger.info("\nSTEP 4: Model Training - Linear Regression")
    logger.info("-" * 70)
    
    model_config = ModelConfig(
        model_type="linear_regression",
        target_column="price",
        hyperparameters={},
        test_size=0.2,
        random_state=42,
    )
    
    training_use_case = container.model_training_use_case
    model_path = Path("models/linear_regression_model.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    trained_model = training_use_case.execute(
        data=processed_data,
        config=model_config,
        model_path=model_path
    )
    
    logger.info(f"✓ Model trained: Linear Regression")
    logger.info(f"✓ Model saved: {model_path}")
    logger.info("Model Performance Metrics:")
    for metric, value in trained_model.metrics.items():
        logger.info(f"  • {metric}: {value:.4f}")
    
    # ========== STEP 5: PREDICTIONS ==========
    logger.info("\nSTEP 5: Making Predictions")
    logger.info("-" * 70)
    
    prediction_use_case = container.prediction_use_case
    test_data = processed_data.data.drop(columns=[model_config.target_column])
    
    predictions = prediction_use_case.execute(
        data=test_data,
        model_path=model_path
    )
    
    logger.info(f"✓ Generated {len(predictions.predictions)} predictions")
    
    # Save predictions
    predictions_path = Path("outputs/linear_regression/predictions.csv")
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    
    results_df = test_data.copy()
    results_df['prediction'] = predictions.predictions.values
    results_df['actual'] = processed_data.data[model_config.target_column].values
    results_df['error'] = results_df['actual'] - results_df['prediction']
    results_df.to_csv(predictions_path, index=False)
    
    logger.info(f"✓ Predictions saved: {predictions_path}")
    logger.info(f"✓ Average error: {results_df['error'].abs().mean():.2f}")
    
    # ========== FINAL SUMMARY ==========
    print("\n" + "="*70)
    print("LINEAR REGRESSION PIPELINE - COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\nData Shape: {processed_data.data.shape}")
    print(f"Target Column: {model_config.target_column}")
    print(f"Model Metrics: {trained_model.metrics}")
    print(f"Model Saved: {model_path}")
    print(f"EDA Outputs: {eda_output_dir}")
    print(f"Predictions: {predictions_path}")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        run_linear_regression_pipeline()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
