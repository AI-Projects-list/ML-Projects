"""
Complete ML Pipeline - Gradient Boosting
=========================================
This script runs the entire ML pipeline for Gradient Boosting from start to finish.
"""

from pathlib import Path
from loguru import logger
import pandas as pd

from src.domain.entities import DataSource, DataSourceType, ModelConfig
from src.infrastructure.config.container import Container
from src.infrastructure.config.logging import setup_logging
from src.infrastructure.config.settings import get_settings


def run_gradient_boosting_pipeline():
    """Execute complete Gradient Boosting pipeline."""
    
    # ========== INITIALIZATION ==========
    print("\n" + "="*70)
    print("GRADIENT BOOSTING - COMPLETE ML PIPELINE")
    print("="*70 + "\n")
    
    settings = get_settings()
    setup_logging(settings)
    container = Container(settings)
    
    # ========== STEP 1: DATA PREPARATION ==========
    logger.info("STEP 1: Data Preparation")
    logger.info("-" * 70)
    
    # Generate sample data if needed
    from scripts.generate_sample_data import generate_classification_dataset
    data_path = Path("data/raw/sample_classification.csv")
    if not data_path.exists():
        logger.info("Generating sample classification dataset...")
        df = generate_classification_dataset()
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
    processed_path = Path("data/processed/gradient_boosting_data.pkl")
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    container.data_repository.save(processed_data, processed_path)
    logger.info(f"✓ Processed data saved: {processed_path}")
    
    # ========== STEP 3: EXPLORATORY DATA ANALYSIS ==========
    logger.info("\nSTEP 3: Exploratory Data Analysis")
    logger.info("-" * 70)
    
    eda_use_case = container.eda_use_case
    eda_output_dir = Path("outputs/gradient_boosting/eda")
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
    logger.info("\nSTEP 4: Model Training - Gradient Boosting")
    logger.info("-" * 70)
    
    model_config = ModelConfig(
        model_type="gradient_boosting",
        target_column="target",
        hyperparameters={
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5,
            "min_samples_split": 2,
            "subsample": 0.8,
        },
        test_size=0.2,
        random_state=42,
    )
    
    training_use_case = container.model_training_use_case
    model_path = Path("models/gradient_boosting_model.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    trained_model = training_use_case.execute(
        data=processed_data,
        config=model_config,
        model_path=model_path
    )
    
    logger.info(f"✓ Model trained: Gradient Boosting")
    logger.info(f"✓ Model saved: {model_path}")
    logger.info("Model Performance Metrics:")
    for metric, value in trained_model.metrics.items():
        logger.info(f"  • {metric}: {value:.4f}")
    
    if trained_model.feature_importance:
        logger.info("Feature Importance (Top 5):")
        sorted_features = sorted(
            trained_model.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for i, (feature, importance) in enumerate(sorted_features[:5], 1):
            logger.info(f"  {i}. {feature}: {importance:.4f}")
    
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
    
    # Save predictions with confidence scores
    predictions_path = Path("outputs/gradient_boosting/predictions.csv")
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    
    results_df = test_data.copy()
    results_df['prediction'] = predictions.predictions.values
    results_df['actual'] = processed_data.data[model_config.target_column].values
    if predictions.confidence_scores is not None:
        results_df['confidence'] = predictions.confidence_scores.values
    results_df['correct'] = (results_df['prediction'] == results_df['actual']).astype(int)
    results_df.to_csv(predictions_path, index=False)
    
    accuracy = results_df['correct'].mean()
    avg_confidence = results_df['confidence'].mean() if 'confidence' in results_df else 0
    logger.info(f"✓ Predictions saved: {predictions_path}")
    logger.info(f"✓ Prediction accuracy: {accuracy:.2%}")
    logger.info(f"✓ Average confidence: {avg_confidence:.2%}")
    
    # ========== FINAL SUMMARY ==========
    print("\n" + "="*70)
    print("GRADIENT BOOSTING PIPELINE - COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\nData Shape: {processed_data.data.shape}")
    print(f"Target Column: {model_config.target_column}")
    print(f"Model Metrics: {trained_model.metrics}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Learning Rate: {model_config.hyperparameters.get('learning_rate', 0.1)}")
    print(f"Number of Estimators: {model_config.hyperparameters.get('n_estimators', 100)}")
    print(f"Max Depth: {model_config.hyperparameters.get('max_depth', 5)}")
    print(f"Model Saved: {model_path}")
    print(f"EDA Outputs: {eda_output_dir}")
    print(f"Predictions: {predictions_path}")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        run_gradient_boosting_pipeline()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
