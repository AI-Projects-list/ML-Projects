"""End-to-end ML pipeline use case."""

from pathlib import Path
from typing import Optional

from loguru import logger

from src.domain.entities import DataSource, ModelConfig, Prediction
from src.application.use_cases.data_ingestion import DataIngestionUseCase
from src.application.use_cases.eda import EDAUseCase
from src.application.use_cases.model_training import ModelTrainingUseCase
from src.application.use_cases.prediction import PredictionUseCase


class MLPipelineUseCase:
    """Orchestrates the complete end-to-end ML pipeline."""
    
    def __init__(
        self,
        data_ingestion: DataIngestionUseCase,
        eda: EDAUseCase,
        model_training: ModelTrainingUseCase,
        prediction: PredictionUseCase,
    ):
        """
        Initialize ML pipeline use case.
        
        Args:
            data_ingestion: Data ingestion use case
            eda: EDA use case
            model_training: Model training use case
            prediction: Prediction use case
        """
        self.data_ingestion = data_ingestion
        self.eda = eda
        self.model_training = model_training
        self.prediction = prediction
    
    def execute(
        self,
        source: DataSource,
        model_config: ModelConfig,
        perform_eda: bool = True,
        eda_output_dir: Optional[Path] = None,
        model_output_path: Optional[Path] = None,
    ) -> dict:
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
        logger.info("=" * 60)
        logger.info("Starting End-to-End ML Pipeline")
        logger.info("=" * 60)
        
        results = {}
        
        # Step 1: Data Ingestion
        logger.info("\n[1/4] Data Ingestion & Preprocessing")
        logger.info("-" * 60)
        processed_data = self.data_ingestion.execute(source)
        results["processed_data"] = processed_data
        logger.info(f"✓ Processed data shape: {processed_data.data.shape}")
        
        # Step 2: EDA (optional)
        if perform_eda:
            logger.info("\n[2/4] Exploratory Data Analysis")
            logger.info("-" * 60)
            eda_report = self.eda.execute(
                processed_data,
                generate_plots=True,
                output_dir=eda_output_dir,
            )
            results["eda_report"] = eda_report
            logger.info(f"✓ Generated {len(eda_report.insights)} insights")
            logger.info(f"✓ Created {len(eda_report.visualizations)} visualizations")
        else:
            logger.info("\n[2/4] Exploratory Data Analysis (SKIPPED)")
            logger.info("-" * 60)
        
        # Step 3: Model Training
        logger.info("\n[3/4] Model Training")
        logger.info("-" * 60)
        trained_model = self.model_training.execute(
            processed_data,
            model_config,
            save_model=True,
            model_path=model_output_path,
        )
        results["trained_model"] = trained_model
        logger.info(f"✓ Model trained: {model_config.model_type}")
        logger.info(f"✓ Metrics: {trained_model.metrics}")
        
        # Step 4: Generate predictions on training data (as example)
        logger.info("\n[4/4] Generating Predictions")
        logger.info("-" * 60)
        prediction = self.prediction.execute(
            processed_data.data,
            model=trained_model,
        )
        results["predictions"] = prediction
        logger.info(f"✓ Generated {len(prediction.predictions)} predictions")
        
        logger.info("\n" + "=" * 60)
        logger.info("ML Pipeline Completed Successfully!")
        logger.info("=" * 60)
        
        return results
