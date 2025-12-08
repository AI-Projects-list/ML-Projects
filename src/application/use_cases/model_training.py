"""Use case for model training."""

from pathlib import Path
from typing import Optional

from loguru import logger

from src.domain.entities import ModelConfig, ProcessedData, TrainedModel
from src.domain.repositories import IModelRepository, IModelTrainer


class ModelTrainingUseCase:
    """Handles the model training workflow."""
    
    def __init__(
        self,
        trainer: IModelTrainer,
        repository: IModelRepository,
    ):
        """
        Initialize model training use case.
        
        Args:
            trainer: Model trainer implementation
            repository: Model repository for persistence
        """
        self.trainer = trainer
        self.repository = repository
    
    def execute(
        self,
        data: ProcessedData,
        config: ModelConfig,
        save_model: bool = True,
        model_path: Optional[Path] = None,
    ) -> TrainedModel:
        """
        Execute the model training workflow.
        
        Args:
            data: Processed data for training
            config: Model configuration
            save_model: Whether to save the trained model
            model_path: Path for saving the model
            
        Returns:
            Trained model with evaluation metrics
        """
        logger.info(f"Starting model training workflow for {config.model_type}")
        
        # Train model
        trained_model = self.trainer.train(data, config)
        
        # Save model if requested
        if save_model:
            if model_path is None:
                model_path = Path(f"models/{config.model_type}_model.pkl")
            
            logger.info(f"Saving model to {model_path}")
            self.repository.save(trained_model, model_path)
        
        logger.info("Model training workflow completed")
        logger.info(f"Model metrics: {trained_model.metrics}")
        
        return trained_model
