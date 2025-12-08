"""Dependency injection container."""

from src.application.use_cases.data_ingestion import DataIngestionUseCase
from src.application.use_cases.eda import EDAUseCase
from src.application.use_cases.ml_pipeline import MLPipelineUseCase
from src.application.use_cases.model_training import ModelTrainingUseCase
from src.application.use_cases.prediction import PredictionUseCase
from src.infrastructure.config.settings import Settings
from src.infrastructure.data_readers.factory import DataReaderFactory
from src.infrastructure.ml.model_repository import ModelRepository
from src.infrastructure.ml.model_trainer import ModelTrainer
from src.infrastructure.ml.predictor import Predictor
from src.infrastructure.persistence.data_repository import DataRepository
from src.infrastructure.processing.data_processor import DataProcessor
from src.infrastructure.processing.eda_analyzer import EDAAnalyzer


class Container:
    """Dependency injection container."""
    
    def __init__(self, settings: Settings):
        """
        Initialize container with settings.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        
        # Infrastructure
        self._data_reader_factory = None
        self._data_processor = None
        self._eda_analyzer = None
        self._model_trainer = None
        self._predictor = None
        self._model_repository = None
        self._data_repository = None
        
        # Use cases
        self._data_ingestion_use_case = None
        self._eda_use_case = None
        self._model_training_use_case = None
        self._prediction_use_case = None
        self._ml_pipeline_use_case = None
    
    @property
    def data_reader_factory(self) -> DataReaderFactory:
        """Get data reader factory."""
        if self._data_reader_factory is None:
            self._data_reader_factory = DataReaderFactory()
        return self._data_reader_factory
    
    @property
    def data_processor(self) -> DataProcessor:
        """Get data processor."""
        if self._data_processor is None:
            self._data_processor = DataProcessor()
        return self._data_processor
    
    @property
    def eda_analyzer(self) -> EDAAnalyzer:
        """Get EDA analyzer."""
        if self._eda_analyzer is None:
            self._eda_analyzer = EDAAnalyzer()
        return self._eda_analyzer
    
    @property
    def model_trainer(self) -> ModelTrainer:
        """Get model trainer."""
        if self._model_trainer is None:
            self._model_trainer = ModelTrainer()
        return self._model_trainer
    
    @property
    def predictor(self) -> Predictor:
        """Get predictor."""
        if self._predictor is None:
            self._predictor = Predictor()
        return self._predictor
    
    @property
    def model_repository(self) -> ModelRepository:
        """Get model repository."""
        if self._model_repository is None:
            self._model_repository = ModelRepository()
        return self._model_repository
    
    @property
    def data_repository(self) -> DataRepository:
        """Get data repository."""
        if self._data_repository is None:
            self._data_repository = DataRepository()
        return self._data_repository
    
    @property
    def data_ingestion_use_case(self) -> DataIngestionUseCase:
        """Get data ingestion use case."""
        if self._data_ingestion_use_case is None:
            self._data_ingestion_use_case = DataIngestionUseCase(
                reader_factory=self.data_reader_factory,
                processor=self.data_processor,
            )
        return self._data_ingestion_use_case
    
    @property
    def eda_use_case(self) -> EDAUseCase:
        """Get EDA use case."""
        if self._eda_use_case is None:
            self._eda_use_case = EDAUseCase(analyzer=self.eda_analyzer)
        return self._eda_use_case
    
    @property
    def model_training_use_case(self) -> ModelTrainingUseCase:
        """Get model training use case."""
        if self._model_training_use_case is None:
            self._model_training_use_case = ModelTrainingUseCase(
                trainer=self.model_trainer,
                repository=self.model_repository,
            )
        return self._model_training_use_case
    
    @property
    def prediction_use_case(self) -> PredictionUseCase:
        """Get prediction use case."""
        if self._prediction_use_case is None:
            self._prediction_use_case = PredictionUseCase(
                predictor=self.predictor,
                model_repository=self.model_repository,
            )
        return self._prediction_use_case
    
    @property
    def ml_pipeline_use_case(self) -> MLPipelineUseCase:
        """Get ML pipeline use case."""
        if self._ml_pipeline_use_case is None:
            self._ml_pipeline_use_case = MLPipelineUseCase(
                data_ingestion=self.data_ingestion_use_case,
                eda=self.eda_use_case,
                model_training=self.model_training_use_case,
                prediction=self.prediction_use_case,
            )
        return self._ml_pipeline_use_case
