"""Configuration management."""

import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class PathConfig(BaseModel):
    """Path configurations."""
    
    data_dir: Path = Field(default=Path("data"))
    raw_data_dir: Path = Field(default=Path("data/raw"))
    processed_data_dir: Path = Field(default=Path("data/processed"))
    models_dir: Path = Field(default=Path("models"))
    outputs_dir: Path = Field(default=Path("outputs"))
    
    def create_directories(self) -> None:
        """Create all configured directories."""
        for path in [
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.models_dir,
            self.outputs_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


class MLConfig(BaseModel):
    """Machine learning configurations."""
    
    random_seed: int = Field(default=42)
    test_size: float = Field(default=0.2, ge=0.0, le=1.0)
    validation_size: float = Field(default=0.2, ge=0.0, le=1.0)
    default_model_type: str = Field(default="random_forest")
    enable_hyperparameter_tuning: bool = Field(default=False)


class OCRConfig(BaseModel):
    """OCR configurations."""
    
    tesseract_path: str | None = Field(default=None)
    language: str = Field(default="eng")
    dpi: int = Field(default=300)


class LogConfig(BaseModel):
    """Logging configurations."""
    
    level: str = Field(default="INFO")
    format: str = Field(
        default="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
        "<level>{message}</level>"
    )
    rotation: str = Field(default="10 MB")
    retention: str = Field(default="1 week")


class Settings(BaseModel):
    """Application settings."""
    
    environment: str = Field(default="development")
    paths: PathConfig = Field(default_factory=PathConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    log: LogConfig = Field(default_factory=LogConfig)
    
    @classmethod
    def load_from_env(cls, env_file: str = ".env") -> "Settings":
        """
        Load settings from environment variables.
        
        Args:
            env_file: Path to .env file
            
        Returns:
            Settings instance
        """
        load_dotenv(env_file, override=True)
        
        return cls(
            environment=os.getenv("ENVIRONMENT", "development"),
            paths=PathConfig(
                data_dir=Path(os.getenv("DATA_DIR", "data")),
                raw_data_dir=Path(os.getenv("RAW_DATA_DIR", "data/raw")),
                processed_data_dir=Path(os.getenv("PROCESSED_DATA_DIR", "data/processed")),
                models_dir=Path(os.getenv("MODELS_DIR", "models")),
                outputs_dir=Path(os.getenv("OUTPUTS_DIR", "outputs")),
            ),
            ml=MLConfig(
                random_seed=int(os.getenv("RANDOM_SEED", "42")),
                test_size=float(os.getenv("TEST_SIZE", "0.2")),
                validation_size=float(os.getenv("VALIDATION_SIZE", "0.2")),
                default_model_type=os.getenv("DEFAULT_MODEL_TYPE", "random_forest"),
                enable_hyperparameter_tuning=os.getenv(
                    "ENABLE_HYPERPARAMETER_TUNING", "false"
                ).lower()
                == "true",
            ),
            ocr=OCRConfig(
                tesseract_path=os.getenv("TESSERACT_PATH"),
                language=os.getenv("OCR_LANGUAGE", "eng"),
                dpi=int(os.getenv("OCR_DPI", "300")),
            ),
            log=LogConfig(
                level=os.getenv("LOG_LEVEL", "INFO"),
            ),
        )
    
    def initialize(self) -> None:
        """Initialize application (create directories, etc.)."""
        self.paths.create_directories()


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings.load_from_env()
        _settings.initialize()
    return _settings


def reset_settings() -> None:
    """Reset global settings (useful for testing)."""
    global _settings
    _settings = None
