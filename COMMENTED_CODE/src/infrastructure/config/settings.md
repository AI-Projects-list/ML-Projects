# settings.py - Complete Documentation

**Source**: `src/infrastructure/config/settings.py`  
**Purpose**: Application configuration management with Pydantic  
**Layer**: Infrastructure (Config)  
**Lines**: 137  
**Patterns**: Settings Pattern, Factory Method

---

## Why Pydantic for Settings?

**Benefits**:
- ✅ Runtime type validation
- ✅ Environment variable parsing
- ✅ Default values with Field()
- ✅ Data validation (ge=0.0, le=1.0)
- ✅ Auto-documentation
- ✅ JSON schema generation

**Trade-offs**:
- ❌ External dependency
- ❌ Learning curve
- ✅ Better than manual parsing

---

## Complete Annotated Code

```python
"""Configuration management."""
# WHAT: Centralized settings management
# WHY: Single source of truth for configuration
# PATTERN: Settings Object Pattern
# BENEFIT: Type-safe, validated configuration

import os
# WHAT: OS module for environment variables
# WHY: Read from .env and system environment
# USAGE: os.getenv("VAR_NAME", "default")

from pathlib import Path
# WHAT: Modern path handling
# WHY: Cross-platform file paths
# BENEFIT: Object-oriented, chainable

from typing import Any, Dict
# WHAT: Type hints
# WHY: Type safety

from dotenv import load_dotenv
# WHAT: python-dotenv library
# WHY: Load .env file into environment
# BENEFIT: Separate config from code
# USAGE: load_dotenv(".env")

from pydantic import BaseModel, Field
# WHAT: Pydantic for data validation
# WHY: Runtime type checking and validation
# BaseModel: Base class for models
# Field: Define field with validation rules


class PathConfig(BaseModel):
    """Path configurations."""
    # WHAT: Group all path-related settings
    # WHY: Organize settings by category
    # BENEFIT: Namespace, easy to extend
    
    data_dir: Path = Field(default=Path("data"))
    # WHAT: Main data directory
    # WHY: Root for all data files
    # DEFAULT: "data/" relative to project root
    # TYPE: Path (cross-platform)
    
    raw_data_dir: Path = Field(default=Path("data/raw"))
    # WHAT: Raw data storage
    # WHY: Separate raw from processed
    # PATTERN: Data lake structure
    
    processed_data_dir: Path = Field(default=Path("data/processed"))
    # WHAT: Processed data storage
    # WHY: Cache preprocessing results
    
    models_dir: Path = Field(default=Path("models"))
    # WHAT: Trained models storage
    # WHY: Model versioning, deployment
    
    outputs_dir: Path = Field(default=Path("outputs"))
    # WHAT: Pipeline outputs
    # WHY: Reports, visualizations, predictions
    
    def create_directories(self) -> None:
        """Create all configured directories."""
        # WHAT: Ensure directories exist
        # WHY: Avoid FileNotFoundError
        # WHEN: Call during app initialization
        
        for path in [
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.models_dir,
            self.outputs_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)
            # WHAT: Create directory recursively
            # parents=True: Create parent dirs if needed
            # exist_ok=True: Don't fail if exists
            # BENEFIT: Idempotent operation


class MLConfig(BaseModel):
    """Machine learning configurations."""
    # WHAT: ML-specific settings
    # WHY: Centralize ML parameters
    
    random_seed: int = Field(default=42)
    # WHAT: Random seed for reproducibility
    # WHY: Same results across runs
    # DEFAULT: 42 (convention)
    # USAGE: np.random.seed(), random_state parameter
    
    test_size: float = Field(default=0.2, ge=0.0, le=1.0)
    # WHAT: Test set proportion
    # WHY: Train/test split ratio
    # DEFAULT: 0.2 = 20% test, 80% train
    # VALIDATION: ge=0.0, le=1.0 (between 0 and 1)
    # BENEFIT: Pydantic validates automatically
    
    validation_size: float = Field(default=0.2, ge=0.0, le=1.0)
    # WHAT: Validation set proportion
    # WHY: Hyperparameter tuning
    # USAGE: Train/validation/test split
    
    default_model_type: str = Field(default="random_forest")
    # WHAT: Default ML algorithm
    # WHY: Fallback when not specified
    # TRADE-OFF: String not enum (flexible but less safe)
    
    enable_hyperparameter_tuning: bool = Field(default=False)
    # WHAT: Enable GridSearch/RandomSearch
    # WHY: Expensive operation, opt-in
    # DEFAULT: False (faster)


class OCRConfig(BaseModel):
    """OCR configurations."""
    # WHAT: Optical Character Recognition settings
    # WHY: Configure tesseract for scanned PDFs
    
    tesseract_path: str | None = Field(default=None)
    # WHAT: Path to tesseract executable
    # WHY: System-specific installation
    # DEFAULT: None (use system PATH)
    # WINDOWS: "C:/Program Files/Tesseract-OCR/tesseract.exe"
    # LINUX/MAC: Usually in PATH
    
    language: str = Field(default="eng")
    # WHAT: OCR language
    # WHY: Language-specific character recognition
    # DEFAULT: "eng" (English)
    # OPTIONS: "fra", "deu", "spa", "chi_sim", etc.
    
    dpi: int = Field(default=300)
    # WHAT: DPI for image processing
    # WHY: Higher DPI = better accuracy
    # DEFAULT: 300 (good balance)
    # TRADE-OFF: Higher DPI = slower, more memory


class LogConfig(BaseModel):
    """Logging configurations."""
    # WHAT: Logging settings for loguru
    # WHY: Configure log format, rotation, retention
    
    level: str = Field(default="INFO")
    # WHAT: Minimum log level
    # WHY: Filter log verbosity
    # OPTIONS: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    # DEFAULT: "INFO" (production)
    # DEVELOPMENT: "DEBUG"
    
    format: str = Field(
        default="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
        "<level>{message}</level>"
    )
    # WHAT: Log message format (loguru syntax)
    # WHY: Structured, readable logs
    # FEATURES:
    #   - Colored timestamps (green)
    #   - Level padding ({level: <8})
    #   - Module and function names
    #   - Colored messages by level
    
    rotation: str = Field(default="10 MB")
    # WHAT: Log file rotation trigger
    # WHY: Prevent huge log files
    # DEFAULT: "10 MB" (rotate at 10 megabytes)
    # OPTIONS: "500 MB", "1 GB", "1 day", "1 week"
    
    retention: str = Field(default="1 week")
    # WHAT: How long to keep old logs
    # WHY: Disk space management
    # DEFAULT: "1 week"
    # OPTIONS: "1 month", "90 days", "1 year"


class Settings(BaseModel):
    """Application settings."""
    # WHAT: Master settings class
    # WHY: Combine all config categories
    # PATTERN: Composite Settings
    
    environment: str = Field(default="development")
    # WHAT: Application environment
    # WHY: Environment-specific behavior
    # OPTIONS: "development", "staging", "production"
    # USAGE: if settings.environment == "production"
    
    paths: PathConfig = Field(default_factory=PathConfig)
    # WHAT: Path settings
    # WHY: Nested configuration
    # DEFAULT: PathConfig() with defaults
    # USAGE: settings.paths.data_dir
    
    ml: MLConfig = Field(default_factory=MLConfig)
    # WHAT: ML settings
    # DEFAULT: MLConfig() with defaults
    # USAGE: settings.ml.random_seed
    
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    # WHAT: OCR settings
    # USAGE: settings.ocr.tesseract_path
    
    log: LogConfig = Field(default_factory=LogConfig)
    # WHAT: Logging settings
    # USAGE: settings.log.level
    
    @classmethod
    def load_from_env(cls, env_file: str = ".env") -> "Settings":
        """Load settings from environment variables."""
        # WHAT: Factory method to load from .env
        # WHY: 12-factor app configuration
        # PATTERN: Factory Method
        # USAGE: settings = Settings.load_from_env()
        
        load_dotenv(env_file, override=True)
        # WHAT: Load .env file
        # WHY: Load variables into os.environ
        # override=True: Overwrite existing vars
        
        return cls(
            environment=os.getenv("ENVIRONMENT", "development"),
            # WHAT: Get from env or use default
            # WHY: Environment-specific config
            
            paths=PathConfig(
                data_dir=Path(os.getenv("DATA_DIR", "data")),
                raw_data_dir=Path(os.getenv("RAW_DATA_DIR", "data/raw")),
                processed_data_dir=Path(os.getenv("PROCESSED_DATA_DIR", "data/processed")),
                models_dir=Path(os.getenv("MODELS_DIR", "models")),
                outputs_dir=Path(os.getenv("OUTPUTS_DIR", "outputs")),
            ),
            # WHAT: Create PathConfig from env vars
            # BENEFIT: Override defaults via environment
            
            ml=MLConfig(
                random_seed=int(os.getenv("RANDOM_SEED", "42")),
                test_size=float(os.getenv("TEST_SIZE", "0.2")),
                validation_size=float(os.getenv("VALIDATION_SIZE", "0.2")),
                default_model_type=os.getenv("DEFAULT_MODEL_TYPE", "random_forest"),
                enable_hyperparameter_tuning=os.getenv(
                    "ENABLE_HYPERPARAMETER_TUNING", "false"
                ).lower() == "true",
            ),
            # WHAT: Parse ML config from env
            # NOTE: Type conversion (int(), float(), bool)
            
            ocr=OCRConfig(
                tesseract_path=os.getenv("TESSERACT_PATH"),
                language=os.getenv("OCR_LANGUAGE", "eng"),
                dpi=int(os.getenv("OCR_DPI", "300")),
            ),
            
            log=LogConfig(
                level=os.getenv("LOG_LEVEL", "INFO"),
                rotation=os.getenv("LOG_ROTATION", "10 MB"),
                retention=os.getenv("LOG_RETENTION", "1 week"),
            ),
        )


# Usage helper
_settings: Settings | None = None

def get_settings() -> Settings:
    """Get or create settings singleton."""
    # WHAT: Singleton pattern for settings
    # WHY: Load once, reuse everywhere
    # PATTERN: Singleton
    
    global _settings
    if _settings is None:
        _settings = Settings.load_from_env()
    return _settings
```

---

## Pros & Cons

### ✅ Pros
1. **Type Safety**: Pydantic validates types
2. **Validation**: Field constraints (ge, le)
3. **Environment Variables**: 12-factor app
4. **Defaults**: Sensible defaults provided
5. **Nested Config**: Organized by category
6. **Singleton**: Load once pattern

### ❌ Cons
1. **Manual Parsing**: load_from_env() verbose
2. **Type Conversion**: Manual int(), float()
3. **No Secrets**: Passwords in .env (use Vault)

---

## Example .env File

```env
ENVIRONMENT=production
DATA_DIR=/data
RANDOM_SEED=42
TEST_SIZE=0.25
LOG_LEVEL=WARNING
TESSERACT_PATH=/usr/bin/tesseract
```

---

**Lines**: 137  
**Classes**: 6  
**Pattern**: Settings Object + Singleton
