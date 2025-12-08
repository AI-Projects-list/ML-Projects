# ML Ollama - End-to-End ML Pipeline

A production-ready, end-to-end machine learning pipeline built with **Clean + Hexagonal Architecture**. Designed for scalability, maintainability, and extensibility.

## 🌟 Features

- **Multi-Format Data Support**: CSV, TXT, PDF, and scanned PDF (with OCR)
- **Complete ML Pipeline**: Data ingestion → EDA → Training → Prediction
- **Clean Architecture**: Separation of concerns with domain, application, infrastructure, and presentation layers
- **Extensible Design**: Easy to add new data sources, models, or processing steps
- **Production Ready**: Logging, configuration management, error handling
- **CLI Interface**: User-friendly command-line interface with rich output

## 🏗️ Architecture

The project follows **Clean + Hexagonal Architecture** principles:

```
src/
├── domain/              # Business logic & entities (innermost layer)
│   ├── entities.py      # Core business objects
│   ├── repositories.py  # Port interfaces
│   └── value_objects.py # Immutable domain values
│
├── application/         # Use cases & workflows
│   └── use_cases/       # Business workflows
│       ├── data_ingestion.py
│       ├── eda.py
│       ├── model_training.py
│       ├── prediction.py
│       └── ml_pipeline.py
│
├── infrastructure/      # External implementations
│   ├── data_readers/    # File format readers
│   │   ├── csv_reader.py
│   │   ├── text_reader.py
│   │   ├── pdf_reader.py
│   │   └── scanned_pdf_reader.py
│   ├── processing/      # Data processing
│   │   ├── data_processor.py
│   │   └── eda_analyzer.py
│   ├── ml/              # ML implementations
│   │   ├── model_trainer.py
│   │   ├── predictor.py
│   │   └── model_repository.py
│   ├── persistence/     # Data storage
│   └── config/          # Configuration
│
└── presentation/        # User interfaces
    └── cli.py           # Command-line interface
```

### Architecture Principles

- **Dependency Inversion**: Inner layers don't depend on outer layers
- **Interface Segregation**: Small, focused interfaces
- **Single Responsibility**: Each module has one clear purpose
- **Open/Closed**: Easy to extend without modifying existing code

## 📋 Requirements

- Python 3.9+
- Poetry (for dependency management)
- Tesseract OCR (for scanned PDF processing)

## 🚀 Installation

### 1. Clone the repository

```powershell
git clone <repository-url>
cd ML_Ollama
```

### 2. Install Poetry (if not already installed)

```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

### 3. Install dependencies

```powershell
poetry install
```

### 4. Install Tesseract (for OCR support)

**Windows:**
Download from: https://github.com/UB-Mannheim/tesseract/wiki

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

### 5. Configure environment

```powershell
cp .env.example .env
# Edit .env with your settings
```

## 💻 Usage

### Command-Line Interface

#### Run Complete Pipeline

```powershell
poetry run ml-pipeline run-pipeline data/raw/data.csv --target-column target --model-type random_forest
```

#### Individual Commands

**Data Ingestion:**
```powershell
poetry run ml-pipeline ingest data/raw/data.csv --data-type csv --output-path data/processed/data.pkl
```

**Exploratory Data Analysis:**
```powershell
poetry run ml-pipeline eda data/processed/data.pkl --output-dir outputs/eda
```

**Train Model:**
```powershell
poetry run ml-pipeline train data/processed/data.pkl --target-column target --model-type random_forest
```

**Make Predictions:**
```powershell
poetry run ml-pipeline predict models/model.pkl data/raw/new_data.csv
```

### Python API

#### Example: Complete Pipeline

```python
from pathlib import Path
from src.domain.entities import DataSource, DataSourceType, ModelConfig
from src.infrastructure.config.container import Container
from src.infrastructure.config.settings import get_settings

# Initialize
settings = get_settings()
container = Container(settings)

# Create data source
source = DataSource(
    source_type=DataSourceType.CSV,
    path="data/raw/data.csv"
)

# Configure model
model_config = ModelConfig(
    model_type="random_forest",
    target_column="target",
    test_size=0.2
)

# Execute pipeline
pipeline = container.ml_pipeline_use_case
results = pipeline.execute(
    source=source,
    model_config=model_config,
    perform_eda=True,
    eda_output_dir=Path("outputs/eda"),
    model_output_path=Path("models/model.pkl")
)

# Access results
print(f"Model metrics: {results['trained_model'].metrics}")
```

#### Example: PDF Processing

```python
from src.domain.entities import DataSource, DataSourceType

source = DataSource(
    source_type=DataSourceType.PDF,
    path="data/raw/document.pdf",
    metadata={"extract_tables": True}
)

processed_data = container.data_ingestion_use_case.execute(source)
print(f"Extracted {len(processed_data.data)} pages")
```

#### Example: OCR Processing

```python
from src.domain.entities import DataSource, DataSourceType

source = DataSource(
    source_type=DataSourceType.PDF_SCAN,
    path="data/raw/scanned.pdf",
    metadata={"language": "eng", "dpi": 300}
)

processed_data = container.data_ingestion_use_case.execute(source)
print(f"OCR confidence: {processed_data.data['ocr_confidence'].mean():.2f}%")
```

## 📊 Supported Data Formats

| Format | Type | Description |
|--------|------|-------------|
| CSV | `csv` | Comma-separated values |
| TXT | `txt` | Plain text files |
| PDF | `pdf` | Text-based PDF documents |
| PDF Scan | `pdf_scan` | Scanned PDF (requires OCR) |

## 🤖 Supported ML Models

| Model | Type | Use Case |
|-------|------|----------|
| Linear Regression | `linear_regression` | Regression |
| Logistic Regression | `logistic_regression` | Classification |
| Decision Tree | `decision_tree` | Classification/Regression |
| Random Forest | `random_forest` | Classification/Regression |
| Gradient Boosting | `gradient_boosting` | Classification/Regression |

## 📁 Project Structure

```
ML_Ollama/
├── src/
│   ├── domain/              # Business logic
│   ├── application/         # Use cases
│   ├── infrastructure/      # Implementations
│   └── presentation/        # UI (CLI)
├── examples/                # Example scripts
├── data/
│   ├── raw/                 # Raw input data
│   └── processed/           # Processed data
├── models/                  # Trained models
├── outputs/                 # Results & visualizations
├── logs/                    # Application logs
├── pyproject.toml          # Poetry configuration
└── README.md               # This file
```

## 🔧 Configuration

Edit `.env` file:

```env
# Environment
ENVIRONMENT=development
LOG_LEVEL=INFO

# Paths
DATA_DIR=data
RAW_DATA_DIR=data/raw
PROCESSED_DATA_DIR=data/processed
MODELS_DIR=models
OUTPUTS_DIR=outputs

# ML Configuration
RANDOM_SEED=42
TEST_SIZE=0.2
DEFAULT_MODEL_TYPE=random_forest

# OCR Configuration
TESSERACT_PATH=/usr/bin/tesseract
OCR_LANGUAGE=eng
```

## 🎯 Key Design Patterns

- **Repository Pattern**: Abstract data access
- **Factory Pattern**: Create readers dynamically
- **Dependency Injection**: Loose coupling via container
- **Use Case Pattern**: Encapsulate business workflows
- **Strategy Pattern**: Pluggable algorithms

## 🧪 Development

### Run Examples

```powershell
poetry run python examples/example_csv_pipeline.py
poetry run python examples/example_pdf_processing.py
poetry run python examples/example_ocr_processing.py
poetry run python examples/example_eda.py
```

### Code Quality

```powershell
# Format code
poetry run black src/

# Sort imports
poetry run isort src/

# Type checking
poetry run mypy src/

# Linting
poetry run flake8 src/
```

## 🎨 EDA Outputs

The EDA module generates:

- Missing values heatmap
- Distribution plots
- Correlation matrix
- Box plots (outlier detection)
- Categorical distributions
- Automated insights

## 🔄 Extending the Pipeline

### Add a New Data Reader

```python
from src.domain.repositories import IDataReader

class MyCustomReader(IDataReader):
    def can_read(self, source: DataSource) -> bool:
        # Implementation
        pass
    
    def read(self, source: DataSource) -> pd.DataFrame:
        # Implementation
        pass

# Register with factory
container.data_reader_factory.add_reader(MyCustomReader())
```

### Add a New Model

Edit `src/infrastructure/ml/model_trainer.py`:

```python
SUPPORTED_MODELS = {
    "my_model": MyModelClass,
    # ... existing models
}
```

## 📝 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please follow the existing architecture patterns.

## 📧 Contact

For questions or support, please open an issue.

---

**Built with Clean Architecture for scalability and maintainability** 🚀
