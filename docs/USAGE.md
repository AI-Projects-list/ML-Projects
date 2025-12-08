# Usage Guide

This guide provides detailed usage instructions and examples.

## Table of Contents

1. [Quick Start](#quick-start)
2. [CLI Usage](#cli-usage)
3. [Python API Usage](#python-api-usage)
4. [Data Formats](#data-formats)
5. [Model Configuration](#model-configuration)
6. [Advanced Usage](#advanced-usage)

## Quick Start

### 1. Install dependencies

```powershell
poetry install
```

### 2. Prepare your data

Place your data file in `data/raw/`:
```
data/raw/my_data.csv
```

### 3. Run the pipeline

```powershell
poetry run ml-pipeline run-pipeline data/raw/my_data.csv `
  --target-column target `
  --model-type random_forest `
  --perform-eda
```

### 4. Check outputs

- **EDA visualizations**: `outputs/eda/`
- **Trained model**: `outputs/models/`
- **Logs**: `logs/`

## CLI Usage

### Complete Pipeline

Run the entire ML pipeline:

```powershell
poetry run ml-pipeline run-pipeline <data-path> [OPTIONS]
```

**Options**:
- `--data-type`: Data source type (csv, txt, pdf, pdf_scan)
- `--target-column`: Target column for ML model
- `--model-type`: Model type (random_forest, logistic_regression, etc.)
- `--test-size`: Test set size ratio (default: 0.2)
- `--perform-eda`: Perform EDA (default: True)
- `--output-dir`: Output directory (default: outputs)

**Example**:
```powershell
poetry run ml-pipeline run-pipeline data/raw/iris.csv `
  --target-column species `
  --model-type random_forest `
  --test-size 0.3
```

### Data Ingestion

Process and save data:

```powershell
poetry run ml-pipeline ingest <data-path> [OPTIONS]
```

**Example**:
```powershell
poetry run ml-pipeline ingest data/raw/data.csv `
  --data-type csv `
  --output-path data/processed/data.pkl `
  --clean `
  --transform
```

### Exploratory Data Analysis

Perform EDA on processed data:

```powershell
poetry run ml-pipeline eda <processed-data-path> [OPTIONS]
```

**Example**:
```powershell
poetry run ml-pipeline eda data/processed/data.pkl `
  --output-dir outputs/my_eda
```

### Model Training

Train a model:

```powershell
poetry run ml-pipeline train <processed-data-path> [OPTIONS]
```

**Example**:
```powershell
poetry run ml-pipeline train data/processed/data.pkl `
  --target-column price `
  --model-type gradient_boosting `
  --output-path models/my_model.pkl
```

### Making Predictions

Use a trained model:

```powershell
poetry run ml-pipeline predict <model-path> <data-path> [OPTIONS]
```

**Example**:
```powershell
poetry run ml-pipeline predict models/my_model.pkl data/raw/new_data.csv `
  --output-path outputs/predictions.csv
```

## Python API Usage

### Complete Pipeline

```python
from pathlib import Path
from src.domain.entities import DataSource, DataSourceType, ModelConfig
from src.infrastructure.config.container import Container
from src.infrastructure.config.settings import get_settings

# Initialize
settings = get_settings()
container = Container(settings)

# Configure data source
source = DataSource(
    source_type=DataSourceType.CSV,
    path="data/raw/data.csv",
    metadata={"encoding": "utf-8"}
)

# Configure model
model_config = ModelConfig(
    model_type="random_forest",
    target_column="target",
    hyperparameters={
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5
    },
    test_size=0.2,
    random_state=42
)

# Run pipeline
pipeline = container.ml_pipeline_use_case
results = pipeline.execute(
    source=source,
    model_config=model_config,
    perform_eda=True,
    eda_output_dir=Path("outputs/eda"),
    model_output_path=Path("models/model.pkl")
)

# Access results
processed_data = results["processed_data"]
eda_report = results["eda_report"]
trained_model = results["trained_model"]
predictions = results["predictions"]

print(f"Model accuracy: {trained_model.metrics['accuracy']:.2%}")
```

### Individual Components

#### Data Ingestion

```python
from src.domain.entities import DataSource, DataSourceType

source = DataSource(
    source_type=DataSourceType.CSV,
    path="data/raw/data.csv"
)

use_case = container.data_ingestion_use_case
processed_data = use_case.execute(
    source=source,
    clean=True,
    transform=True,
    validate=True
)

print(f"Data shape: {processed_data.data.shape}")
print(f"Steps: {processed_data.processing_steps}")
```

#### EDA Only

```python
from pathlib import Path

# Load processed data
data_repo = container.data_repository
processed_data = data_repo.load(Path("data/processed/data.pkl"))

# Perform EDA
eda_use_case = container.eda_use_case
report = eda_use_case.execute(
    data=processed_data,
    generate_plots=True,
    output_dir=Path("outputs/eda")
)

# View insights
for insight in report.insights:
    print(f"- {insight}")
```

#### Training Only

```python
from src.domain.entities import ModelConfig

# Load processed data
processed_data = container.data_repository.load(Path("data/processed/data.pkl"))

# Configure model
config = ModelConfig(
    model_type="logistic_regression",
    target_column="target",
    hyperparameters={"C": 1.0, "max_iter": 1000}
)

# Train
training_use_case = container.model_training_use_case
model = training_use_case.execute(
    data=processed_data,
    config=config,
    save_model=True,
    model_path=Path("models/logistic_model.pkl")
)

print(f"Metrics: {model.metrics}")
```

#### Prediction Only

```python
import pandas as pd
from pathlib import Path

# Load new data
new_data = pd.read_csv("data/raw/new_data.csv")

# Make predictions
prediction_use_case = container.prediction_use_case
predictions = prediction_use_case.execute(
    data=new_data,
    model_path=Path("models/model.pkl")
)

# Save results
new_data["prediction"] = predictions.predictions
new_data.to_csv("outputs/predictions.csv", index=False)
```

## Data Formats

### CSV Files

```python
source = DataSource(
    source_type=DataSourceType.CSV,
    path="data/raw/data.csv",
    metadata={
        "encoding": "utf-8",
        "read_params": {
            "sep": ",",
            "header": 0,
            "index_col": None
        }
    }
)
```

### Text Files

```python
source = DataSource(
    source_type=DataSourceType.TXT,
    path="data/raw/document.txt",
    metadata={
        "encoding": "utf-8",
        "mode": "lines"  # or "full" for entire document
    }
)
```

### PDF Files

```python
source = DataSource(
    source_type=DataSourceType.PDF,
    path="data/raw/document.pdf",
    metadata={
        "extract_tables": True  # Extract tables from PDF
    }
)
```

### Scanned PDF (OCR)

```python
source = DataSource(
    source_type=DataSourceType.PDF_SCAN,
    path="data/raw/scanned.pdf",
    metadata={
        "language": "eng",  # OCR language
        "dpi": 300  # Image resolution
    }
)
```

## Model Configuration

### Linear Regression

```python
config = ModelConfig(
    model_type="linear_regression",
    target_column="price",
    hyperparameters={}
)
```

### Logistic Regression

```python
config = ModelConfig(
    model_type="logistic_regression",
    target_column="class",
    hyperparameters={
        "C": 1.0,
        "max_iter": 1000,
        "solver": "lbfgs"
    }
)
```

### Decision Tree

```python
config = ModelConfig(
    model_type="decision_tree",
    target_column="category",
    hyperparameters={
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2
    }
)
```

### Random Forest

```python
config = ModelConfig(
    model_type="random_forest",
    target_column="target",
    hyperparameters={
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2
    }
)
```

### Gradient Boosting

```python
config = ModelConfig(
    model_type="gradient_boosting",
    target_column="target",
    hyperparameters={
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3
    }
)
```

## Advanced Usage

### Custom Data Reader

```python
from src.domain.repositories import IDataReader
import pandas as pd

class JSONDataReader(IDataReader):
    def can_read(self, source: DataSource) -> bool:
        return source.path.endswith('.json')
    
    def read(self, source: DataSource) -> pd.DataFrame:
        return pd.read_json(source.path)

# Register
container.data_reader_factory.add_reader(JSONDataReader())
```

### Custom Processing Steps

```python
# Access processor
processor = container.data_processor

# Custom cleaning
df = processor.clean(raw_data)

# Custom transformation
df = processor.transform(df)

# Validate
is_valid = processor.validate(df)
```

### Feature Selection

```python
config = ModelConfig(
    model_type="random_forest",
    target_column="target",
    feature_columns=["feature1", "feature2", "feature3"],  # Specify features
    test_size=0.2
)
```

### Access Feature Importance

```python
trained_model = training_use_case.execute(data, config)

if trained_model.feature_importance:
    sorted_features = sorted(
        trained_model.feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    print("Top features:")
    for feature, importance in sorted_features[:10]:
        print(f"  {feature}: {importance:.4f}")
```

### Batch Processing

```python
import glob

# Process multiple files
for file_path in glob.glob("data/raw/*.csv"):
    source = DataSource(
        source_type=DataSourceType.CSV,
        path=file_path
    )
    
    processed = container.data_ingestion_use_case.execute(source)
    
    output_name = Path(file_path).stem + "_processed.pkl"
    container.data_repository.save(
        processed,
        Path("data/processed") / output_name
    )
```

## Environment Configuration

Edit `.env`:

```env
# Paths
DATA_DIR=data
MODELS_DIR=models
OUTPUTS_DIR=outputs

# ML Settings
RANDOM_SEED=42
TEST_SIZE=0.2
DEFAULT_MODEL_TYPE=random_forest

# OCR Settings (for scanned PDFs)
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
OCR_LANGUAGE=eng
OCR_DPI=300

# Logging
LOG_LEVEL=INFO
```

---

For more examples, see the `examples/` directory.
