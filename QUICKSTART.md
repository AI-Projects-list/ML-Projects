# Quick Start Guide

Get started with ML Ollama in 5 minutes!

## Prerequisites

- Python 3.9 or higher
- Poetry (Python dependency manager)

## Installation

### 1. Install Poetry (if not already installed)

**Windows (PowerShell):**
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

**Linux/macOS:**
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### 2. Clone and Setup Project

```powershell
cd "c:\Users\budis\source\repos\AI projects(ok & nok)\ML_Ollama"
```

### 3. Run Setup Script

```powershell
python scripts/setup.py
```

### 4. Install Dependencies

```powershell
poetry install
```

## Generate Sample Data

```powershell
poetry run python scripts/generate_sample_data.py
```

This creates:
- `data/raw/sample_classification.csv` - Binary classification dataset
- `data/raw/sample_regression.csv` - Regression dataset
- `data/raw/sample_text.txt` - Text dataset

## Run Your First Pipeline

### Option 1: Using CLI

```powershell
poetry run ml-pipeline run-pipeline data/raw/sample_classification.csv `
  --target-column target `
  --model-type random_forest `
  --perform-eda
```

### Option 2: Using Python

```powershell
poetry run python examples/example_csv_pipeline.py
```

## View Results

- **EDA Visualizations**: `outputs/eda/`
- **Trained Model**: `models/`
- **Logs**: `logs/`

## What Just Happened?

The pipeline automatically:

1. ✅ **Loaded** your CSV data
2. ✅ **Cleaned** the data (handled missing values, removed duplicates)
3. ✅ **Transformed** the data (encoded categories, extracted features)
4. ✅ **Analyzed** the data (generated statistics and visualizations)
5. ✅ **Trained** a Random Forest model
6. ✅ **Evaluated** model performance
7. ✅ **Saved** the trained model

## Next Steps

### Try Different Data Formats

**PDF Processing:**
```powershell
# Place a PDF in data/raw/document.pdf
poetry run python examples/example_pdf_processing.py
```

**OCR (Scanned PDF):**
```powershell
# Install Tesseract OCR first
# Place a scanned PDF in data/raw/scanned_document.pdf
poetry run python examples/example_ocr_processing.py
```

### Try Different Models

**Logistic Regression:**
```powershell
poetry run ml-pipeline run-pipeline data/raw/sample_classification.csv `
  --target-column target `
  --model-type logistic_regression
```

**Gradient Boosting:**
```powershell
poetry run ml-pipeline run-pipeline data/raw/sample_classification.csv `
  --target-column target `
  --model-type gradient_boosting
```

**Linear Regression (for regression tasks):**
```powershell
poetry run ml-pipeline run-pipeline data/raw/sample_regression.csv `
  --target-column price `
  --model-type linear_regression
```

### Run Individual Steps

**1. Data Ingestion Only:**
```powershell
poetry run ml-pipeline ingest data/raw/sample_classification.csv `
  --output-path data/processed/my_data.pkl
```

**2. EDA Only:**
```powershell
poetry run ml-pipeline eda data/processed/my_data.pkl `
  --output-dir outputs/my_eda
```

**3. Train Only:**
```powershell
poetry run ml-pipeline train data/processed/my_data.pkl `
  --target-column target `
  --model-type random_forest `
  --output-path models/my_model.pkl
```

**4. Predict Only:**
```powershell
poetry run ml-pipeline predict models/my_model.pkl `
  data/raw/sample_classification.csv `
  --output-path outputs/my_predictions.csv
```

## Use Your Own Data

### CSV Data

1. Place your CSV file in `data/raw/`
2. Run:
```powershell
poetry run ml-pipeline run-pipeline data/raw/YOUR_FILE.csv `
  --target-column YOUR_TARGET_COLUMN `
  --model-type random_forest
```

### PDF Data

1. Place your PDF in `data/raw/`
2. Run:
```powershell
poetry run ml-pipeline run-pipeline data/raw/YOUR_FILE.pdf `
  --data-type pdf
```

### Scanned PDF (OCR)

1. Install Tesseract OCR
2. Place your scanned PDF in `data/raw/`
3. Run:
```powershell
poetry run ml-pipeline run-pipeline data/raw/YOUR_SCANNED.pdf `
  --data-type pdf_scan
```

## Configuration

Edit `.env` to customize:

```env
# Data paths
DATA_DIR=data
MODELS_DIR=models
OUTPUTS_DIR=outputs

# ML settings
RANDOM_SEED=42
TEST_SIZE=0.2
DEFAULT_MODEL_TYPE=random_forest

# Logging
LOG_LEVEL=INFO
```

## Common Commands

**Get help:**
```powershell
poetry run ml-pipeline --help
```

**Get command-specific help:**
```powershell
poetry run ml-pipeline run-pipeline --help
```

**View logs:**
```powershell
Get-Content logs/*.log -Tail 50
```

## Troubleshooting

### "Command not found: poetry"

Install Poetry first (see step 1 above).

### "Module not found"

Run:
```powershell
poetry install
```

### OCR not working

Install Tesseract OCR and set the path in `.env`:
```env
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
```

### Need more help?

- Check `docs/USAGE.md` for detailed usage
- Check `docs/ARCHITECTURE.md` for architecture details
- Check `examples/` for more examples

## Learn More

- **Architecture**: See `docs/ARCHITECTURE.md`
- **Detailed Usage**: See `docs/USAGE.md`
- **Examples**: See `examples/` directory

---

🎉 **You're ready to build ML pipelines!**
