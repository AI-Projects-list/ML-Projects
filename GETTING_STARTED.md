# Getting Started Checklist

Follow these steps to get your ML pipeline running!

## ☑️ Pre-Setup

- [ ] Python 3.9+ installed
- [ ] PowerShell or terminal access
- [ ] Project files downloaded/cloned

## ☑️ Installation Steps

### Step 1: Install Poetry
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```
- [ ] Poetry installed successfully

### Step 2: Navigate to Project
```powershell
cd "c:\Users\budis\source\repos\AI projects(ok & nok)\ML_Ollama"
```
- [ ] In project directory

### Step 3: Run Setup Script
```powershell
python scripts/setup.py
```
- [ ] Directories created
- [ ] .env file created

### Step 4: Install Dependencies
```powershell
poetry install
```
- [ ] All packages installed (may take a few minutes)

## ☑️ First Run

### Step 5: Generate Sample Data
```powershell
poetry run python scripts/generate_sample_data.py
```
- [ ] Sample CSV files created
- [ ] Sample text file created

### Step 6: Run Your First Pipeline
```powershell
poetry run ml-pipeline run-pipeline data/raw/sample_classification.csv --target-column target --model-type random_forest
```
- [ ] Pipeline executed successfully
- [ ] Model trained
- [ ] EDA visualizations created

### Step 7: Check Outputs
- [ ] Check `outputs/eda/` for visualizations
- [ ] Check `models/` for trained model
- [ ] Check `logs/` for execution logs

## ☑️ Optional: OCR Support

### For Scanned PDF Processing

**Windows:**
1. Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install to: `C:\Program Files\Tesseract-OCR\`
3. Update `.env`:
```env
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
```

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

- [ ] Tesseract installed (if needed)
- [ ] Path configured in .env (if needed)

## ☑️ Verification

### Test Each Component

**1. Data Ingestion:**
```powershell
poetry run ml-pipeline ingest data/raw/sample_classification.csv
```
- [ ] Data loaded and processed

**2. EDA:**
```powershell
poetry run ml-pipeline eda data/processed/data.pkl
```
- [ ] Visualizations generated

**3. Training:**
```powershell
poetry run ml-pipeline train data/processed/data.pkl --target-column target
```
- [ ] Model trained successfully

**4. Prediction:**
```powershell
poetry run ml-pipeline predict models/random_forest_model.pkl data/raw/sample_classification.csv
```
- [ ] Predictions generated

## ☑️ Explore Examples

Run each example to see different capabilities:

```powershell
poetry run python examples/example_csv_pipeline.py
```
- [ ] CSV example works

```powershell
poetry run python examples/example_eda.py
```
- [ ] EDA example works

```powershell
poetry run python examples/example_pdf_processing.py
```
- [ ] PDF example works (requires PDF file)

```powershell
poetry run python examples/example_ocr_processing.py
```
- [ ] OCR example works (requires Tesseract + scanned PDF)

## ☑️ Use Your Own Data

### With CSV Data
1. Place your CSV in `data/raw/your_data.csv`
2. Run:
```powershell
poetry run ml-pipeline run-pipeline data/raw/your_data.csv --target-column YOUR_TARGET --model-type random_forest
```
- [ ] Your data processed successfully

### With PDF Data
1. Place your PDF in `data/raw/your_doc.pdf`
2. Run:
```powershell
poetry run ml-pipeline run-pipeline data/raw/your_doc.pdf --data-type pdf
```
- [ ] PDF processed successfully

## ☑️ Troubleshooting

### Issue: "poetry: command not found"
**Solution:** Restart terminal after installing Poetry, or add Poetry to PATH

### Issue: "Module not found"
**Solution:** Run `poetry install` again

### Issue: Import errors
**Solution:** Make sure you're using `poetry run` before commands

### Issue: OCR not working
**Solution:** Install Tesseract and configure path in `.env`

### Issue: Permission errors
**Solution:** Run terminal as administrator (Windows) or use sudo (Linux/Mac)

## ☑️ Next Steps

- [ ] Read `README.md` for comprehensive documentation
- [ ] Read `QUICKSTART.md` for quick reference
- [ ] Read `docs/ARCHITECTURE.md` to understand the design
- [ ] Read `docs/USAGE.md` for detailed usage examples
- [ ] Explore `examples/` for different use cases

## ☑️ Configuration

Edit `.env` to customize:

```env
# Change these as needed
DATA_DIR=data
MODELS_DIR=models
OUTPUTS_DIR=outputs
RANDOM_SEED=42
TEST_SIZE=0.2
LOG_LEVEL=INFO
```
- [ ] Configuration reviewed
- [ ] Paths customized (if needed)

## ☑️ Development

### Run with Different Models

**Logistic Regression:**
```powershell
poetry run ml-pipeline run-pipeline data/raw/sample_classification.csv --target-column target --model-type logistic_regression
```

**Decision Tree:**
```powershell
poetry run ml-pipeline run-pipeline data/raw/sample_classification.csv --target-column target --model-type decision_tree
```

**Gradient Boosting:**
```powershell
poetry run ml-pipeline run-pipeline data/raw/sample_classification.csv --target-column target --model-type gradient_boosting
```

**Linear Regression (for regression tasks):**
```powershell
poetry run ml-pipeline run-pipeline data/raw/sample_regression.csv --target-column price --model-type linear_regression
```

- [ ] Tried different models

## ☑️ Success Criteria

You're ready when:

- [ ] Poetry installed and working
- [ ] All dependencies installed
- [ ] Sample data generated
- [ ] First pipeline runs successfully
- [ ] Outputs generated (models, visualizations, logs)
- [ ] CLI commands work
- [ ] At least one example runs
- [ ] Can process your own data

## 🎉 All Done!

If all checkboxes are checked, you're ready to build ML solutions!

### Quick Reference Commands

```powershell
# Complete pipeline
poetry run ml-pipeline run-pipeline <data> --target-column <col>

# Individual steps
poetry run ml-pipeline ingest <data>
poetry run ml-pipeline eda <processed-data>
poetry run ml-pipeline train <processed-data> --target-column <col>
poetry run ml-pipeline predict <model> <data>

# Help
poetry run ml-pipeline --help
```

### Get Support

- Check documentation in `docs/`
- Review examples in `examples/`
- Read error messages in `logs/`

---

**Happy ML Pipeline Building! 🚀**
