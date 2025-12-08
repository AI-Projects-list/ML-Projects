# ML Project Flow - Complete Documentation

## Project Overview
This is an end-to-end Machine Learning project built with **Clean + Hexagonal Architecture**, supporting multiple data formats and ML models with a complete pipeline from data ingestion to prediction.

---

## 🏗️ Architecture Layers

### 1. **Domain Layer** (Business Logic Core)
- **Entities**: Core business objects (DataSource, ProcessedData, TrainedModel, etc.)
- **Value Objects**: Immutable objects (FileMetadata, DataQuality, etc.)
- **Repository Interfaces**: Contracts for data access
- **No Dependencies**: Pure business logic

### 2. **Application Layer** (Use Cases)
- **Use Cases**: Orchestrate business workflows
  - `DataIngestionUseCase`: Data loading and preprocessing
  - `EDAUseCase`: Exploratory data analysis
  - `ModelTrainingUseCase`: Model training workflow
  - `PredictionUseCase`: Prediction workflow
  - `MLPipelineUseCase`: Complete end-to-end pipeline

### 3. **Infrastructure Layer** (Technical Implementation)
- **Data Readers**: CSV, TXT, PDF, Scanned PDF (OCR)
- **Data Processors**: Cleaning, transformation, validation
- **EDA Analyzer**: Statistical analysis and visualizations
- **ML Components**: Model trainers, predictors, repository
- **Configuration**: Settings, logging, dependency injection

### 4. **Presentation Layer** (User Interface)
- **CLI**: Typer-based command-line interface
- **Commands**: run-pipeline, ingest, eda, train, predict

---

## 📊 Complete System Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INPUT                               │
│  (CLI Command / Python Script / Direct API Call)                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PRESENTATION LAYER                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  CLI Application (src/presentation/cli.py)                │  │
│  │  - Parse arguments                                        │  │
│  │  - Validate inputs                                        │  │
│  │  - Initialize container                                   │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   APPLICATION LAYER                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Use Case Selection:                                      │  │
│  │  ┌────────────────┐  ┌────────────────┐                  │  │
│  │  │ ML Pipeline UC │  │ Data Ingest UC │                  │  │
│  │  └────────────────┘  └────────────────┘                  │  │
│  │  ┌────────────────┐  ┌────────────────┐                  │  │
│  │  │   EDA UC       │  │  Training UC   │                  │  │
│  │  └────────────────┘  └────────────────┘                  │  │
│  │  ┌────────────────┐                                       │  │
│  │  │ Prediction UC  │                                       │  │
│  │  └────────────────┘                                       │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   INFRASTRUCTURE LAYER                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Data Processing Pipeline:                                │  │
│  │                                                            │  │
│  │  [Reader] → [Processor] → [Analyzer] → [Trainer/Predictor]│  │
│  │                                                            │  │
│  │  Components:                                              │  │
│  │  • Data Readers (CSV, TXT, PDF, PDF+OCR)                 │  │
│  │  • Data Processor (Clean, Transform, Validate)           │  │
│  │  • EDA Analyzer (Statistics, Visualizations)             │  │
│  │  • Model Trainer (5 ML Models)                           │  │
│  │  • Predictor (Inference Engine)                          │  │
│  │  • Repositories (Data, Model Persistence)                │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       DOMAIN LAYER                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Business Entities & Rules:                               │  │
│  │  • DataSource      • ProcessedData                        │  │
│  │  • EDAReport       • ModelConfig                          │  │
│  │  • TrainedModel    • Prediction                           │  │
│  │                                                            │  │
│  │  Repository Interfaces (Contracts)                        │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 End-to-End Pipeline Flow (Detailed)

```
START
  │
  ├─► [1] DATA PREPARATION
  │    │
  │    ├─ Identify data source (CSV/TXT/PDF/Scanned PDF)
  │    ├─ Create DataSource entity
  │    └─ Configure model settings (ModelConfig)
  │
  ├─► [2] DATA INGESTION
  │    │
  │    ├─ Select appropriate reader
  │    │   ├─ CSVDataReader
  │    │   ├─ TextDataReader
  │    │   ├─ PDFDataReader
  │    │   └─ ScannedPDFDataReader (with OCR)
  │    │
  │    ├─ Read data into DataFrame
  │    │
  │    ├─ DATA CLEANING
  │    │   ├─ Handle missing values
  │    │   │   ├─ Numeric: Fill with median
  │    │   │   └─ Categorical: Fill with mode
  │    │   ├─ Remove duplicates
  │    │   └─ Handle outliers (optional)
  │    │
  │    ├─ DATA TRANSFORMATION
  │    │   ├─ Identify column types
  │    │   │   ├─ Numeric columns
  │    │   │   ├─ Categorical columns
  │    │   │   └─ Datetime columns
  │    │   │
  │    │   ├─ Encode categorical variables
  │    │   │   └─ Label Encoding (A→0, B→1, C→2)
  │    │   │
  │    │   ├─ Scale numeric features (optional)
  │    │   └─ Parse datetime columns
  │    │
  │    ├─ DATA VALIDATION
  │    │   ├─ Completeness check (missing values %)
  │    │   ├─ Consistency check (data types)
  │    │   ├─ Validity check (value ranges)
  │    │   └─ Generate quality score
  │    │
  │    └─ Create ProcessedData entity
  │
  ├─► [3] EXPLORATORY DATA ANALYSIS (EDA)
  │    │
  │    ├─ Statistical Analysis
  │    │   ├─ Descriptive statistics
  │    │   │   ├─ Mean, median, std dev
  │    │   │   ├─ Min, max, quartiles
  │    │   │   └─ Count, unique values
  │    │   │
  │    │   ├─ Correlation analysis
  │    │   │   └─ Feature correlations
  │    │   │
  │    │   ├─ Outlier detection
  │    │   │   └─ IQR method
  │    │   │
  │    │   └─ Distribution analysis
  │    │
  │    ├─ Visualizations
  │    │   ├─ Distribution plots
  │    │   │   └─ Histograms for all numeric features
  │    │   │
  │    │   ├─ Correlation heatmap
  │    │   │   └─ Feature correlation matrix
  │    │   │
  │    │   └─ Outlier boxplots
  │    │       └─ Boxplots for numeric features
  │    │
  │    ├─ Generate insights
  │    │   ├─ Dataset size and shape
  │    │   ├─ Outlier counts per feature
  │    │   └─ Key patterns detected
  │    │
  │    └─ Create EDAReport entity
  │
  ├─► [4] MODEL TRAINING
  │    │
  │    ├─ Prepare training data
  │    │   ├─ Select features (X)
  │    │   ├─ Extract target (y)
  │    │   └─ Train/test split (80/20)
  │    │
  │    ├─ Select ML model
  │    │   ├─ Linear Regression (regression)
  │    │   ├─ Logistic Regression (classification)
  │    │   ├─ Decision Tree (classification/regression)
  │    │   ├─ Random Forest (classification/regression)
  │    │   └─ Gradient Boosting (classification/regression)
  │    │
  │    ├─ Train model
  │    │   ├─ Fit model on training data
  │    │   └─ Apply hyperparameters
  │    │
  │    ├─ Evaluate model
  │    │   ├─ Make predictions on test set
  │    │   ├─ Calculate metrics
  │    │   │   ├─ Classification: accuracy, precision, recall, F1
  │    │   │   └─ Regression: MSE, RMSE, MAE, R²
  │    │   │
  │    │   └─ Extract feature importance (if available)
  │    │
  │    ├─ Save model
  │    │   └─ Pickle to .pkl file
  │    │
  │    └─ Create TrainedModel entity
  │
  ├─► [5] PREDICTION
  │    │
  │    ├─ Load trained model from disk
  │    │
  │    ├─ Prepare input data
  │    │   ├─ Select same features as training
  │    │   ├─ Handle missing values (fill with 0)
  │    │   └─ Encode categorical variables
  │    │
  │    ├─ Make predictions
  │    │   ├─ Model.predict(X)
  │    │   └─ Get confidence scores (if classifier)
  │    │       └─ Model.predict_proba(X)
  │    │
  │    ├─ Post-process results
  │    │   ├─ Attach predictions to original data
  │    │   ├─ Add confidence scores
  │    │   └─ Calculate accuracy (if labels available)
  │    │
  │    ├─ Save predictions
  │    │   └─ Export to CSV
  │    │
  │    └─ Create Prediction entity
  │
  └─► END
       │
       └─ Return results to user
```

---

## 📁 Data Flow Through System

```
┌─────────────────┐
│   Raw Data      │
│  (CSV/TXT/PDF)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Data Reader    │ ──► Factory Pattern
│   (Interface)   │     Selects appropriate reader
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   DataFrame     │
│  (Raw Data)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Processor  │
│   - Clean       │ ──► Handle missing, duplicates
│   - Transform   │ ──► Encode, scale, parse
│   - Validate    │ ──► Quality checks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ProcessedData   │ ──► Entity with metadata
│   DataFrame +   │     Processing steps
│   Metadata      │     Quality metrics
└────────┬────────┘
         │
         ├────────────────────┬────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  EDA Analyzer   │  │ Model Trainer   │  │   Repository    │
│  - Statistics   │  │  - Split data   │  │   - Save data   │
│  - Visuals      │  │  - Train model  │  │   - Load data   │
│  - Insights     │  │  - Evaluate     │  │                 │
└────────┬────────┘  └────────┬────────┘  └─────────────────┘
         │                    │
         ▼                    ▼
┌─────────────────┐  ┌─────────────────┐
│   EDAReport     │  │  TrainedModel   │
│  - Insights     │  │   - Model obj   │
│  - Statistics   │  │   - Metrics     │
│  - Visuals path │  │   - Features    │
└─────────────────┘  └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │    Predictor    │
                     │  - Load model   │
                     │  - Predict      │
                     │  - Confidence   │
                     └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │   Prediction    │
                     │  - Predictions  │
                     │  - Confidence   │
                     │  - Metadata     │
                     └─────────────────┘
```

---

## 🎯 Use Case Execution Flow

### 1. **Data Ingestion Use Case**
```
execute(source, clean=True, transform=True)
  │
  ├─► Get reader from factory
  │    └─► reader = factory.get_reader(source.source_type)
  │
  ├─► Read data
  │    └─► raw_data = reader.read(source)
  │
  ├─► Clean data (if clean=True)
  │    ├─► Handle missing values
  │    ├─► Remove duplicates
  │    └─► Log cleaning stats
  │
  ├─► Transform data (if transform=True)
  │    ├─► Encode categoricals
  │    ├─► Scale numerics
  │    └─► Parse datetimes
  │
  ├─► Validate data
  │    ├─► Check completeness
  │    ├─► Check consistency
  │    └─► Calculate quality score
  │
  └─► Return ProcessedData entity
```

### 2. **EDA Use Case**
```
execute(data, generate_plots=True, output_dir=None)
  │
  ├─► Analyze data
  │    ├─► Calculate statistics
  │    ├─► Find correlations
  │    ├─► Detect outliers
  │    └─► Generate insights
  │
  ├─► Generate visualizations (if generate_plots=True)
  │    ├─► Distribution plots
  │    ├─► Correlation heatmap
  │    ├─► Outlier boxplots
  │    └─► Save to output_dir
  │
  └─► Return EDAReport entity
```

### 3. **Model Training Use Case**
```
execute(data, config, save_model=True, model_path=None)
  │
  ├─► Train model
  │    ├─► Prepare data (X, y split)
  │    ├─► Train/test split
  │    ├─► Fit model
  │    └─► Evaluate on test set
  │
  ├─► Calculate metrics
  │    ├─► Accuracy, precision, recall (classification)
  │    └─► MSE, RMSE, R² (regression)
  │
  ├─► Extract feature importance
  │
  ├─► Save model (if save_model=True)
  │    └─► repository.save(model, model_path)
  │
  └─► Return TrainedModel entity
```

### 4. **Prediction Use Case**
```
execute(data, model_path)
  │
  ├─► Load model
  │    └─► model = repository.load(model_path)
  │
  ├─► Prepare features
  │    ├─► Select same features as training
  │    ├─► Handle missing values
  │    └─► Encode categoricals
  │
  ├─► Make predictions
  │    ├─► predictions = model.predict(X)
  │    └─► confidence = model.predict_proba(X) [if available]
  │
  └─► Return Prediction entity
```

### 5. **ML Pipeline Use Case** (End-to-End)
```
execute(source, model_config, perform_eda=True, eda_output_dir, model_output_path)
  │
  ├─► [Step 1] Data Ingestion
  │    └─► processed_data = data_ingestion_use_case.execute(source)
  │
  ├─► [Step 2] EDA (if perform_eda=True)
  │    └─► eda_report = eda_use_case.execute(processed_data, output_dir)
  │
  ├─► [Step 3] Model Training
  │    └─► trained_model = training_use_case.execute(processed_data, config)
  │
  ├─► [Step 4] Prediction
  │    └─► predictions = prediction_use_case.execute(data, model_path)
  │
  └─► Return complete results dictionary
       {
         'processed_data': ProcessedData,
         'eda_report': EDAReport,
         'trained_model': TrainedModel,
         'predictions': Prediction
       }
```

---

## 🔧 Component Interaction Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                     CLI Application                          │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                   Container (DI)                             │
│  Creates and injects all dependencies                        │
└─┬──────────┬──────────┬──────────┬──────────┬────────────────┘
  │          │          │          │          │
  ▼          ▼          ▼          ▼          ▼
┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐
│ UC1 │  │ UC2 │  │ UC3 │  │ UC4 │  │ UC5 │
└──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘
   │        │        │        │        │
   └────────┴────────┴────────┴────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│           Infrastructure Components                          │
│                                                              │
│  ┌─────────┐  ┌──────────┐  ┌────────┐  ┌──────────┐      │
│  │ Readers │  │Processor │  │Analyzer│  │  Trainer │       │
│  └─────────┘  └──────────┘  └────────┘  └──────────┘       │
│                                                              │
│  ┌─────────┐  ┌──────────┐  ┌────────┐                     │
│  │Predictor│  │Repository│  │ Logger │                      │
│  └─────────┘  └──────────┘  └────────┘                      │
└──────────────────────────────────────────────────────────────┘
```

---

## 🎨 Supported ML Models

| Model | Type | Use Case | Key Parameters |
|-------|------|----------|----------------|
| **Linear Regression** | Regression | Continuous prediction | - |
| **Logistic Regression** | Classification | Binary/Multi-class | max_iter, solver, C |
| **Decision Tree** | Both | Interpretable model | max_depth, min_samples_split |
| **Random Forest** | Both | Ensemble, robust | n_estimators, max_depth |
| **Gradient Boosting** | Both | High performance | learning_rate, n_estimators |

---

## 📦 File Organization

```
ML_Ollama/
├── src/
│   ├── domain/              # Business logic core
│   │   ├── entities.py      # Business entities
│   │   ├── repositories.py  # Interface contracts
│   │   └── value_objects.py # Immutable objects
│   │
│   ├── application/         # Use cases
│   │   └── use_cases/
│   │       ├── data_ingestion.py
│   │       ├── eda.py
│   │       ├── model_training.py
│   │       ├── prediction.py
│   │       └── ml_pipeline.py
│   │
│   ├── infrastructure/      # Technical implementations
│   │   ├── data_readers/    # CSV, TXT, PDF readers
│   │   ├── processing/      # Data processor, EDA
│   │   ├── ml/              # Models, predictor
│   │   ├── persistence/     # Data repository
│   │   └── config/          # Settings, DI container
│   │
│   └── presentation/        # User interfaces
│       └── cli.py           # Command-line interface
│
├── models/                  # Saved trained models (.pkl)
├── data/
│   ├── raw/                # Original data files
│   └── processed/          # Cleaned data files
├── outputs/
│   ├── eda/                # EDA visualizations
│   └── predictions/        # Prediction results
│
├── scripts/                # Utility scripts
├── examples/               # Example usage scripts
└── full_pipeline_*.py      # Complete pipeline scripts
```

---

## 🚀 Execution Modes

### **Mode 1: CLI Command**
```bash
ml-pipeline run-pipeline data.csv --target-column price --model-type random_forest
```

### **Mode 2: Python Script**
```python
from src.infrastructure.config.container import Container
pipeline = container.ml_pipeline_use_case
results = pipeline.execute(source, config)
```

### **Mode 3: Full Pipeline Script**
```bash
python full_pipeline_random_forest.py
```

---

## 🔍 Key Design Patterns Used

1. **Dependency Injection**: Container manages all dependencies
2. **Factory Pattern**: Data reader selection based on file type
3. **Repository Pattern**: Data and model persistence abstraction
4. **Strategy Pattern**: Different ML models, different readers
5. **Use Case Pattern**: Business logic orchestration
6. **Entity Pattern**: Rich domain models

---

## 📈 Quality Assurance

- **Data Quality Metrics**: Completeness, consistency, validity scores
- **Model Metrics**: Accuracy, precision, recall, F1, MSE, R²
- **Feature Importance**: Understand model decisions
- **Logging**: Comprehensive logging at all levels
- **Validation**: Data quality checks at each step

---

This architecture ensures:
✅ **Separation of Concerns**: Each layer has single responsibility
✅ **Testability**: Easy to unit test each component
✅ **Maintainability**: Changes in one layer don't affect others
✅ **Scalability**: Easy to add new models, readers, or features
✅ **Extensibility**: Plugin new components without breaking existing code
