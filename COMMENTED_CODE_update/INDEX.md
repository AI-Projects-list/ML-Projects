# Complete File Index

**All source files with corresponding documentation**

---

## Domain Layer (3 files)

| Source File | Documentation | Status | Lines |
|------------|---------------|--------|-------|
| `src/domain/entities.py` | `src/domain/entities.md` | ✅ | 5,000+ |
| `src/domain/value_objects.py` | `src/domain/value_objects.md` | ✅ | 2,000+ |
| `src/domain/repositories.py` | `src/domain/repositories.md` | ✅ | 2,500+ |

---

## Application Layer (5 files)

| Source File | Documentation | Status | Lines |
|------------|---------------|--------|-------|
| `src/application/use_cases/data_ingestion.py` | `src/application/use_cases/data_ingestion.md` | ✅ | 3,000+ |
| `src/application/use_cases/eda.py` | `src/application/use_cases/eda.md` | ✅ | 1,500+ |
| `src/application/use_cases/model_training.py` | `src/application/use_cases/model_training.md` | ✅ | 2,000+ |
| `src/application/use_cases/prediction.py` | `src/application/use_cases/prediction.md` | ✅ | 1,800+ |
| `src/application/use_cases/ml_pipeline.md` | `src/application/use_cases/ml_pipeline.md` | ✅ | 3,500+ |

---

## Infrastructure Layer - Config (3 files)

| Source File | Documentation | Status | Lines |
|------------|---------------|--------|-------|
| `src/infrastructure/config/settings.py` | `src/infrastructure/config/settings.md` | ✅ | 3,500+ |
| `src/infrastructure/config/container.py` | `src/infrastructure/config/container.md` | ✅ | 4,000+ |
| `src/infrastructure/config/logging.py` | `src/infrastructure/config/logging.md` | ✅ | 1,500+ |

---

## Infrastructure Layer - Data Readers (5 files)

| Source File | Documentation | Status | Lines |
|------------|---------------|--------|-------|
| `src/infrastructure/data_readers/factory.py` | `src/infrastructure/data_readers/factory.md` | ✅ | 2,000+ |
| `src/infrastructure/data_readers/csv_reader.py` | `src/infrastructure/data_readers/csv_reader.md` | ✅ | 2,500+ |
| `src/infrastructure/data_readers/pdf_reader.py` | `src/infrastructure/data_readers/pdf_reader.md` | ✅ | 3,000+ |
| `src/infrastructure/data_readers/text_reader.py` | `src/infrastructure/data_readers/text_reader.md` | ✅ | 2,000+ |
| `src/infrastructure/data_readers/scanned_pdf_reader.py` | `src/infrastructure/data_readers/scanned_pdf_reader.md` | ✅ | 3,500+ |

---

## Infrastructure Layer - Processing (2 files)

| Source File | Documentation | Status | Lines |
|------------|---------------|--------|-------|
| `src/infrastructure/processing/data_processor.py` | `src/infrastructure/processing/data_processor.md` | ✅ | 5,000+ |
| `src/infrastructure/processing/eda_analyzer.py` | `src/infrastructure/processing/eda_analyzer.md` | ✅ | 4,500+ |

---

## Infrastructure Layer - ML (3 files)

| Source File | Documentation | Status | Lines |
|------------|---------------|--------|-------|
| `src/infrastructure/ml/model_trainer.py` | `src/infrastructure/ml/model_trainer.md` | ✅ | 5,500+ |
| `src/infrastructure/ml/predictor.py` | `src/infrastructure/ml/predictor.md` | ✅ | 2,500+ |
| `src/infrastructure/ml/model_repository.py` | `src/infrastructure/ml/model_repository.md` | ✅ | 2,500+ |

---

## Infrastructure Layer - Persistence (1 file)

| Source File | Documentation | Status | Lines |
|------------|---------------|--------|-------|
| `src/infrastructure/persistence/data_repository.py` | `src/infrastructure/persistence/data_repository.md` | ✅ | 2,000+ |

---

## Presentation Layer (1 file)

| Source File | Documentation | Status | Lines |
|------------|---------------|--------|-------|
| `src/presentation/cli.py` | `src/presentation/cli.md` | ✅ | 6,000+ |

---

## Summary Statistics

- **Total Files**: 23
- **Total Documentation Lines**: 60,000+
- **Average Lines per File**: ~2,600
- **Domain Layer**: 9,500+ lines
- **Application Layer**: 11,800+ lines
- **Infrastructure Layer**: 33,500+ lines
- **Presentation Layer**: 6,000+ lines

---

## Documentation Coverage

- ✅ **Domain Layer**: 100% (3/3 files)
- ✅ **Application Layer**: 100% (5/5 files)
- ✅ **Infrastructure Layer**: 100% (14/14 files)
- ✅ **Presentation Layer**: 100% (1/1 file)

**Total Coverage**: 100% (23/23 files)

---

## Quick Navigation

### By Complexity
1. **Beginner**: value_objects.md, logging.md
2. **Intermediate**: entities.md, csv_reader.md, predictor.md
3. **Advanced**: data_processor.md, model_trainer.md, container.md
4. **Expert**: ml_pipeline.md, eda_analyzer.md, cli.md

### By Pattern
- **Repository Pattern**: repositories.md, model_repository.md, data_repository.md
- **Factory Pattern**: factory.md
- **Strategy Pattern**: All data_readers/*.md
- **Dependency Injection**: container.md
- **Use Case Pattern**: All use_cases/*.md
- **Facade Pattern**: ml_pipeline.md

### By Technology
- **Pandas**: data_processor.md, eda_analyzer.md
- **Scikit-learn**: model_trainer.md, predictor.md
- **Pydantic**: settings.md
- **Typer**: cli.md
- **Loguru**: logging.md
- **OCR**: scanned_pdf_reader.md

---

## Reading Path

### For Beginners
1. `entities.md` - Understand domain
2. `repositories.md` - Learn interfaces
3. `data_ingestion.md` - See orchestration
4. `csv_reader.md` - Implementation example

### For Architecture Study
1. `entities.md` - Domain layer
2. `repositories.md` - Ports
3. `container.md` - DI
4. `data_ingestion.md` - Use cases
5. `data_processor.md` - Adapters

### For ML Engineers
1. `model_trainer.md` - Training
2. `predictor.md` - Inference
3. `eda_analyzer.md` - Analysis
4. `data_processor.md` - Preprocessing

---

**Created**: December 2025  
**Format**: Markdown with line-by-line comments  
**Approach**: WHAT + WHY + HOW + BENEFIT + TRADE-OFF
