# Application Layer - Complete Documentation Index

This folder contains detailed line-by-line documentation for all application layer use cases.

---

## Files Documented

1. **data_ingestion.md** - Data ingestion orchestration
2. **eda.md** - EDA workflow orchestration
3. **model_training.md** - Model training workflow
4. **prediction.md** - Prediction workflow
5. **ml_pipeline.md** - End-to-end pipeline orchestration

---

## Quick Reference

### Data Ingestion Use Case
- **Purpose**: Orchestrate read → clean → transform → validate
- **Pattern**: Template Method with boolean flags
- **Key**: Flexible pipeline execution

### EDA Use Case
- **Purpose**: Orchestrate analysis + visualization generation
- **Pattern**: Simple orchestration
- **Key**: Optional plot generation

### Model Training Use Case
- **Purpose**: Orchestrate train → evaluate → save
- **Pattern**: Simple orchestration with persistence
- **Key**: Optional model saving

### Prediction Use Case
- **Purpose**: Orchestrate model loading + prediction
- **Pattern**: Lazy loading pattern
- **Key**: Accepts model or path

### ML Pipeline Use Case
- **Purpose**: Orchestrate entire end-to-end workflow
- **Pattern**: Facade pattern
- **Key**: Combines all use cases sequentially

---

## Common Patterns

All use cases follow these patterns:
- **Constructor**: Dependency injection
- **execute()**: Standard method name
- **Logging**: Extensive observability
- **Rich Returns**: Domain entities, not primitives
- **Error Handling**: Try-except with logging
