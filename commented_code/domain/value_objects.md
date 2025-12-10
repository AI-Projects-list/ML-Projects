# Domain Value Objects - Detailed Code Documentation

**File**: `src/domain/value_objects.py`  
**Purpose**: Define immutable value objects for the domain layer  
**Layer**: Domain (Core Business Logic)  
**Dependencies**: None (Pure domain logic)

---

## Overview

Value Objects are **immutable objects without identity** that describe characteristics of things. Unlike entities which have identity and lifecycle, value objects are interchangeable if their values are the same.

**Key Principle**: Two value objects with the same values are considered equal, regardless of when/where they were created.

---

## Complete Code with Line-by-Line Comments

```python
"""Value objects for the domain layer."""
# Module docstring
# WHAT: This module contains immutable value objects
# WHY: Separate value objects from entities for cleaner design
# PATTERN: Value Object pattern from Domain-Driven Design (DDD)

from dataclasses import dataclass
# WHAT: Import dataclass decorator
# WHY: Create immutable data structures easily
# HOW: Use frozen=True parameter
# BENEFIT: Reduces boilerplate, automatic __eq__, __hash__

from typing import Any, Dict, List
# WHAT: Import type hints
# WHY: Type safety and documentation
# HOW: Annotate all parameters and returns
# BENEFIT: Catch errors early, better IDE support


@dataclass(frozen=True)
class ColumnSchema:
    """Represents a column schema definition."""
    # WHAT: Value object for column metadata/schema
    # WHY: Define expected structure of data columns
    # HOW: Immutable dataclass with validation
    # BENEFIT: Type-safe schema definitions
    # PATTERN: Value Object (immutable, no identity)
    
    # WHY frozen=True?
    # - Makes object immutable (cannot change after creation)
    # - Allows use as dictionary key
    # - Thread-safe
    # - Prevents accidental modifications
    
    name: str
    # WHAT: Column name
    # WHY: Identify the column
    # REQUIRED: Yes (no default)
    # VALIDATION: Should not be empty
    # USE CASE: "age", "income", "price"
    
    dtype: str
    # WHAT: Data type of the column
    # WHY: Specify expected type
    # HOW: String representation of type
    # TRADE-OFF: String not type-safe
    # IMPROVEMENT: Use Enum for common types
    # USE CASE: "int64", "float64", "object", "datetime64"
    
    nullable: bool = True
    # WHAT: Whether null values are allowed
    # WHY: Schema validation rule
    # DEFAULT: True (permissive by default)
    # BENEFIT: Explicit nullability contract
    # USE CASE: False for required fields
    
    constraints: Dict[str, Any] = None
    # WHAT: Additional validation constraints
    # WHY: Define value constraints
    # HOW: Dictionary of constraint name → value
    # DEFAULT: None (no constraints)
    # TRADE-OFF: Mutable dict in frozen dataclass
    # USE CASE: {"min": 0, "max": 100, "pattern": "^\d+$"}
    
    def __post_init__(self) -> None:
        """Validate the column schema."""
        # WHAT: Post-initialization hook
        # WHY: Perform validation after object creation
        # WHEN: Called automatically after __init__
        # HOW: Use object.__setattr__ to modify frozen object
        # BENEFIT: Validation at creation time
        
        if self.constraints is None:
            # WHAT: Check if constraints is None
            # WHY: Avoid shared mutable default
            # HOW: Set empty dict if None
            
            object.__setattr__(self, 'constraints', {})
            # WHAT: Set constraints to empty dict
            # WHY: frozen=True prevents self.constraints = {}
            # HOW: Use object.__setattr__ to bypass frozen
            # TRADE-OFF: Breaks immutability contract slightly
            # REASON: Necessary to avoid mutable default pitfall
            
            # WHY NOT use field(default_factory=dict)?
            # - Would make constraints mutable
            # - frozen=True with mutable fields is problematic
            # BETTER APPROACH: Use tuple of tuples instead

# IMPROVEMENT SUGGESTION:
# Use immutable tuple for constraints:
# constraints: Tuple[Tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class DataQualityMetrics:
    """Represents data quality metrics."""
    # WHAT: Value object for data quality scores
    # WHY: Quantify data quality
    # HOW: Immutable snapshot of quality metrics
    # BENEFIT: Can compare quality over time
    # PATTERN: Value Object + Calculation Object
    
    # WHY frozen=True?
    # - Quality metrics shouldn't change after calculation
    # - Can be used as dictionary key
    # - Thread-safe for parallel processing
    
    completeness: float  # 0-1 score
    # WHAT: Ratio of non-missing values
    # WHY: Measure data completeness
    # RANGE: 0.0 (all missing) to 1.0 (none missing)
    # CALCULATION: 1 - (missing_cells / total_cells)
    # TRADE-OFF: No range validation
    # IMPROVEMENT: Add __post_init__ validation
    # USE CASE: 0.95 = 95% complete
    
    consistency: float  # 0-1 score
    # WHAT: Ratio of consistent/unique data
    # WHY: Measure data consistency
    # RANGE: 0.0 (all duplicates) to 1.0 (all unique)
    # CALCULATION: 1 - (duplicate_rows / total_rows)
    # BENEFIT: Detect data duplication issues
    # USE CASE: 0.98 = 98% unique rows
    
    validity: float  # 0-1 score
    # WHAT: Ratio of valid data
    # WHY: Measure type/constraint compliance
    # RANGE: 0.0 (all invalid) to 1.0 (all valid)
    # CALCULATION: valid_columns / total_columns
    # BENEFIT: Type consistency check
    # USE CASE: 1.0 = all data meets type requirements
    
    total_rows: int
    # WHAT: Number of rows in dataset
    # WHY: Context for other metrics
    # BENEFIT: Understand dataset size
    # TRADE-OFF: No validation (should be >= 0)
    # USE CASE: 1000 rows
    
    total_columns: int
    # WHAT: Number of columns in dataset
    # WHY: Dataset shape information
    # BENEFIT: Schema size
    # USE CASE: 50 columns
    
    missing_cells: int
    # WHAT: Count of missing values
    # WHY: Absolute missing value count
    # BENEFIT: Complements completeness ratio
    # CALCULATION: data.isnull().sum().sum()
    # USE CASE: 50 missing cells out of 50,000 total
    
    duplicate_rows: int
    # WHAT: Count of duplicate rows
    # WHY: Absolute duplicate count
    # BENEFIT: Data quality indicator
    # CALCULATION: data.duplicated().sum()
    # USE CASE: 20 duplicate rows
    
    @property
    def overall_quality(self) -> float:
        """Calculate overall quality score."""
        # WHAT: Computed property for overall quality
        # WHY: Single metric combining all dimensions
        # HOW: Average of three quality dimensions
        # BENEFIT: DRY - calculated not stored
        # PATTERN: Calculated property
        
        # @property decorator:
        # - Makes method callable as attribute (no parentheses)
        # - Read-only (no setter)
        # - Calculated on-demand
        # BENEFIT: Always up-to-date, no storage overhead
        
        return (self.completeness + self.consistency + self.validity) / 3
        # WHAT: Simple average of three metrics
        # WHY: Combine dimensions into single score
        # HOW: Sum three metrics, divide by 3
        # RANGE: 0.0 to 1.0
        # TRADE-OFF: Equal weighting (may not be appropriate)
        
        # IMPROVEMENT: Weighted average
        # weights = {"completeness": 0.4, "consistency": 0.3, "validity": 0.3}
        # return (self.completeness * 0.4 + self.consistency * 0.3 + 
        #         self.validity * 0.3)
        
        # USE CASE:
        # If completeness=0.9, consistency=0.95, validity=1.0
        # overall_quality = (0.9 + 0.95 + 1.0) / 3 = 0.95
    
    def is_acceptable(self, threshold: float = 0.7) -> bool:
        """Check if data quality meets the threshold."""
        # WHAT: Quality gate method
        # WHY: Business logic for quality decision
        # HOW: Compare overall_quality to threshold
        # BENEFIT: Encapsulates business rule
        # PATTERN: Business logic in domain model
        
        # Parameters:
        # threshold: Minimum acceptable quality (0-1)
        # DEFAULT: 0.7 (70% quality required)
        # BENEFIT: Configurable per use case
        
        return self.overall_quality >= threshold
        # WHAT: Boolean comparison
        # WHY: Pass/fail decision
        # HOW: >= allows threshold to be inclusive
        # RETURN: True if acceptable, False if not
        
        # USE CASE:
        # metrics.is_acceptable(0.8)  # Require 80% quality
        # metrics.is_acceptable()      # Use default 70%
        
        # TRADE-OFF:
        # - Only checks overall, not individual metrics
        # - A dataset could pass with one very low metric
        # IMPROVEMENT: Check individual metrics too
        # return (self.overall_quality >= threshold and
        #         self.completeness >= threshold * 0.8 and
        #         self.consistency >= threshold * 0.8)


@dataclass(frozen=True)
class FeatureEngineering:
    """Represents feature engineering specifications."""
    # WHAT: Value object for feature metadata
    # WHY: Document feature types and transformations
    # HOW: Categorize features by type
    # BENEFIT: Explicit feature categorization
    # PATTERN: Specification Object
    
    # WHY frozen=True?
    # - Feature specs shouldn't change during processing
    # - Can be shared across pipeline steps
    # - Thread-safe for parallel feature engineering
    
    numerical_features: List[str]
    # WHAT: List of numeric column names
    # WHY: Identify numeric features for scaling/normalization
    # HOW: Column names as strings
    # TRADE-OFF: Mutable list in frozen dataclass
    # IMPROVEMENT: Use Tuple[str, ...] instead
    # USE CASE: ["age", "income", "balance", "score"]
    
    # WHY mutable list is problematic:
    # numerical_features.append("new")  # This would work!
    # - Breaks immutability guarantee
    # - Can cause unexpected bugs
    # BETTER: numerical_features: Tuple[str, ...]
    
    categorical_features: List[str]
    # WHAT: List of categorical column names
    # WHY: Identify features needing encoding
    # HOW: Column names as strings
    # TRADE-OFF: Mutable list
    # USE CASE: ["gender", "country", "product_type"]
    
    # ENCODING STRATEGIES:
    # - Label Encoding: Map to integers (0, 1, 2, ...)
    # - One-Hot Encoding: Binary columns per category
    # - Target Encoding: Map to target mean
    
    datetime_features: List[str]
    # WHAT: List of datetime column names
    # WHY: Identify temporal features
    # HOW: Column names as strings
    # TRADE-OFF: Mutable list
    # USE CASE: ["created_at", "updated_at", "birth_date"]
    
    # DATETIME ENGINEERING:
    # - Extract: year, month, day, hour, day_of_week
    # - Calculate: age, days_since, is_weekend
    # - Cyclical encoding: sin/cos for periodic features
    
    derived_features: Dict[str, str]  # feature_name: formula/description
    # WHAT: Computed/derived features
    # WHY: Document feature transformations
    # HOW: Feature name → formula/description mapping
    # TRADE-OFF: Mutable dict, string formula (not executable)
    # USE CASE: {"age": "current_year - birth_year",
    #           "bmi": "weight / (height ** 2)"}
    
    # WHY string formula?
    # - Human-readable documentation
    # - Simple to store/serialize
    # - No code injection risk
    # TRADE-OFF: Not executable, just documentation
    
    # IMPROVEMENT: Use AST or lambda
    # derived_features: Dict[str, Callable]
    
    @property
    def all_features(self) -> List[str]:
        """Get all feature names."""
        # WHAT: Computed property returning all features
        # WHY: Convenient access to complete feature set
        # HOW: Concatenate all feature lists
        # BENEFIT: Single source of truth
        # PATTERN: Calculated property
        
        return (
            self.numerical_features
            + self.categorical_features
            + self.datetime_features
            + list(self.derived_features.keys())
        )
        # WHAT: Concatenate all feature lists
        # WHY: Get complete feature inventory
        # HOW: List addition + dict keys
        # TRADE-OFF: Creates new list each time
        # IMPROVEMENT: Cache result with @lru_cache
        
        # ORDER MATTERS:
        # - Numerical first
        # - Then categorical
        # - Then datetime
        # - Finally derived
        # USE CASE: Feature selection, model input
        
        # TRADE-OFF:
        # - Creates new list (memory allocation)
        # - Called frequently? Could cache
        # - But frozen object guarantees result is same
        
        # MEMORY:
        # If numerical_features = 100 items
        #    categorical_features = 50 items
        #    datetime_features = 10 items
        #    derived_features = 20 items
        # Total = 180 feature names each call

```

---

## Design Patterns Analysis

### 1. **Value Object Pattern** (Core)
**What**: Immutable objects without identity  
**Why**: Model concepts that are defined by their values  
**Benefit**: 
- Thread-safe
- Can be dictionary keys
- No side effects
- Easier to reason about

**Examples**:
- `ColumnSchema`: Two schemas with same values are equal
- `DataQualityMetrics`: Quality at a point in time
- `FeatureEngineering`: Feature specification

### 2. **Specification Pattern**
**What**: Encapsulate business rules  
**Why**: Separate logic from data  
**Example**: `is_acceptable()` method

### 3. **Calculation Object Pattern**
**What**: Object that performs calculations  
**Why**: Encapsulate complex calculations  
**Example**: `overall_quality` property

---

## Immutability Analysis

### ✅ **Properly Immutable**
```python
@dataclass(frozen=True)
class Example:
    value: int  # Cannot change after creation
```

### ⚠️ **Partially Immutable** (Current Implementation)
```python
@dataclass(frozen=True)
class FeatureEngineering:
    numerical_features: List[str]  # List is mutable!
```

**Problem**:
```python
features = FeatureEngineering(
    numerical_features=["age"],
    categorical_features=[],
    datetime_features=[],
    derived_features={}
)
features.numerical_features.append("income")  # THIS WORKS!
```

### ✅ **Fully Immutable** (Recommended)
```python
@dataclass(frozen=True)
class FeatureEngineering:
    numerical_features: Tuple[str, ...]  # Tuple is immutable
    categorical_features: Tuple[str, ...]
    datetime_features: Tuple[str, ...]
    derived_features: Tuple[Tuple[str, str], ...]  # Tuple of tuples
```

---

## Key Benefits

✅ **Immutability**: `frozen=True` prevents modifications  
✅ **Type Safety**: Clear type hints  
✅ **Calculated Properties**: DRY principle  
✅ **Business Logic**: Domain logic in domain models  
✅ **Equality**: Automatic `__eq__` based on values  
✅ **Hashing**: Can be used as dict keys

---

## Areas for Improvement

### High Priority
1. ⚠️ **Use Tuples Instead of Lists**
```python
numerical_features: Tuple[str, ...] = ()
```

2. ⚠️ **Add Validation**
```python
def __post_init__(self):
    if not 0 <= self.completeness <= 1:
        raise ValueError("Completeness must be 0-1")
```

3. ⚠️ **Make constraints Immutable**
```python
constraints: Tuple[Tuple[str, Any], ...] = ()
```

### Medium Priority
4. ⚠️ **Use Enum for dtypes**
```python
class DataType(Enum):
    INT64 = "int64"
    FLOAT64 = "float64"
    OBJECT = "object"
```

5. ⚠️ **Weighted Quality Score**
```python
def overall_quality(self, weights=(0.4, 0.3, 0.3)):
    return (self.completeness * weights[0] + 
            self.consistency * weights[1] + 
            self.validity * weights[2])
```

---

## Usage Examples

```python
# Column Schema
schema = ColumnSchema(
    name="age",
    dtype="int64",
    nullable=False,
    constraints={"min": 0, "max": 120}
)

# Data Quality Metrics
metrics = DataQualityMetrics(
    completeness=0.95,
    consistency=0.98,
    validity=1.0,
    total_rows=1000,
    total_columns=50,
    missing_cells=250,  # 5% missing
    duplicate_rows=20   # 2% duplicates
)

print(metrics.overall_quality)  # 0.976
print(metrics.is_acceptable())  # True (> 0.7)
print(metrics.is_acceptable(0.98))  # False (< 0.98)

# Feature Engineering
features = FeatureEngineering(
    numerical_features=["age", "income", "balance"],
    categorical_features=["gender", "country"],
    datetime_features=["created_at"],
    derived_features={
        "age": "current_year - birth_year",
        "days_active": "current_date - created_at"
    }
)

print(len(features.all_features))  # 7 total features
```

---

## Testing Considerations

### Test Immutability
```python
def test_immutability():
    metrics = DataQualityMetrics(...)
    with pytest.raises(FrozenInstanceError):
        metrics.completeness = 0.5
```

### Test Equality
```python
def test_equality():
    m1 = DataQualityMetrics(completeness=0.9, ...)
    m2 = DataQualityMetrics(completeness=0.9, ...)
    assert m1 == m2  # Value equality
```

### Test Business Logic
```python
def test_quality_threshold():
    metrics = DataQualityMetrics(completeness=0.8, consistency=0.8, validity=0.8, ...)
    assert metrics.is_acceptable(0.7)
    assert not metrics.is_acceptable(0.9)
```

---

## Dependencies

- `dataclasses`: Standard library (Python 3.7+)
- `typing`: Standard library

---

**Total Lines**: 62  
**Complexity**: Low  
**Maintainability**: High  
**Immutability**: Partial (needs improvement)  
**Test Coverage**: Should be 100%
