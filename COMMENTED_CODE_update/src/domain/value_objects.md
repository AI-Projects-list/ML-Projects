# value_objects.py - Complete Line-by-Line Documentation

**Source**: `src/domain/value_objects.py`  
**Purpose**: Immutable value objects for the domain  
**Layer**: Domain  
**Lines**: 60  
**Patterns**: Value Object Pattern, Frozen DataClass

---

## Complete Code with Comments

```python
"""Value objects for the domain layer."""
# WHAT: Module defining immutable value objects
# WHY: Represent domain concepts without identity
# PATTERN: Value Object (DDD)
# CHARACTERISTIC: Defined by values, not identity
# IMMUTABILITY: frozen=True makes instances immutable

from dataclasses import dataclass
# WHAT: Python dataclass for automatic methods
# WHY: Less boilerplate than manual classes
# KEY FEATURE: frozen=True for immutability

from typing import Any, Dict, List
# WHAT: Type hints
# WHY: Type safety and documentation

@dataclass(frozen=True)
class ColumnSchema:
    """Represents a column schema definition."""
    # WHAT: Immutable schema for a single column
    # WHY: Data validation, type checking
    # FROZEN: Cannot modify after creation
    # BENEFIT: Thread-safe, hashable, cacheable
    
    name: str
    # WHAT: Column name
    # WHY: Identify column
    # REQUIRED: Yes
    
    dtype: str
    # WHAT: Data type (int64, float64, object, etc.)
    # WHY: Type validation and casting
    # EXAMPLES: "int64", "float64", "object", "datetime64"
    
    nullable: bool = True
    # WHAT: Whether column allows null values
    # WHY: Data validation
    # DEFAULT: True (permissive)
    # USE CASE: Validate data quality
    
    constraints: Dict[str, Any] = None
    # WHAT: Validation constraints
    # WHY: Advanced validation rules
    # DEFAULT: None (no constraints)
    # EXAMPLES: {"min": 0, "max": 100}, {"pattern": "^[A-Z]+$"}
    # NOTE: Dict is mutable - violation of frozen principle
    
    def __post_init__(self) -> None:
        """Validate the column schema."""
        # WHAT: Post-initialization validation
        # WHY: Set default for mutable field
        # HOW: object.__setattr__ to bypass frozen
        # PROBLEM: Mutable dict in frozen dataclass
        
        if self.constraints is None:
            object.__setattr__(self, 'constraints', {})
        # WHAT: Set empty dict if None
        # WHY: Avoid None checks everywhere
        # HOW: Use object.__setattr__ (bypass frozen)
        # TRADE-OFF: Hacky, better to use field(default_factory)


@dataclass(frozen=True)
class DataQualityMetrics:
    """Represents data quality metrics."""
    # WHAT: Immutable quality metrics
    # WHY: Assess data fitness for ML
    # FROZEN: Immutable value object
    # BENEFIT: Can be cached, compared
    
    completeness: float  # 0-1 score
    # WHAT: Ratio of non-missing values
    # WHY: Measure data completeness
    # RANGE: 0.0 (all missing) to 1.0 (no missing)
    # CALCULATION: (total_cells - missing_cells) / total_cells
    
    consistency: float  # 0-1 score
    # WHAT: Ratio of non-duplicate rows
    # WHY: Measure data consistency
    # RANGE: 0.0 to 1.0
    # CALCULATION: (total_rows - duplicate_rows) / total_rows
    
    validity: float  # 0-1 score
    # WHAT: Ratio of valid values
    # WHY: Measure data correctness
    # RANGE: 0.0 to 1.0
    # CHECKS: Type correctness, range validation
    
    total_rows: int
    # WHAT: Total number of rows
    # WHY: Context for other metrics
    
    total_columns: int
    # WHAT: Total number of columns
    # WHY: Context for other metrics
    
    missing_cells: int
    # WHAT: Count of missing values
    # WHY: Raw missing data count
    # USE WITH: completeness
    
    duplicate_rows: int
    # WHAT: Count of duplicate rows
    # WHY: Raw duplicate count
    # USE WITH: consistency
    
    @property
    def overall_quality(self) -> float:
        """Calculate overall quality score."""
        # WHAT: Computed property for overall score
        # WHY: Single quality metric
        # HOW: Average of three dimensions
        # BENEFIT: Easy comparison between datasets
        
        return (self.completeness + self.consistency + self.validity) / 3
        # WHAT: Simple average
        # WHY: Equal weight to all dimensions
        # TRADE-OFF: Could use weighted average
        # RANGE: 0.0 to 1.0
    
    def is_acceptable(self, threshold: float = 0.7) -> bool:
        """Check if data quality meets the threshold."""
        # WHAT: Quality threshold check
        # WHY: Binary decision (accept/reject)
        # DEFAULT: 0.7 = 70% quality required
        # RETURN: Boolean
        # USE CASE: Validation step in pipeline
        
        return self.overall_quality >= threshold
        # WHAT: Compare to threshold
        # WHY: Make accept/reject decision
        # BENEFIT: Clear pass/fail criteria


@dataclass(frozen=True)
class FeatureEngineering:
    """Represents feature engineering specifications."""
    # WHAT: Immutable feature engineering config
    # WHY: Document feature transformations
    # FROZEN: Cannot modify after creation
    # BENEFIT: Reproducible feature engineering
    
    numerical_features: List[str]
    # WHAT: List of numeric feature names
    # WHY: Identify features for scaling
    # EXAMPLES: ["age", "income", "score"]
    # USE CASE: Apply StandardScaler to these
    
    categorical_features: List[str]
    # WHAT: List of categorical feature names
    # WHY: Identify features for encoding
    # EXAMPLES: ["gender", "category", "city"]
    # USE CASE: Apply LabelEncoder/OneHotEncoder
    
    datetime_features: List[str]
    # WHAT: List of datetime feature names
    # WHY: Identify features for time extraction
    # EXAMPLES: ["created_at", "transaction_date"]
    # USE CASE: Extract year, month, day, hour, etc.
    
    derived_features: Dict[str, str]  # feature_name: formula/description
    # WHAT: Mapping of derived feature names to formulas
    # WHY: Document feature engineering logic
    # EXAMPLES: {"age_squared": "age ** 2", "income_per_age": "income / age"}
    # BENEFIT: Self-documenting transformations
    # TRADE-OFF: Formula is string (not executable)
    
    @property
    def all_features(self) -> List[str]:
        """Get all feature names."""
        # WHAT: Computed property for all features
        # WHY: Get complete feature list
        # HOW: Concatenate all feature lists
        # BENEFIT: Single list of all features
        
        return (
            self.numerical_features
            + self.categorical_features
            + self.datetime_features
            + list(self.derived_features.keys())
        )
        # WHAT: Combine all feature lists
        # WHY: Complete feature inventory
        # RETURN: Flat list of all feature names
        # USE CASE: Feature selection, column subsetting
```

---

## Design Patterns

### **Value Object Pattern** (DDD)
- **WHAT**: Objects defined by values, not identity
- **CHARACTERISTIC**: Two objects equal if values equal
- **IMMUTABILITY**: frozen=True
- **BENEFIT**: Thread-safe, hashable, comparable

---

## Pros & Cons

### ✅ Pros

1. **Immutability**: frozen=True prevents accidental changes
2. **Type Safety**: Type hints for all fields
3. **Computed Properties**: overall_quality, all_features
4. **Value Semantics**: Equality by value not identity
5. **Hashable**: Can be dict keys, set members

### ❌ Cons

1. **Mutable Fields**: Dict and List are mutable
   - **Issue**: Can modify `constraints`, `numerical_features` after creation
   - **Fix**: Use tuple instead of List, frozenset instead of Dict
2. **__post_init__ Hack**: Uses object.__setattr__ to bypass frozen
   - **Better**: Use `field(default_factory=dict)`
3. **No Validation**: No range checks (completeness 0-1)
   - **Fix**: Add __post_init__ validation

---

## Improvements

```python
from typing import Tuple, FrozenSet

@dataclass(frozen=True)
class FeatureEngineering:
    numerical_features: Tuple[str, ...]  # Immutable
    categorical_features: Tuple[str, ...]
    datetime_features: Tuple[str, ...]
    derived_features: FrozenSet[str]  # Truly immutable

@dataclass(frozen=True)
class ColumnSchema:
    name: str
    dtype: str
    nullable: bool = True
    constraints: Dict[str, Any] = field(default_factory=dict)  # Better default
```

---

**Lines**: 60  
**Classes**: 3  
**All Frozen**: Yes  
**Complexity**: Low
