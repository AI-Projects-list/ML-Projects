"""Generate sample dataset for testing."""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_classification_dataset(n_samples: int = 1000, output_path: str = "data/raw/sample_classification.csv"):
    """Generate a sample classification dataset."""
    np.random.seed(42)
    
    # Generate features
    data = {
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'feature_3': np.random.randint(1, 10, n_samples),
        'feature_4': np.random.uniform(0, 100, n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generate target based on features
    df['target'] = (
        (df['feature_1'] > 0).astype(int) + 
        (df['feature_2'] > 0).astype(int)
    )
    
    # Add some noise
    noise_indices = np.random.choice(df.index, size=int(0.1 * n_samples), replace=False)
    df.loc[noise_indices, 'target'] = 1 - df.loc[noise_indices, 'target']
    
    # Add some missing values
    missing_indices = np.random.choice(df.index, size=int(0.05 * n_samples), replace=False)
    df.loc[missing_indices, 'feature_1'] = np.nan
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Generated classification dataset: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Target distribution:\n{df['target'].value_counts()}")
    
    return df


def generate_regression_dataset(n_samples: int = 1000, output_path: str = "data/raw/sample_regression.csv"):
    """Generate a sample regression dataset."""
    np.random.seed(42)
    
    # Generate features
    data = {
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'feature_3': np.random.randint(1, 10, n_samples),
        'feature_4': np.random.uniform(0, 100, n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generate target based on features
    df['price'] = (
        50 + 
        10 * df['feature_1'] + 
        15 * df['feature_2'] + 
        5 * df['feature_3'] +
        0.5 * df['feature_4'] +
        np.random.randn(n_samples) * 5  # noise
    )
    
    # Add some missing values
    missing_indices = np.random.choice(df.index, size=int(0.05 * n_samples), replace=False)
    df.loc[missing_indices, 'feature_2'] = np.nan
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Generated regression dataset: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Target statistics:\n{df['price'].describe()}")
    
    return df


def generate_text_dataset(n_samples: int = 100, output_path: str = "data/raw/sample_text.txt"):
    """Generate a sample text dataset."""
    sentences = [
        "Machine learning is transforming the world.",
        "Data science combines statistics and programming.",
        "Python is a popular language for AI development.",
        "Deep learning uses neural networks.",
        "Natural language processing enables text understanding.",
    ]
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(n_samples):
            sentence = np.random.choice(sentences)
            f.write(f"Line {i+1}: {sentence}\n")
    
    print(f"Generated text dataset: {output_path}")
    print(f"Lines: {n_samples}")
    
    return output_path


if __name__ == "__main__":
    print("Generating sample datasets...\n")
    
    # Generate classification dataset
    generate_classification_dataset()
    print()
    
    # Generate regression dataset
    generate_regression_dataset()
    print()
    
    # Generate text dataset
    generate_text_dataset()
    print()
    
    print("All sample datasets generated successfully!")
