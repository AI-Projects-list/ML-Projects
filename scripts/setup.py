"""Quick setup script to initialize the project."""

import os
import sys
from pathlib import Path


def create_directories():
    """Create necessary directories."""
    directories = [
        "data/raw",
        "data/processed",
        "models",
        "outputs/eda",
        "logs",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def create_env_file():
    """Create .env file from .env.example if it doesn't exist."""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        import shutil
        shutil.copy(env_example, env_file)
        print(f"✓ Created .env file from .env.example")
    else:
        print("✓ .env file already exists")


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("✗ Python 3.9+ is required")
        sys.exit(1)
    else:
        print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")


def main():
    """Main setup function."""
    print("=" * 60)
    print("ML Ollama - Project Setup")
    print("=" * 60)
    print()
    
    # Check Python version
    print("Checking Python version...")
    check_python_version()
    print()
    
    # Create directories
    print("Creating project directories...")
    create_directories()
    print()
    
    # Create .env file
    print("Setting up environment configuration...")
    create_env_file()
    print()
    
    print("=" * 60)
    print("Setup completed successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Install dependencies: poetry install")
    print("2. Generate sample data: poetry run python scripts/generate_sample_data.py")
    print("3. Run example: poetry run python examples/example_csv_pipeline.py")
    print("4. Or use CLI: poetry run ml-pipeline --help")
    print()


if __name__ == "__main__":
    main()
