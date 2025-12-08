"""Command-line interface for ML pipeline."""

from pathlib import Path
from typing import Optional, Annotated

import typer
from rich.console import Console
from rich.table import Table

from src.domain.entities import DataSource, DataSourceType, ModelConfig
from src.infrastructure.config.container import Container
from src.infrastructure.config.logging import setup_logging
from src.infrastructure.config.settings import get_settings

app = typer.Typer(help="ML Pipeline - End-to-end machine learning pipeline")
console = Console()


@app.command()
def run_pipeline(
    data_path: Annotated[str, typer.Argument(help="Path to input data file")],
    data_type: Annotated[str, typer.Option(help="Data source type (csv, txt, pdf, pdf_scan)")] = "csv",
    target_column: Annotated[str, typer.Option(help="Target column for ML model")] = None,
    model_type: Annotated[str, typer.Option(help="Model type (linear_regression, logistic_regression, decision_tree, random_forest, gradient_boosting)")] = "random_forest",
    test_size: Annotated[float, typer.Option(help="Test set size ratio")] = 0.2,
    perform_eda: Annotated[bool, typer.Option(help="Perform exploratory data analysis")] = True,
    output_dir: Annotated[str, typer.Option(help="Output directory for results")] = "outputs",
):
    """Run the complete end-to-end ML pipeline."""
    console.print("\n[bold blue]ML Pipeline Starting...[/bold blue]\n")
    
    # Validate required parameters
    if not target_column:
        console.print("[red]Error: --target-column is required[/red]")
        raise typer.Exit(1)
    
    # Setup
    settings = get_settings()
    setup_logging(settings)
    container = Container(settings)
    
    # Validate data type
    try:
        source_type = DataSourceType(data_type.lower())
    except ValueError:
        console.print(f"[red]Error: Invalid data type '{data_type}'[/red]")
        console.print(f"Valid types: csv, txt, pdf, pdf_scan")
        raise typer.Exit(1)
    
    # Create data source
    source = DataSource(
        source_type=source_type,
        path=data_path,
        metadata={},
    )
    
    # Create model config
    model_config = ModelConfig(
        model_type=model_type,
        target_column=target_column,
        test_size=test_size,
        random_state=settings.ml.random_seed,
    )
    
    # Execute pipeline
    try:
        pipeline = container.ml_pipeline_use_case
        results = pipeline.execute(
            source=source,
            model_config=model_config,
            perform_eda=perform_eda,
            eda_output_dir=Path(output_dir) / "eda",
            model_output_path=Path(output_dir) / "models" / f"{model_type}_model.pkl",
        )
        
        # Display results
        _display_results(results)
        
        console.print("\n[bold green]✓ Pipeline completed successfully![/bold green]\n")
        
    except Exception as e:
        console.print(f"\n[bold red]✗ Pipeline failed: {e}[/bold red]\n")
        raise typer.Exit(1)


@app.command()
def ingest(
    data_path: Annotated[str, typer.Argument(help="Path to input data file")],
    data_type: Annotated[str, typer.Option(help="Data source type")] = "csv",
    output_path: Annotated[str, typer.Option(help="Output path for processed data")] = "data/processed/data.pkl",
    clean: Annotated[bool, typer.Option(help="Clean the data")] = True,
    transform: Annotated[bool, typer.Option(help="Transform the data")] = True,
):
    """Ingest and preprocess data."""
    console.print("\n[bold blue]Data Ingestion Starting...[/bold blue]\n")
    
    settings = get_settings()
    setup_logging(settings)
    container = Container(settings)
    
    source_type = DataSourceType(data_type.lower())
    source = DataSource(source_type=source_type, path=data_path)
    
    use_case = container.data_ingestion_use_case
    processed_data = use_case.execute(source, clean=clean, transform=transform)
    
    # Save processed data
    container.data_repository.save(processed_data, Path(output_path))
    
    console.print(f"\n[green]✓ Data processed and saved to {output_path}[/green]\n")
    console.print(f"Shape: {processed_data.data.shape}")


@app.command()
def eda(
    data_path: Annotated[str, typer.Argument(help="Path to processed data file")],
    output_dir: Annotated[str, typer.Option(help="Output directory for EDA")] = "outputs/eda",
):
    """Perform exploratory data analysis."""
    console.print("\n[bold blue]EDA Starting...[/bold blue]\n")
    
    settings = get_settings()
    setup_logging(settings)
    container = Container(settings)
    
    # Load data
    processed_data = container.data_repository.load(Path(data_path))
    
    # Perform EDA
    use_case = container.eda_use_case
    report = use_case.execute(processed_data, output_dir=Path(output_dir))
    
    # Display insights
    console.print("\n[bold]Insights:[/bold]")
    for insight in report.insights:
        console.print(f"  • {insight}")
    
    console.print(f"\n[green]✓ EDA completed. Outputs saved to {output_dir}[/green]\n")


@app.command()
def train(
    data_path: Annotated[str, typer.Argument(help="Path to processed data file")],
    target_column: Annotated[str, typer.Option(help="Target column name")],
    model_type: Annotated[str, typer.Option(help="Model type")] = "random_forest",
    output_path: Annotated[str, typer.Option(help="Model output path")] = "models/model.pkl",
):
    """Train a machine learning model."""
    console.print("\n[bold blue]Model Training Starting...[/bold blue]\n")
    
    settings = get_settings()
    setup_logging(settings)
    container = Container(settings)
    
    # Load data
    processed_data = container.data_repository.load(Path(data_path))
    
    # Create config
    config = ModelConfig(
        model_type=model_type,
        target_column=target_column,
        random_state=settings.ml.random_seed,
    )
    
    # Train model
    use_case = container.model_training_use_case
    trained_model = use_case.execute(
        processed_data, config, model_path=Path(output_path)
    )
    
    # Display metrics
    console.print("\n[bold]Model Metrics:[/bold]")
    for metric, value in trained_model.metrics.items():
        console.print(f"  {metric}: {value:.4f}")
    
    console.print(f"\n[green]✓ Model trained and saved to {output_path}[/green]\n")


@app.command()
def predict(
    model_path: Annotated[str, typer.Argument(help="Path to trained model")],
    data_path: Annotated[str, typer.Argument(help="Path to data for prediction")],
    output_path: Annotated[str, typer.Option(help="Output path for predictions")] = "outputs/predictions.csv",
):
    """Make predictions using a trained model."""
    console.print("\n[bold blue]Prediction Starting...[/bold blue]\n")
    
    settings = get_settings()
    setup_logging(settings)
    container = Container(settings)
    
    # Load data
    import pandas as pd
    
    data = pd.read_csv(data_path)
    
    # Make predictions
    use_case = container.prediction_use_case
    prediction = use_case.execute(data, model_path=Path(model_path))
    
    # Save predictions
    output = data.copy()
    output["prediction"] = prediction.predictions
    if prediction.confidence_scores is not None:
        output["confidence"] = prediction.confidence_scores
    
    output.to_csv(output_path, index=False)
    
    console.print(f"\n[green]✓ Predictions saved to {output_path}[/green]\n")
    console.print(f"Total predictions: {len(prediction.predictions)}")


def _display_results(results: dict) -> None:
    """Display pipeline results."""
    console.print("\n[bold]Pipeline Results:[/bold]\n")
    
    # Data info
    if "processed_data" in results:
        data = results["processed_data"]
        console.print(f"[cyan]Data Shape:[/cyan] {data.data.shape}")
        console.print(f"[cyan]Processing Steps:[/cyan] {', '.join(data.processing_steps)}")
    
    # EDA info
    if "eda_report" in results:
        report = results["eda_report"]
        console.print(f"\n[cyan]EDA Insights:[/cyan]")
        for insight in report.insights[:5]:
            console.print(f"  • {insight}")
    
    # Model metrics
    if "trained_model" in results:
        model = results["trained_model"]
        console.print(f"\n[cyan]Model:[/cyan] {model.config.model_type}")
        
        table = Table(title="Model Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for metric, value in model.metrics.items():
            table.add_row(metric, f"{value:.4f}")
        
        console.print(table)


if __name__ == "__main__":
    app()
