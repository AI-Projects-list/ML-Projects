"""Exploratory Data Analysis implementation."""

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger

from src.domain.entities import EDAReport, ProcessedData
from src.domain.repositories import IEDAAnalyzer


class EDAAnalyzer(IEDAAnalyzer):
    """Performs exploratory data analysis and generates insights."""
    
    def __init__(self, figure_size: tuple = (10, 6)):
        """
        Initialize EDA analyzer.
        
        Args:
            figure_size: Default figure size for plots
        """
        self.figure_size = figure_size
        sns.set_style("whitegrid")
    
    def analyze(self, data: ProcessedData) -> EDAReport:
        """
        Perform comprehensive exploratory data analysis.
        
        Args:
            data: Processed data to analyze
            
        Returns:
            EDA report with statistics and insights
        """
        logger.info("Performing exploratory data analysis...")
        df = data.data
        
        # Basic information
        data_shape = df.shape
        column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
        missing_values = df.isnull().sum().to_dict()
        
        # Statistical summary
        statistics = self._generate_statistics(df)
        
        # Correlation analysis
        correlations = self._calculate_correlations(df)
        
        # Generate insights
        insights = self._generate_insights(df, statistics, correlations)
        
        report = EDAReport(
            data_shape=data_shape,
            column_types=column_types,
            missing_values=missing_values,
            statistics=statistics,
            correlations=correlations,
            insights=insights,
        )
        
        logger.info(f"EDA completed. Generated {len(insights)} insights.")
        return report
    
    def generate_visualizations(
        self, data: ProcessedData, output_dir: Path
    ) -> List[str]:
        """
        Generate and save visualization plots.
        
        Args:
            data: Processed data to visualize
            output_dir: Directory to save plots
            
        Returns:
            List of saved plot file paths
        """
        logger.info("Generating visualizations...")
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_plots: List[str] = []
        
        df = data.data
        
        # 1. Missing values heatmap
        if df.isnull().sum().sum() > 0:
            plot_path = self._plot_missing_values(df, output_dir)
            saved_plots.append(plot_path)
        
        # 2. Distribution plots for numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            plot_path = self._plot_distributions(df, numeric_cols, output_dir)
            saved_plots.append(plot_path)
        
        # 3. Correlation heatmap
        if len(numeric_cols) > 1:
            plot_path = self._plot_correlation_matrix(df, numeric_cols, output_dir)
            saved_plots.append(plot_path)
        
        # 4. Box plots for outlier detection
        if len(numeric_cols) > 0:
            plot_path = self._plot_boxplots(df, numeric_cols, output_dir)
            saved_plots.append(plot_path)
        
        # 5. Categorical distribution
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        if len(categorical_cols) > 0:
            plot_path = self._plot_categorical(df, categorical_cols, output_dir)
            saved_plots.append(plot_path)
        
        logger.info(f"Generated {len(saved_plots)} visualizations")
        return saved_plots
    
    def _generate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate statistical summary."""
        stats: Dict[str, Any] = {
            "shape": df.shape,
            "memory_usage": df.memory_usage(deep=True).sum(),
            "numeric_summary": {},
            "categorical_summary": {},
        }
        
        # Numeric statistics
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            stats["numeric_summary"] = df[numeric_cols].describe().to_dict()
        
        # Categorical statistics
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        if len(categorical_cols) > 0:
            stats["categorical_summary"] = {
                col: {
                    "unique_values": df[col].nunique(),
                    "top_value": df[col].mode()[0] if len(df[col].mode()) > 0 else None,
                    "top_frequency": df[col].value_counts().iloc[0]
                    if len(df[col].value_counts()) > 0
                    else 0,
                }
                for col in categorical_cols
            }
        
        return stats
    
    def _calculate_correlations(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """Calculate correlation matrix for numeric columns."""
        numeric_cols = df.select_dtypes(include=["number"]).columns
        
        if len(numeric_cols) > 1:
            return df[numeric_cols].corr()
        return None
    
    def _generate_insights(
        self,
        df: pd.DataFrame,
        statistics: Dict[str, Any],
        correlations: pd.DataFrame | None,
    ) -> List[str]:
        """Generate automated insights from the data."""
        insights: List[str] = []
        
        # Data size insight
        insights.append(
            f"Dataset contains {df.shape[0]:,} rows and {df.shape[1]} columns"
        )
        
        # Missing values insight
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        if missing_pct > 0:
            insights.append(f"Missing data: {missing_pct:.2f}% of total values")
        
        # High correlation insight
        if correlations is not None:
            high_corr = []
            for i in range(len(correlations.columns)):
                for j in range(i + 1, len(correlations.columns)):
                    if abs(correlations.iloc[i, j]) > 0.8:
                        high_corr.append(
                            f"{correlations.columns[i]} & {correlations.columns[j]}"
                            f" ({correlations.iloc[i, j]:.2f})"
                        )
            
            if high_corr:
                insights.append(f"High correlations detected: {', '.join(high_corr[:3])}")
        
        # Outliers insight (simplified)
        numeric_cols = df.select_dtypes(include=["number"]).columns
        for col in numeric_cols[:5]:  # Check first 5 numeric columns
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            if outliers > 0:
                insights.append(f"'{col}' has {outliers} potential outliers")
        
        return insights
    
    def _plot_missing_values(self, df: pd.DataFrame, output_dir: Path) -> str:
        """Plot missing values heatmap."""
        plt.figure(figsize=self.figure_size)
        sns.heatmap(df.isnull(), cbar=True, cmap="viridis", yticklabels=False)
        plt.title("Missing Values Heatmap")
        plt.tight_layout()
        
        path = str(output_dir / "missing_values.png")
        plt.savefig(path, dpi=100)
        plt.close()
        return path
    
    def _plot_distributions(
        self, df: pd.DataFrame, columns: pd.Index, output_dir: Path
    ) -> str:
        """Plot distributions of numeric columns."""
        n_cols = min(len(columns), 6)  # Max 6 subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, col in enumerate(columns[:n_cols]):
            df[col].hist(bins=30, ax=axes[idx], edgecolor="black")
            axes[idx].set_title(f"Distribution of {col}")
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel("Frequency")
        
        # Hide unused subplots
        for idx in range(n_cols, 6):
            axes[idx].axis("off")
        
        plt.tight_layout()
        path = str(output_dir / "distributions.png")
        plt.savefig(path, dpi=100)
        plt.close()
        return path
    
    def _plot_correlation_matrix(
        self, df: pd.DataFrame, columns: pd.Index, output_dir: Path
    ) -> str:
        """Plot correlation matrix heatmap."""
        plt.figure(figsize=(12, 10))
        corr = df[columns].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
        plt.title("Correlation Matrix")
        plt.tight_layout()
        
        path = str(output_dir / "correlation_matrix.png")
        plt.savefig(path, dpi=100)
        plt.close()
        return path
    
    def _plot_boxplots(
        self, df: pd.DataFrame, columns: pd.Index, output_dir: Path
    ) -> str:
        """Plot box plots for outlier detection."""
        n_cols = min(len(columns), 6)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, col in enumerate(columns[:n_cols]):
            df.boxplot(column=col, ax=axes[idx])
            axes[idx].set_title(f"Box Plot - {col}")
        
        for idx in range(n_cols, 6):
            axes[idx].axis("off")
        
        plt.tight_layout()
        path = str(output_dir / "boxplots.png")
        plt.savefig(path, dpi=100)
        plt.close()
        return path
    
    def _plot_categorical(
        self, df: pd.DataFrame, columns: pd.Index, output_dir: Path
    ) -> str:
        """Plot categorical value distributions."""
        n_cols = min(len(columns), 4)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, col in enumerate(columns[:n_cols]):
            value_counts = df[col].value_counts().head(10)
            value_counts.plot(kind="bar", ax=axes[idx])
            axes[idx].set_title(f"Top Values - {col}")
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel("Count")
            axes[idx].tick_params(axis="x", rotation=45)
        
        for idx in range(n_cols, 4):
            axes[idx].axis("off")
        
        plt.tight_layout()
        path = str(output_dir / "categorical_distributions.png")
        plt.savefig(path, dpi=100)
        plt.close()
        return path
