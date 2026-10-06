"""
Exploratory Data Analysis (EDA) Module
=======================================
Implements univariate, correlation, and class balance analysis functions
for the CPCB Water Quality Monitoring pipeline.

Milestone 3.1: Univariate, Correlation & Class-Balance Analysis
Owner: Data Lead (Person A)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.load_data import load_raw_data
from src.data.preprocessing import run_full_cleaning_pipeline, FROZEN_CANDIDATE_FEATURES

logger = logging.getLogger("water_quality.eda")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def compute_summary_statistics(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute comprehensive univariate summary statistics for numeric features.
    
    Includes: count, mean, std, min, 25%, 50% (median), 75%, max,
    IQR, Skewness, and Kurtosis.
    """
    if feature_columns is None:
        feature_columns = [col for col in FROZEN_CANDIDATE_FEATURES if col in df.columns]
        
    stats_list = []
    for col in feature_columns:
        series = df[col].dropna()
        q25 = float(series.quantile(0.25))
        q75 = float(series.quantile(0.75))
        iqr = q75 - q25
        skew = float(series.skew())
        kurt = float(series.kurt())
        
        stats_list.append({
            "feature": col,
            "count": int(series.count()),
            "missing_count": int(df[col].isna().sum()),
            "missing_pct": float((df[col].isna().sum() / len(df)) * 100.0),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "q25": q25,
            "median": float(series.median()),
            "q75": q75,
            "max": float(series.max()),
            "iqr": iqr,
            "skewness": skew,
            "kurtosis": kurt,
        })
        
    summary_df = pd.DataFrame(stats_list)
    return summary_df


def generate_univariate_plots(
    df: pd.DataFrame,
    output_dir: str | Path = "reports/figures/eda",
    feature_columns: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """
    Generate and save histogram + boxplot distribution figures for each numeric feature.
    
    Returns a dictionary mapping feature column names to their saved file paths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    if feature_columns is None:
        feature_columns = [col for col in FROZEN_CANDIDATE_FEATURES if col in df.columns]
        
    saved_plots: Dict[str, Path] = {}
    
    # Palette configuration
    sns.set_theme(style="whitegrid")
    
    for col in feature_columns:
        series = df[col].dropna()
        safe_name = col.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. Histogram with KDE
        sns.histplot(series, kde=True, ax=axes[0], color="#2b5c8f", bins=30)
        axes[0].set_title(f"Distribution: {col}", fontsize=12, fontweight="bold")
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("Frequency")
        
        # Annotation for skewness
        skew_val = series.skew()
        axes[0].text(
            0.95, 0.95,
            f"Skewness: {skew_val:.2f}\nMedian: {series.median():.2f}",
            transform=axes[0].transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="gray"),
            fontsize=10,
        )
        
        # 2. Box Plot
        sns.boxplot(x=series, ax=axes[1], color="#4ca1af", fliersize=4)
        axes[1].set_title(f"Box Plot: {col}", fontsize=12, fontweight="bold")
        axes[1].set_xlabel(col)
        
        plt.tight_layout()
        plot_file = out_path / f"{safe_name}_distribution.png"
        fig.savefig(plot_file, dpi=300)
        plt.close(fig)
        
        saved_plots[col] = plot_file
        logger.info(f"Saved distribution plot for '{col}' to {plot_file}")
        
    return saved_plots


def compute_correlation_matrix(
    df: pd.DataFrame,
    method: str = "pearson",
    feature_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute correlation matrix for numeric candidate features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing feature columns.
    method : str
        Correlation method ('pearson' or 'spearman').
    feature_columns : Optional[List[str]]
        Specific columns to correlate. Defaults to FROZEN_CANDIDATE_FEATURES.
    """
    if feature_columns is None:
        feature_columns = [col for col in FROZEN_CANDIDATE_FEATURES if col in df.columns]
        
    corr_df = df[feature_columns].corr(method=method)
    return corr_df


def find_redundant_feature_pairs(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.85,
) -> List[Dict[str, Any]]:
    """
    Identify pairs of features with absolute correlation strictly greater than threshold.
    
    Note: Redundant pairs are flagged for review only; never automatically removed.
    """
    redundant_pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col1 = cols[i]
            col2 = cols[j]
            val = float(corr_matrix.loc[col1, col2])
            if abs(val) > threshold:
                redundant_pairs.append({
                    "feature_1": col1,
                    "feature_2": col2,
                    "correlation": val,
                    "abs_correlation": abs(val),
                })
    return redundant_pairs


def generate_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    output_path: str | Path = "reports/figures/eda/correlation_heatmap.png",
    title: str = "Feature Correlation Matrix (Pearson)",
) -> Path:
    """Generate and save an annotated correlation heatmap."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        linewidths=0.8,
        cbar_kws={"label": "Correlation Coefficient"},
    )
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    
    logger.info(f"Saved correlation heatmap to {out_file}")
    return out_file


def compute_class_distribution(
    df: pd.DataFrame,
    target_column: str = "wqi_class",
    output_csv: Optional[str | Path] = "reports/class_distribution.csv",
) -> pd.DataFrame:
    """
    Compute count and percentage distribution for target classes.
    
    Exports table to reports/class_distribution.csv if output_csv is provided.
    """
    counts = df[target_column].value_counts()
    percentages = df[target_column].value_counts(normalize=True) * 100.0
    
    dist_df = pd.DataFrame({
        "class": counts.index,
        "count": counts.values,
        "percentage": percentages.values.round(4),
    })
    
    # Sort logically if wqi_class
    order_map = {"Excellent": 0, "Good": 1, "Poor": 2, "Very Poor": 3, "Unsuitable": 4}
    if set(dist_df["class"]).issubset(set(order_map.keys())):
        dist_df["sort_order"] = dist_df["class"].map(order_map)
        dist_df = dist_df.sort_values("sort_order").drop(columns=["sort_order"]).reset_index(drop=True)
    else:
        dist_df = dist_df.reset_index(drop=True)
        
    if output_csv:
        out_csv_path = Path(output_csv)
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        dist_df.to_csv(out_csv_path, index=False)
        logger.info(f"Exported class distribution table to {out_csv_path}")
        
    return dist_df


def generate_class_distribution_plot(
    dist_df: pd.DataFrame,
    output_path: str | Path = "reports/figures/eda/class_distribution_barplot.png",
) -> Path:
    """Generate and save class distribution bar chart with count and percentage annotations."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 5.5))
    palette = ["#2ecc71", "#3498db", "#f39c12", "#e67e22", "#e74c3c"]
    
    ax = sns.barplot(
        data=dist_df,
        x="class",
        y="count",
        hue="class",
        palette=palette[:len(dist_df)],
        legend=False,
        edgecolor="black",
        linewidth=0.8,
    )
    
    total = dist_df["count"].sum()
    for i, row in dist_df.iterrows():
        cnt = int(row["count"])
        pct = float(row["percentage"])
        ax.text(
            i, cnt + (total * 0.015),
            f"{cnt}\n({pct:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
        
    plt.title("WQI Ground-Truth Class Distribution (CPCB Dataset)", fontsize=13, fontweight="bold")
    plt.xlabel("Water Quality Class", fontsize=11)
    plt.ylabel("Number of Observations", fontsize=11)
    plt.ylim(0, dist_df["count"].max() * 1.15)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    
    logger.info(f"Saved class distribution plot to {out_file}")
    return out_file


def run_full_eda(
    data_path: str | Path = "data/interim/cpcb_labeled.csv",
    reports_dir: str | Path = "reports",
) -> Dict[str, Any]:
    """
    Orchestrate full Milestone 3.1 EDA pipeline:
    1. Load data
    2. Summary stats
    3. Univariate plots
    4. Pearson & Spearman correlation matrices & heatmap
    5. Redundant pairs identification
    6. Class balance analysis & export
    """
    reports_path = Path(reports_dir)
    figures_path = reports_path / "figures" / "eda"
    figures_path.mkdir(parents=True, exist_ok=True)
    
    df = load_raw_data(data_path)
    
    summary_stats = compute_summary_statistics(df)
    univariate_plots = generate_univariate_plots(df, output_dir=figures_path)
    
    pearson_corr = compute_correlation_matrix(df, method="pearson")
    spearman_corr = compute_correlation_matrix(df, method="spearman")
    
    redundant_pairs = find_redundant_feature_pairs(pearson_corr, threshold=0.85)
    
    heatmap_file = generate_correlation_heatmap(
        pearson_corr,
        output_path=figures_path / "correlation_heatmap.png",
        title="Feature Correlation Heatmap (Pearson |r|)",
    )
    
    spearman_heatmap_file = generate_correlation_heatmap(
        spearman_corr,
        output_path=figures_path / "spearman_correlation_heatmap.png",
        title="Feature Correlation Heatmap (Spearman Rank)",
    )
    
    class_dist = compute_class_distribution(
        df,
        target_column="wqi_class",
        output_csv=reports_path / "class_distribution.csv",
    )
    
    class_plot = generate_class_distribution_plot(
        class_dist,
        output_path=figures_path / "class_distribution_barplot.png",
    )
    
    results = {
        "summary_statistics": summary_stats,
        "univariate_plots": univariate_plots,
        "pearson_correlation": pearson_corr,
        "spearman_correlation": spearman_corr,
        "redundant_pairs": redundant_pairs,
        "heatmap_file": heatmap_file,
        "spearman_heatmap_file": spearman_heatmap_file,
        "class_distribution": class_dist,
        "class_distribution_plot": class_plot,
    }
    
    logger.info("Milestone 3.1 EDA pipeline completed successfully.")
    return results


if __name__ == "__main__":
    run_full_eda()
