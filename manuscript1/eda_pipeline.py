#!/usr/bin/env python3
"""
EDA Pipeline for Crop Yield Dataset

- Dataset overview (per crop + overall)
- Yield distribution plots (histograms, boxplots)
- Correlation heatmap (numeric features vs Yield)
- Scatterplots of Yield vs key features (N, P, K, Temp, Humidity, pH, Rainfall)
- Outlier detection (boxplots)
- Saves all outputs to an output directory
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.switch_backend("Agg")  # headless

# ---------------------- EDA Functions ----------------------

def dataset_overview(df, out_dir: Path):
    """Basic stats + missing values per crop and overall."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # overall summary
    overall_stats = df.describe(include="all").transpose()
    overall_stats.to_csv(out_dir / "overall_summary.csv")

    # missing values
    missing = df.isnull().sum().to_frame("missing_count")
    missing["missing_pct"] = (missing["missing_count"] / len(df)) * 100
    missing.to_csv(out_dir / "missing_values.csv")

    # per-crop counts
    crop_counts = df["Crop"].value_counts().to_frame("count")
    crop_counts.to_csv(out_dir / "crop_counts.csv")

    return overall_stats, missing, crop_counts


def yield_distribution(df, out_dir: Path):
    """Histogram + boxplot of yield per crop and overall."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # histogram per crop
    for crop, sub in df.groupby("Crop"):
        plt.figure()
        sns.histplot(sub["Yield"], kde=True, bins=20)
        plt.title(f"Yield Distribution — {crop}")
        plt.xlabel("Yield")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_dir / f"hist_{crop}.png", dpi=150)
        plt.close()

    # overall boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="Crop", y="Yield", data=df)
    plt.xticks(rotation=45, ha="right")
    plt.title("Yield Distribution by Crop (Boxplot)")
    plt.tight_layout()
    plt.savefig(out_dir / "yield_boxplot.png", dpi=200)
    plt.close()


def correlation_heatmap(df, out_dir: Path):
    """Correlation heatmap for numeric features."""
    out_dir.mkdir(parents=True, exist_ok=True)

    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(out_dir / "correlation_heatmap.png", dpi=200)
    plt.close()

    corr.to_csv(out_dir / "correlation_matrix.csv")


def scatterplots(df, out_dir: Path):
    """Scatterplots of yield vs each key feature."""
    out_dir.mkdir(parents=True, exist_ok=True)

    features = ["Nitrogen","Phosphorus","Potassium","Temperature",
                "Humidity","pH_Value","Rainfall"]

    for feat in features:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=feat, y="Yield", hue="Crop", data=df, alpha=0.7)
        sns.regplot(x=feat, y="Yield", data=df, scatter=False, color="black", lowess=True)
        plt.title(f"Yield vs {feat}")
        plt.tight_layout()
        plt.savefig(out_dir / f"scatter_{feat}.png", dpi=150)
        plt.close()


def outlier_analysis(df, out_dir: Path):
    """Boxplots to detect yield outliers per crop."""
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    sns.boxplot(x="Crop", y="Yield", data=df)
    plt.xticks(rotation=45, ha="right")
    plt.title("Outlier Analysis — Yield per Crop")
    plt.tight_layout()
    plt.savefig(out_dir / "outliers_boxplot.png", dpi=200)
    plt.close()


# ---------------------- Main ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to Crop_Yield_Prediction.csv")
    ap.add_argument("--out", required=True, help="Output directory for EDA results")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)

    # 1. Dataset overview
    dataset_overview(df, out_dir / "overview")

    # 2. Yield distribution
    yield_distribution(df, out_dir / "yield_distribution")

    # 3. Correlation heatmap
    correlation_heatmap(df, out_dir / "correlation")

    # 4. Scatterplots
    scatterplots(df, out_dir / "scatterplots")

    # 5. Outlier analysis
    outlier_analysis(df, out_dir / "outliers")

    print("✅ EDA complete. Results saved to:", out_dir)


if __name__ == "__main__":
    main()


# python eda_pipeline.py --data Crop_Yield_Prediction.csv --out eda_outputs
