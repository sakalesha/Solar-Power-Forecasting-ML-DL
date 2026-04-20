"""
visualize.py — Reusable plotting functions for solar power forecasting.

All functions return a matplotlib Figure so the caller can save or display.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

# ─── consistent aesthetics across all plots ───────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE = sns.color_palette("tab10")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _savefig(fig: plt.Figure, path: Path | None) -> None:
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"  💾 Figure saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# EDA PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_time_series(
    df: pd.DataFrame,
    columns: list[str],
    title: str = "Time-Series Overview",
    figsize: tuple = (14, 8),
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot one or more columns over time (stacked sub-plots)."""
    n = len(columns)
    fig, axes = plt.subplots(n, 1, figsize=(figsize[0], figsize[1] * n // 2),
                              sharex=True)
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        if col not in df.columns:
            ax.text(0.5, 0.5, f"Column '{col}' not found", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        ax.plot(df.index, df[col], lw=0.8, color=PALETTE[0])
        ax.set_ylabel(col, fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, save_path)
    return fig


def plot_daily_profile(
    df: pd.DataFrame,
    col: str,
    title: str = "Average Daily Profile",
    figsize: tuple = (10, 4),
    save_path: Path | None = None,
) -> plt.Figure:
    """Average value per hour-of-day (±std band)."""
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not in DataFrame")

    hourly = df[col].groupby(df.index.hour)
    mean   = hourly.mean()
    std    = hourly.std()

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(mean.index, mean.values, lw=2, color=PALETTE[1], label="Mean")
    ax.fill_between(mean.index, mean - std, mean + std,
                    alpha=0.25, color=PALETTE[1], label="±1 std")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel(col)
    ax.set_title(title, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    _savefig(fig, save_path)
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    figsize: tuple = (12, 10),
    save_path: Path | None = None,
) -> plt.Figure:
    """Correlation heatmap for all numeric columns."""
    corr = df.select_dtypes("number").corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, linewidths=0.5, ax=ax, annot_kws={"size": 7})
    ax.set_title("Feature Correlation Matrix", fontweight="bold")
    fig.tight_layout()
    _savefig(fig, save_path)
    return fig


def plot_missing_data(
    df: pd.DataFrame,
    figsize: tuple = (12, 5),
    save_path: Path | None = None,
) -> plt.Figure:
    """Bar chart of % missing values per column."""
    null_pct = df.isnull().mean().sort_values(ascending=False) * 100
    null_pct = null_pct[null_pct > 0]

    fig, ax = plt.subplots(figsize=figsize)
    null_pct.plot(kind="bar", ax=ax, color=PALETTE[3], edgecolor="white")
    ax.axhline(50, color="red", ls="--", lw=1, label="50% threshold")
    ax.set_ylabel("Missing (%)")
    ax.set_title("Missing Data by Column", fontweight="bold")
    ax.legend()
    plt.xticks(rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    _savefig(fig, save_path)
    return fig


def plot_distribution(
    df: pd.DataFrame,
    col: str,
    figsize: tuple = (8, 4),
    save_path: Path | None = None,
) -> plt.Figure:
    """Histogram + KDE for a single column."""
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not in DataFrame")

    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(df[col].dropna(), bins=60, kde=True, color=PALETTE[0], ax=ax)
    ax.set_xlabel(col)
    ax.set_title(f"Distribution of {col}", fontweight="bold")
    fig.tight_layout()
    _savefig(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MODEL EVALUATION PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_actual_vs_predicted(
    y_true,
    y_pred,
    label: str = "Model",
    n_points: int = 500,
    figsize: tuple = (12, 4),
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Time-series plot of actual vs predicted values.
    Shows the first `n_points` samples for clarity.
    """
    y_true = np.asarray(y_true).flatten()[:n_points]
    y_pred = np.asarray(y_pred).flatten()[:n_points]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(y_true, lw=1.2, label="Actual",    color=PALETTE[0])
    ax.plot(y_pred, lw=1.2, label="Predicted", color=PALETTE[1], alpha=0.8)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("INVPWR (W)")
    ax.set_title(f"Actual vs Predicted — {label}", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    _savefig(fig, save_path)
    return fig


def plot_scatter(
    y_true,
    y_pred,
    label: str = "Model",
    figsize: tuple = (6, 6),
    save_path: Path | None = None,
) -> plt.Figure:
    """Scatter plot: actual vs predicted with 45-degree reference line."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(y_true, y_pred, alpha=0.3, s=8, color=PALETTE[2])
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual INVPWR (W)")
    ax.set_ylabel("Predicted INVPWR (W)")
    ax.set_title(f"Actual vs Predicted Scatter — {label}", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    _savefig(fig, save_path)
    return fig


def plot_residuals(
    y_true,
    y_pred,
    label: str = "Model",
    figsize: tuple = (10, 4),
    save_path: Path | None = None,
) -> plt.Figure:
    """Residual plot (error = actual − predicted)."""
    residuals = np.asarray(y_true).flatten() - np.asarray(y_pred).flatten()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].plot(residuals, lw=0.7, color=PALETTE[3], alpha=0.7)
    axes[0].axhline(0, color="black", lw=1)
    axes[0].set_xlabel("Sample")
    axes[0].set_ylabel("Residual (W)")
    axes[0].set_title("Residuals over Time")

    sns.histplot(residuals, bins=50, kde=True, ax=axes[1], color=PALETTE[3])
    axes[1].axvline(0, color="black", lw=1)
    axes[1].set_xlabel("Residual (W)")
    axes[1].set_title("Residual Distribution")
    axes[1].set_ylabel("")

    fig.suptitle(f"Residual Analysis — {label}", fontweight="bold")
    fig.tight_layout()
    _savefig(fig, save_path)
    return fig


def plot_metrics_comparison(
    results_df: pd.DataFrame,
    metric: str = "rmse",
    figsize: tuple = (8, 4),
    save_path: Path | None = None,
) -> plt.Figure:
    """Horizontal bar chart comparing a metric across all models."""
    fig, ax = plt.subplots(figsize=figsize)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(results_df))]
    results_df[metric].sort_values(ascending=True).plot(
        kind="barh", ax=ax, color=colors, edgecolor="white"
    )
    ax.set_xlabel(metric.upper())
    ax.set_title(f"Model Comparison — {metric.upper()}", fontweight="bold")
    fig.tight_layout()
    _savefig(fig, save_path)
    return fig


def plot_training_history(
    train_losses: list[float],
    val_losses:   list[float],
    label: str = "Model",
    figsize: tuple = (8, 4),
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot training and validation loss curves for DL models."""
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(epochs, train_losses, lw=2, label="Train Loss", color=PALETTE[0])
    ax.plot(epochs, val_losses,   lw=2, label="Val Loss",   color=PALETTE[1])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title(f"Training History — {label}", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    _savefig(fig, save_path)
    return fig


def plot_feature_importance(
    feature_names: list[str],
    importances:   np.ndarray,
    label: str = "Random Forest",
    top_n: int = 20,
    figsize: tuple = (8, 6),
    save_path: Path | None = None,
) -> plt.Figure:
    """Horizontal bar chart of feature importances (RF / XGBoost)."""
    feat_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    feat_df = feat_df.sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=figsize)
    feat_df.set_index("feature")["importance"].plot(
        kind="barh", ax=ax, color=PALETTE[4], edgecolor="white"
    )
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Feature Importance — {label} (Top {top_n})", fontweight="bold")
    fig.tight_layout()
    _savefig(fig, save_path)
    return fig
