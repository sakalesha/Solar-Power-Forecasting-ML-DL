"""
evaluate.py — Regression metrics for solar power forecasting.

All functions accept plain numpy arrays or pandas Series and return
a dict of metric scores.  Designed to work identically for baseline
and deep-learning models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL METRICS
# ─────────────────────────────────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(mean_absolute_error(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """
    Mean Absolute Percentage Error (%).
    `eps` prevents division by zero when y_true ≈ 0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Determination (R²)."""
    return float(r2_score(y_true, y_pred))


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation coefficient between predictions and actuals."""
    r, _ = pearsonr(np.asarray(y_true).flatten(), np.asarray(y_pred).flatten())
    return float(r)


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Normalised RMSE (%) — RMSE divided by the range of true values.
    Useful for comparing error across sites / capacity levels.
    """
    y_true = np.asarray(y_true)
    rng = y_true.max() - y_true.min()
    if rng < 1e-6:
        return float("nan")
    return float(rmse(y_true, y_pred) / rng * 100)


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE REPORT
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    label:  str = "",
) -> dict:
    """
    Compute all regression metrics and return as a dict.

    Parameters
    ----------
    y_true : ground-truth target values
    y_pred : model predictions
    label  : optional string tag (e.g. model name) added to the dict

    Returns
    -------
    dict with keys: label, rmse, mae, mape, r2, pearson_r, nrmse
    """
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()

    results = {
        "label":     label,
        "rmse":      round(rmse(y_true, y_pred),     4),
        "mae":       round(mae(y_true, y_pred),      4),
        "mape":      round(mape(y_true, y_pred),     4),
        "r2":        round(r2(y_true, y_pred),       4),
        "pearson_r": round(pearson_r(y_true, y_pred), 4),
        "nrmse_pct": round(nrmse(y_true, y_pred),   4),
    }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

def build_comparison_table(results_list: list[dict]) -> pd.DataFrame:
    """
    Build a nicely formatted comparison DataFrame from a list of metric dicts
    (as returned by compute_metrics).

    Parameters
    ----------
    results_list : list of dicts, one per model

    Returns
    -------
    pd.DataFrame sorted by RMSE ascending (best model first).
    """
    df = pd.DataFrame(results_list).set_index("label")
    df = df.sort_values("rmse", ascending=True)
    return df


def print_metrics(metrics: dict) -> None:
    """Pretty-print a single model's metrics to console."""
    label = metrics.get("label", "Model")
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  RMSE      : {metrics['rmse']:>10.2f} W")
    print(f"  MAE       : {metrics['mae']:>10.2f} W")
    print(f"  MAPE      : {metrics['mape']:>10.2f} %")
    print(f"  R²        : {metrics['r2']:>10.4f}")
    print(f"  Pearson r : {metrics['pearson_r']:>10.4f}")
    print(f"  NRMSE     : {metrics['nrmse_pct']:>10.2f} %")
