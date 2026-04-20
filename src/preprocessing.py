"""
preprocessing.py — Clean, resample, split, and scale the FPV dataset.

Pipeline stages:
  1. Drop rows where target (INVPWR) is NaN.
  2. Resample 1-min data → 15-min (mean aggregation).
  3. Filter nighttime rows (irradiance below threshold).
  4. Drop columns with > threshold % missing values.
  5. Forward-fill remaining short gaps; drop long-gap rows.
  6. Chronological train / val / test split (no shuffle).
  7. Fit StandardScaler on train, transform all splits.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
from pathlib import Path

from src.config import (
    RESAMPLE_RULE, IRRADIANCE_THRESHOLD,
    TRAIN_FRAC, VAL_FRAC, TEST_FRAC,
    SCALER_TYPE, TARGET, SEED,
    MODELS_DIR,
)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Basic cleaning
# ─────────────────────────────────────────────────────────────────────────────

def drop_bad_target_rows(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """Drop rows where the target column is NaN or negative."""
    before = len(df)
    df = df[df[target].notna() & (df[target] >= 0)]
    print(f"  drop_bad_target: removed {before - len(df):,} rows → {len(df):,} remain")
    return df


def drop_high_null_columns(df: pd.DataFrame, threshold: float = 0.50) -> pd.DataFrame:
    """Drop any column that has more than `threshold` fraction of NaNs."""
    null_frac = df.isnull().mean()
    drop_cols = null_frac[null_frac > threshold].index.tolist()
    if drop_cols:
        print(f"  drop_high_null_columns: dropping {len(drop_cols)} cols "
              f"with >{threshold*100:.0f}% NaN: {drop_cols}")
        df = df.drop(columns=drop_cols)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Resample 1-min → 15-min
# ─────────────────────────────────────────────────────────────────────────────

def resample_to_15min(df: pd.DataFrame, rule: str = RESAMPLE_RULE) -> pd.DataFrame:
    """
    Aggregate 1-minute data to 15-minute intervals using mean.
    Drops the non-numeric 'site' column before resampling and re-attaches it.
    """
    site_label = df["site"].iloc[0] if "site" in df.columns else "unknown"

    numeric_df = df.select_dtypes(include="number")
    resampled  = numeric_df.resample(rule).mean()

    resampled["site"] = site_label
    print(f"  resample: {len(df):,} rows → {len(resampled):,} rows "
          f"({rule} aggregation)")
    return resampled


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Filter nighttime
# ─────────────────────────────────────────────────────────────────────────────

def filter_daytime(
    df: pd.DataFrame,
    irr_col: str = "irr_horiz",
    threshold: float = IRRADIANCE_THRESHOLD,
) -> pd.DataFrame:
    """
    Keep only rows where irradiance > threshold (daytime records).
    If irr_col doesn't exist, falls back to irr_plane or skips filtering.
    """
    if irr_col not in df.columns:
        # Try alternative column
        for alt in ["irr_plane", "FHZIRR", "FPHIRR"]:
            if alt in df.columns:
                irr_col = alt
                break
        else:
            print("  filter_daytime: no irradiance column found — skipping")
            return df

    before = len(df)
    df = df[df[irr_col] > threshold]
    print(f"  filter_daytime: removed {before - len(df):,} nighttime rows "
          f"→ {len(df):,} remain")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Fill short gaps
# ─────────────────────────────────────────────────────────────────────────────

def fill_gaps(df: pd.DataFrame, max_gap: int = 4) -> pd.DataFrame:
    """
    Forward-fill NaN values up to `max_gap` consecutive periods.
    Any remaining NaN rows are dropped.
    """
    before_nan = df[TARGET].isna().sum()
    df = df.ffill(limit=max_gap)
    after_nan  = df[TARGET].isna().sum()
    df = df.dropna(subset=[TARGET])
    print(f"  fill_gaps: filled {before_nan - after_nan} NaNs, "
          f"dropped {after_nan} remaining → {len(df):,} rows")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Full cleaning pipeline
# ─────────────────────────────────────────────────────────────────────────────

def clean(
    df: pd.DataFrame,
    resample: bool = True,
    filter_night: bool = True,
) -> pd.DataFrame:
    """
    Apply all cleaning steps in order.

    Parameters
    ----------
    df            : standardised DataFrame from data_loader.load_and_standardise()
    resample      : if True, aggregate 1-min → 15-min
    filter_night  : if True, remove nighttime rows

    Returns
    -------
    Cleaned DataFrame with DatetimeIndex.
    """
    print("🔧 Starting cleaning pipeline...")
    df = drop_bad_target_rows(df)
    df = drop_high_null_columns(df)

    if resample:
        df = resample_to_15min(df)

    if filter_night:
        df = filter_daytime(df)

    df = fill_gaps(df)

    # Drop 'site' column (string) and any remaining non-numeric
    df = df.select_dtypes(include="number")

    print(f"✅ Clean complete: {len(df):,} rows, {len(df.columns)} columns")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Train / Val / Test chronological split
# ─────────────────────────────────────────────────────────────────────────────

def split_chronological(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac:   float = VAL_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split into train / validation / test sets in chronological order.
    No shuffling — this preserves temporal structure for time-series models.

    Returns
    -------
    (train_df, val_df, test_df)
    """
    n     = len(df)
    n_tr  = int(n * train_frac)
    n_val = int(n * val_frac)

    train = df.iloc[:n_tr]
    val   = df.iloc[n_tr : n_tr + n_val]
    test  = df.iloc[n_tr + n_val :]

    print(f"📊 Split: train={len(train):,} | val={len(val):,} | test={len(test):,}")
    print(f"   Train:  {train.index[0]} → {train.index[-1]}")
    print(f"   Val:    {val.index[0]}   → {val.index[-1]}")
    print(f"   Test:   {test.index[0]}  → {test.index[-1]}")
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Feature / target extraction + scaling
# ─────────────────────────────────────────────────────────────────────────────

def get_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str = TARGET,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extract feature matrix X and target series y from the DataFrame.
    Drops any rows where features have NaN.
    """
    available = [c for c in feature_cols if c in df.columns]
    missing   = set(feature_cols) - set(available)
    if missing:
        print(f"  ⚠️  Feature cols not found (skipped): {missing}")

    df_clean = df[available + [target]].dropna()
    X = df_clean[available]
    y = df_clean[target]
    return X, y


def fit_scaler(
    X_train: pd.DataFrame,
    scaler_type: str = SCALER_TYPE,
    save_path: Path | None = None,
) -> StandardScaler | MinMaxScaler:
    """
    Fit a scaler on the training feature matrix.

    Parameters
    ----------
    X_train    : training features DataFrame
    scaler_type: "standard" | "minmax"
    save_path  : if given, persist scaler to disk with joblib

    Returns
    -------
    Fitted scaler instance.
    """
    scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
    scaler.fit(X_train)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, save_path)
        print(f"💾 Scaler saved → {save_path}")

    return scaler


def apply_scaler(
    X: pd.DataFrame,
    scaler,
) -> pd.DataFrame:
    """Apply a pre-fitted scaler and return a DataFrame with original index/cols."""
    scaled = scaler.transform(X)
    return pd.DataFrame(scaled, index=X.index, columns=X.columns)


def load_scaler(path: Path) -> StandardScaler | MinMaxScaler:
    """Load a previously saved scaler from disk."""
    return joblib.load(path)


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE  (convenience function)
# ─────────────────────────────────────────────────────────────────────────────

def full_pipeline(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str = TARGET,
    resample: bool = True,
    filter_night: bool = True,
) -> dict:
    """
    Run the entire preprocessing pipeline and return a results dict.

    Returns
    -------
    {
        "train": (X_train_scaled, y_train),
        "val":   (X_val_scaled,   y_val),
        "test":  (X_test_scaled,  y_test),
        "scaler": scaler,
        "feature_cols": available_features,
        "train_df": train_df,
        "val_df":   val_df,
        "test_df":  test_df,
    }
    """
    # Clean
    df_clean = clean(df, resample=resample, filter_night=filter_night)

    # Split
    train_df, val_df, test_df = split_chronological(df_clean)

    # Extract X, y
    X_tr, y_tr = get_xy(train_df, feature_cols, target)
    X_va, y_va = get_xy(val_df,   feature_cols, target)
    X_te, y_te = get_xy(test_df,  feature_cols, target)

    # Scale — fit only on train
    scaler_path = MODELS_DIR / "scaler.joblib"
    scaler = fit_scaler(X_tr, save_path=scaler_path)
    X_tr_sc = apply_scaler(X_tr, scaler)
    X_va_sc = apply_scaler(X_va, scaler)
    X_te_sc = apply_scaler(X_te, scaler)

    return {
        "train":        (X_tr_sc, y_tr),
        "val":          (X_va_sc, y_va),
        "test":         (X_te_sc, y_te),
        "scaler":       scaler,
        "feature_cols": X_tr.columns.tolist(),
        "train_df":     train_df,
        "val_df":       val_df,
        "test_df":      test_df,
    }
