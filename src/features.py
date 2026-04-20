"""
features.py — Feature engineering for the solar power forecasting pipeline.

Features added:
  • Time cyclical (sin/cos of hour-of-day, day-of-year)
  • Lag values  (target and irradiance at t-1, t-4, t-8 steps)
  • Rolling statistics (mean and std over 4-step and 8-step windows)
  • Clearness index  (measured irr / theoretical clear-sky irr — if available)
"""

import numpy as np
import pandas as pd

from src.config import RESAMPLE_RULE

# ─────────────────────────────────────────────────────────────────────────────
# TIME CYCLICAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode time cyclically so the model understands periodicity.

    Added columns:
      hour_sin, hour_cos  — position within the day  (24-h cycle)
      doy_sin,  doy_cos   — position within the year (365-day cycle)
      month               — raw integer month (1–12) for tree-based models
    """
    idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.DatetimeIndex(df.index)

    hour_rad = 2 * np.pi * idx.hour / 24.0
    doy_rad  = 2 * np.pi * idx.dayofyear / 365.0

    df = df.copy()
    df["hour_sin"] = np.sin(hour_rad)
    df["hour_cos"] = np.cos(hour_rad)
    df["doy_sin"]  = np.sin(doy_rad)
    df["doy_cos"]  = np.cos(doy_rad)
    df["month"]    = idx.month

    return df


# ─────────────────────────────────────────────────────────────────────────────
# LAG FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def add_lag_features(
    df: pd.DataFrame,
    columns: list[str],
    lags: list[int] = [1, 4, 8],
) -> pd.DataFrame:
    """
    Add lagged versions of specified columns.

    Parameters
    ----------
    df      : DataFrame with DatetimeIndex (already resampled)
    columns : list of column names to lag
    lags    : lag offsets in number of time-steps (e.g. 4 × 15-min = 1 h)

    Returns
    -------
    DataFrame with new columns named `{col}_lag{n}`.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ROLLING STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def add_rolling_features(
    df: pd.DataFrame,
    columns: list[str],
    windows: list[int] = [4, 8],
) -> pd.DataFrame:
    """
    Add rolling mean and rolling std for specified columns.

    Parameters
    ----------
    columns : column names to compute rolling stats on
    windows : window sizes in time-steps (min_periods=1 to avoid leading NaNs)

    Returns
    -------
    DataFrame with `{col}_roll{w}_mean` and `{col}_roll{w}_std` columns.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        for w in windows:
            roll = df[col].rolling(window=w, min_periods=1)
            df[f"{col}_roll{w}_mean"] = roll.mean()
            df[f"{col}_roll{w}_std"]  = roll.std().fillna(0)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLEARNESS INDEX  (optional — requires theoretical clear-sky values)
# ─────────────────────────────────────────────────────────────────────────────

def add_clearness_index(
    df: pd.DataFrame,
    irr_col: str = "irr_horiz",
    clearsky_col: str | None = None,
    latitude_deg: float = 43.45,    # Oakville, ON latitude
) -> pd.DataFrame:
    """
    Compute the clearness index (Kt = measured_irr / clear_sky_irr).

    If `clearsky_col` is provided and exists in df, use it directly.
    Otherwise compute a simple sinusoidal theoretical clear-sky approximation.

    Kt = 1.0 → perfectly clear sky
    Kt < 0.5 → overcast / cloudy conditions
    """
    df = df.copy()
    idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.DatetimeIndex(df.index)

    if clearsky_col and clearsky_col in df.columns:
        clear_sky = df[clearsky_col]
    else:
        # Simple clear-sky approximation:
        # GHI_clear ≈ 1367 * cos(zenith) * 0.75  (simplified)
        hour_angle = (idx.hour + idx.minute / 60) - 12   # –12 … +12
        decl = 23.45 * np.sin(np.radians(360 * (284 + idx.dayofyear) / 365))
        lat_r  = np.radians(latitude_deg)
        decl_r = np.radians(decl)
        cos_z  = (np.sin(lat_r) * np.sin(decl_r) +
                  np.cos(lat_r) * np.cos(decl_r) * np.cos(np.radians(15 * hour_angle)))
        cos_z  = np.maximum(cos_z, 0)        # clamp nighttime to 0
        clear_sky = pd.Series(1367 * cos_z * 0.75, index=idx)

    if irr_col in df.columns:
        kt = df[irr_col] / clear_sky.replace(0, np.nan)
        df["clearness_index"] = kt.clip(0, 1.5)   # cap at 1.5 (sensor outliers)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# FULL FEATURE ENGINEERING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(
    df: pd.DataFrame,
    target: str = "target",
    irr_col: str = "irr_horiz",
    lag_cols: list[str] | None = None,
    lags: list[int] = [1, 4, 8],
    roll_windows: list[int] = [4, 8],
    add_clearness: bool = True,
    latitude: float = 43.45,
) -> pd.DataFrame:
    """
    Master feature engineering function.

    Applies all transformations in the correct order:
      1. Time cyclical features
      2. Clearness index
      3. Lag features
      4. Rolling statistics
      5. Drop rows with any NaN created by lags (first `max(lags)` rows)

    Parameters
    ----------
    df             : clean, resampled DataFrame from preprocessing.py
    target         : name of the target column (kept but not lagged here)
    irr_col        : irradiance column name for clearness + rolling
    lag_cols       : columns to lag (defaults to [irr_col, target])
    lags           : lag offsets in time-steps
    roll_windows   : rolling window sizes in time-steps
    add_clearness  : whether to compute Kt
    latitude       : site latitude in degrees (for clear-sky model)

    Returns
    -------
    Feature-enriched DataFrame (NaN rows from lags dropped).
    """
    if lag_cols is None:
        lag_cols = [c for c in [irr_col, target] if c in df.columns]

    df = add_time_features(df)

    if add_clearness and irr_col in df.columns:
        df = add_clearness_index(df, irr_col=irr_col, latitude_deg=latitude)

    df = add_lag_features(df, columns=lag_cols, lags=lags)
    df = add_rolling_features(df, columns=[irr_col] if irr_col in df.columns else [],
                               windows=roll_windows)

    # Drop the initial rows where lags create NaNs
    max_lag = max(lags) if lags else 0
    df = df.iloc[max_lag:]
    df = df.dropna()

    print(f"✅ Feature engineering complete: {len(df):,} rows, "
          f"{len(df.columns)} columns (incl. target)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SEQUENCE BUILDER  (for LSTM / CNN — sliding window)
# ─────────────────────────────────────────────────────────────────────────────

def build_sequences(
    X: np.ndarray,
    y: np.ndarray,
    lookback: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create sliding-window sequences for sequence-to-one forecasting.

    Parameters
    ----------
    X        : 2-D array (n_samples, n_features) — already scaled
    y        : 1-D array (n_samples,) — target values
    lookback : number of past time steps in each input sequence

    Returns
    -------
    X_seq : shape (n_sequences, lookback, n_features)
    y_seq : shape (n_sequences,)  — the value at time t+1
    """
    X_seq, y_seq = [], []
    for i in range(lookback, len(X)):
        X_seq.append(X[i - lookback : i])   # past `lookback` timesteps
        y_seq.append(y[i])                   # next step value
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)
