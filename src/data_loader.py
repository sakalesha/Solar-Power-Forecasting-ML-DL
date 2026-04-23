"""
data_loader.py — Load raw FPV CSV files safely and efficiently.

Key challenges handled:
  1. Files are ~500 MB / ~1 M rows → use chunked reading.
  2. DAY column uses Julian date format YYYYDDD (e.g. 2022216 = day 216 of 2022).
  3. HOUR column is "HH:MM:SS" string.
  4. Sentinel values (32767, 32766, -99) must become NaN immediately on read.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.config import (
    OAKVILLE_CSV, ORLANDO_CSV,
    SENTINEL_VALUES, OAKVILLE_COLS, ORLANDO_COLS,
    ACTIVE_SITE,
)

# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _julian_to_datetime(day_series: pd.Series, hour_series: pd.Series) -> pd.Series:
    """
    Convert YYYYDDD + HH:MM:SS columns into a proper DatetimeIndex.

    Parameters
    ----------
    day_series  : int series, e.g. 2022216
    hour_series : str series, e.g. '13:45:00'
    # Example: 2022216 + 13:45:00 -> 2022-08-03 13:45:00

    Returns
    -------
    pd.Series of datetime64[ns]
    """
    year = (day_series // 1000).astype(str)
    doy  = (day_series  % 1000).astype(str).str.zfill(3)
    dt_str = year + "-" + doy + " " + hour_series.astype(str)
    return pd.to_datetime(dt_str, format="%Y-%j %H:%M:%S", errors="coerce")


def _replace_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Replace all known sentinel fill-values with NaN."""
    return df.replace(SENTINEL_VALUES, np.nan)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def load_site_csv(
    site: str = "oakville",
    chunksize: int = 100_000,
    nrows: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load a raw FPV site CSV into a single DataFrame with a DatetimeIndex.

    Parameters
    ----------
    site      : "oakville" | "orlando"
    chunksize : rows per chunk (reduce if RAM limited)
    nrows     : limit total rows loaded (useful for testing with small samples)
    verbose   : show tqdm progress bar

    Returns
    -------
    pd.DataFrame with DatetimeIndex, sentinels replaced by NaN,
    and a column 'site' added.
    """
    csv_path = OAKVILLE_CSV if site == "oakville" else ORLANDO_CSV

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found at {csv_path}.\n"
            "Please copy or symlink your raw data file there."
        )

    if verbose:
        print(f"📂 Loading {site} data from:\n   {csv_path}")
        file_mb = csv_path.stat().st_size / 1_048_576
        print(f"   File size: {file_mb:.0f} MB")

    reader = pd.read_csv(
        csv_path,
        chunksize=chunksize,
        index_col=0,        # first unnamed index column
        nrows=nrows,
        low_memory=False,
    )

    chunks = []
    total_rows = 0
    iterator = tqdm(reader, desc="Reading chunks", unit="chunk") if verbose else reader

    for chunk in iterator:
        # 1. Replace sentinels
        chunk = _replace_sentinels(chunk)

        # 2. Build DatetimeIndex from YYYYDDD + HH:MM:SS
        chunk["timestamp"] = _julian_to_datetime(chunk["DAY"], chunk["HOUR"])
        chunk = chunk.drop(columns=["DAY", "HOUR"])
        chunk = chunk.set_index("timestamp")
        chunk = chunk[~chunk.index.isna()]   # drop rows with unparseable dates

        # 3. Add site label
        chunk["site"] = site

        chunks.append(chunk)
        total_rows += len(chunk)

    df = pd.concat(chunks, axis=0)
    df = df.sort_index()

    if verbose:
        print(f"✅ Loaded {total_rows:,} rows | "
              f"date range: {df.index.min()} → {df.index.max()}")

    return df


def get_column_map(site: str = "oakville") -> dict:
    """Return the column-name mapping dict for a given site."""
    return OAKVILLE_COLS if site == "oakville" else ORLANDO_COLS


def rename_to_standard(df: pd.DataFrame, site: str = "oakville") -> pd.DataFrame:
    """
    Rename raw site-specific columns to standardised names defined in config.

    The renaming uses the *values* of the col-map dict as keys (raw names)
    and the *keys* as values (standard names), then selects only those columns
    that exist in the DataFrame.

    Parameters
    ----------
    df   : raw DataFrame from load_site_csv()
    site : "oakville" | "orlando"

    Returns
    -------
    DataFrame with standardised column names + 'site' + 'timestamp' index.
    """
    col_map = get_column_map(site)
    rename_dict = {v: k for k, v in col_map.items()}  # raw → standard

    available = {k: v for k, v in rename_dict.items() if k in df.columns}
    df = df.rename(columns=available)

    # Keep only standardised columns that exist
    keep = [v for v in col_map.keys() if v in df.columns] + ["site"]
    keep = [c for c in keep if c in df.columns]
    return df[keep]


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: one-shot load + rename
# ─────────────────────────────────────────────────────────────────────────────

def load_and_standardise(
    site: str = "oakville",
    chunksize: int = 100_000,
    nrows: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Full pipeline: load CSV → replace sentinels → parse dates → rename columns.

    Returns a clean DataFrame ready for preprocessing.
    """
    df = load_site_csv(site=site, chunksize=chunksize, nrows=nrows, verbose=verbose)
    df = rename_to_standard(df, site=site)
    return df
