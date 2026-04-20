# %% [markdown]
# # 01 - Exploratory Data Analysis (EDA)
# **Site**: Oakville, CA | **FPV Dataset** — OpenEI
#
# Goals:
# 1. Understand dataset shape, date range, and column types
# 2. Detect and quantify sentinel values (32767, 32766, -99)
# 3. Visualise time-series, daily profiles, and distributions
# 4. Compute and plot correlation matrix
# 5. Build intuition for feature selection

# %% [markdown]
# ## 0. Setup

# %%
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path("..").resolve()))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from src.config import OAKVILLE_CSV, SENTINEL_VALUES, TARGET, OAKVILLE_COLS
from src.data_loader import load_and_standardise, load_site_csv
from src.visualize import (
    plot_time_series, plot_daily_profile, plot_correlation_heatmap,
    plot_missing_data, plot_distribution,
)

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", "{:.3f}".format)

print("✅ Setup complete")

# %% [markdown]
# ## 1. Quick Peek at Raw File

# %%
# Read just 5 rows to understand schema WITHOUT sentinel replacement
df_peek = pd.read_csv(OAKVILLE_CSV, nrows=5, low_memory=False)
print(f"Shape: {df_peek.shape}")
print(f"Columns: {df_peek.columns.tolist()}")
df_peek.head()

# %%
# Read first 200 rows to check sentinel presence
df_raw_sample = pd.read_csv(OAKVILLE_CSV, nrows=200, index_col=0, low_memory=False)
print("Null counts in first 200 rows (before sentinel removal):")
print(df_raw_sample.isnull().sum())

# %%
# Count sentinel occurrences in sample
for s in SENTINEL_VALUES:
    count = (df_raw_sample == s).sum().sum()
    print(f"  Sentinel {s}: {count} occurrences in first 200 rows")

# %% [markdown]
# ## 2. Load Full Dataset (15-min sample for EDA speed)
#
# We load ~100K rows (nrows=100000) for EDA — this is fast.
# The full load happens in preprocessing.

# %%
# Load full data with sentinel replacement and date parsing
df = load_and_standardise(site="oakville", nrows=200_000, verbose=True)
print(f"\nShape: {df.shape}")
print(f"Date range: {df.index.min()} → {df.index.max()}")
df.head(3)

# %%
print("\n━━━ Data Types ━━━")
print(df.dtypes)

print("\n━━━ Summary Statistics ━━━")
df.describe().T

# %% [markdown]
# ## 3. Missing Data Analysis

# %%
null_pct = df.isnull().mean().sort_values(ascending=False) * 100
print("Missing data (%) per column:")
print(null_pct[null_pct > 0].to_string())

# %%
fig = plot_missing_data(df, figsize=(12, 4),
                        save_path=Path("../outputs/figures/01_missing_data.png"))
plt.show()

# %% [markdown]
# ## 4. Target Variable — INVPWR Distribution

# %%
print(f"INVPWR stats:")
print(df[TARGET].describe())
print(f"\nZero values: {(df[TARGET] == 0).sum():,}")
print(f"Negative values: {(df[TARGET] < 0).sum():,}")
print(f"NaN values: {df[TARGET].isna().sum():,}")

# %%
fig = plot_distribution(df, TARGET,
                        save_path=Path("../outputs/figures/01_invpwr_distribution.png"))
plt.show()

# %% [markdown]
# ## 5. Time-Series Visualisation

# %%
# Subsample for plotting speed
plot_cols = ["target", "irr_horiz", "amb_temp"]
available = [c for c in plot_cols if c in df.columns]

fig = plot_time_series(
    df.resample("1h").mean(),   # downsample to hourly for visual clarity
    columns=available,
    title="Oakville CA — Hourly Overview",
    save_path=Path("../outputs/figures/01_timeseries_overview.png"),
)
plt.show()

# %% [markdown]
# ## 6. Daily Profile (average over hour-of-day)

# %%
fig = plot_daily_profile(
    df, col=TARGET,
    title="Oakville — Average Hourly INVPWR Profile",
    save_path=Path("../outputs/figures/01_daily_profile_invpwr.png"),
)
plt.show()

# %%
if "irr_horiz" in df.columns:
    fig = plot_daily_profile(
        df, col="irr_horiz",
        title="Oakville — Average Hourly Irradiance Profile",
        save_path=Path("../outputs/figures/01_daily_profile_irr.png"),
    )
    plt.show()

# %% [markdown]
# ## 7. Seasonal Analysis

# %%
df_day = df.resample("D").mean()

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
axes[0].plot(df_day.index, df_day[TARGET], lw=0.9, color="steelblue")
axes[0].set_ylabel("INVPWR (W)")
axes[0].set_title("Daily Mean INVPWR — Seasonal Pattern")

if "irr_horiz" in df_day.columns:
    axes[1].plot(df_day.index, df_day["irr_horiz"], lw=0.9, color="darkorange")
    axes[1].set_ylabel("Irradiance (W/m²)")
    axes[1].set_title("Daily Mean Horizontal Irradiance")

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
plt.tight_layout()
plt.savefig("../outputs/figures/01_seasonal_analysis.png", dpi=150)
plt.show()

# %% [markdown]
# ## 8. Correlation Matrix

# %%
# Keep only numeric columns
df_numeric = df.select_dtypes("number")

fig = plot_correlation_heatmap(
    df_numeric,
    figsize=(12, 10),
    save_path=Path("../outputs/figures/01_correlation_heatmap.png"),
)
plt.show()

# Print top correlations with INVPWR
print("\nTop correlations with INVPWR:")
corr_target = df_numeric.corr()[TARGET].sort_values(ascending=False)
print(corr_target.to_string())

# %% [markdown]
# ## 9. Irradiance vs Power Scatter

# %%
if "irr_horiz" in df.columns:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["irr_horiz"], df[TARGET], s=2, alpha=0.2, color="steelblue")
    ax.set_xlabel("Horizontal Irradiance (W/m²)")
    ax.set_ylabel("INVPWR (W)")
    ax.set_title("Irradiance vs Inverter Power — Oakville CA")
    plt.tight_layout()
    plt.savefig("../outputs/figures/01_irr_vs_invpwr.png", dpi=150)
    plt.show()

# %% [markdown]
# ## 10. Key EDA Findings Summary

# %%
print("=" * 60)
print("EDA SUMMARY — OAKVILLE CA")
print("=" * 60)
print(f"  Total rows loaded   : {len(df):,}")
print(f"  Date range          : {df.index.min()} → {df.index.max()}")
print(f"  Columns             : {df.columns.tolist()}")
print(f"  INVPWR mean (W)     : {df[TARGET].mean():.1f}")
print(f"  INVPWR max (W)      : {df[TARGET].max():.1f}")
print(f"  Missing INVPWR (%)  : {df[TARGET].isna().mean()*100:.1f}%")

if "irr_horiz" in df.columns:
    corr = df["irr_horiz"].corr(df[TARGET])
    print(f"  Irr–Power Pearson r : {corr:.3f}")
print("=" * 60)
print("\n📁 Figures saved to outputs/figures/")
