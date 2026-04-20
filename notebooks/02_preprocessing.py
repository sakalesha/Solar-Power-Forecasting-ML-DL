# %% [markdown]
# # 02 - Preprocessing & Feature Engineering
# **Site**: Oakville, CA
#
# Goals:
# 1. Run the full cleaning pipeline (sentinel → NaN, resample to 15-min)
# 2. Filter nighttime records
# 3. Engineer time, lag, and rolling features
# 4. Chronological train/val/test split
# 5. Fit StandardScaler on training set
# 6. Save processed data to Parquet for fast downstream loading

# %% [markdown]
# ## 0. Setup

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path("..").resolve()))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import TARGET, FEATURE_COLS, OAKVILLE_PARQUET, DATA_PROC
from src.data_loader      import load_and_standardise
from src.preprocessing    import clean, split_chronological, get_xy, fit_scaler, apply_scaler
from src.features         import engineer_features, add_time_features

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 40)
print("✅ Setup complete")

# %% [markdown]
# ## 1. Load Raw Data

# %%
print("📥 Loading Oakville raw data...")
df_raw = load_and_standardise(site="oakville", verbose=True)
print(f"\nRaw shape : {df_raw.shape}")
print(f"Columns   : {df_raw.columns.tolist()}")
df_raw.head(3)

# %% [markdown]
# ## 2. Feature Engineering BEFORE Resampling
#
# We add time features and compute lags/rolling stats BEFORE we clean,
# because we need the 1-min index for correct cyclical features.

# %%
print("🔧 Engineering features on raw 1-min data...")
df_feat = engineer_features(
    df_raw,
    target       = TARGET,
    irr_col      = "irr_horiz",
    lags         = [1, 4, 8],
    roll_windows = [4, 8],
    add_clearness= True,
    latitude     = 43.45,       # Oakville, Ontario
)
print(f"\nAfter feature engineering: {df_feat.shape}")
df_feat.head(3)

# %%
print("\nAll columns after feature engineering:")
print(df_feat.columns.tolist())

# %% [markdown]
# ## 3. Clean Pipeline (resample + nighttime filter)

# %%
df_clean = clean(
    df_feat,
    resample     = True,    # 1-min → 15-min
    filter_night = True,    # remove rows where irr < 5 W/m²
)
print(f"\nAfter cleaning: {df_clean.shape}")

# %%
# Verify index is monotonic and no date gaps > 1 day
is_monotonic = df_clean.index.is_monotonic_increasing
time_diffs   = pd.Series(df_clean.index).diff().dropna()
max_gap      = time_diffs.max()

print(f"Index monotonic : {is_monotonic}")
print(f"Max time gap    : {max_gap}")
print(f"Date range      : {df_clean.index[0]} → {df_clean.index[-1]}")

# %% [markdown]
# ## 4. Determine Feature Columns

# %%
# Build list of available features (exclude target)
candidate_features = [
    "irr_horiz", "amb_temp", "rh", "wind_speed", "pressure",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "month",
    "clearness_index",
    "irr_horiz_lag1", "irr_horiz_lag4", "irr_horiz_lag8",
    "target_lag1", "target_lag4",
    "irr_horiz_roll4_mean", "irr_horiz_roll4_std",
    "irr_horiz_roll8_mean", "irr_horiz_roll8_std",
]

feature_cols = [c for c in candidate_features if c in df_clean.columns]
print(f"Features available ({len(feature_cols)}):")
for c in feature_cols:
    print(f"  • {c}")

# %% [markdown]
# ## 5. Train / Val / Test Split

# %%
train_df, val_df, test_df = split_chronological(df_clean)

print(f"\nSplit summary:")
print(f"  Train : {len(train_df):>7,} rows | {train_df.index[0]} → {train_df.index[-1]}")
print(f"  Val   : {len(val_df):>7,} rows | {val_df.index[0]} → {val_df.index[-1]}")
print(f"  Test  : {len(test_df):>7,} rows | {test_df.index[0]} → {test_df.index[-1]}")

# %%
# Visualise split
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(train_df.index, train_df[TARGET], lw=0.8, label="Train", color="steelblue", alpha=0.8)
ax.plot(val_df.index,   val_df[TARGET],   lw=0.8, label="Val",   color="orange",    alpha=0.8)
ax.plot(test_df.index,  test_df[TARGET],  lw=0.8, label="Test",  color="tomato",    alpha=0.8)
ax.set_ylabel("INVPWR (W)")
ax.set_title("Train / Val / Test Split — INVPWR")
ax.legend()
plt.tight_layout()
plt.savefig("../outputs/figures/02_split_overview.png", dpi=150)
plt.show()

# %% [markdown]
# ## 6. Extract X, y and Scale

# %%
X_train, y_train = get_xy(train_df, feature_cols, TARGET)
X_val,   y_val   = get_xy(val_df,   feature_cols, TARGET)
X_test,  y_test  = get_xy(test_df,  feature_cols, TARGET)

print(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
print(f"X_val  : {X_val.shape}   | y_val  : {y_val.shape}")
print(f"X_test : {X_test.shape}  | y_test : {y_test.shape}")

# %%
from src.config import MODELS_DIR

scaler = fit_scaler(
    X_train,
    scaler_type = "standard",
    save_path   = MODELS_DIR / "scaler.joblib",
)

X_train_sc = apply_scaler(X_train, scaler)
X_val_sc   = apply_scaler(X_val,   scaler)
X_test_sc  = apply_scaler(X_test,  scaler)

print(f"\nScaled X_train — mean≈0: {X_train_sc.mean().mean():.4f}")
print(f"Scaled X_train — std≈1 : {X_train_sc.std().mean():.4f}")

# %% [markdown]
# ## 7. Feature Distribution Comparison

# %%
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
axes = axes.flatten()

for ax, col in zip(axes, feature_cols[:9]):
    ax.hist(X_train_sc[col].dropna(), bins=40, color="steelblue", alpha=0.7,
            label="train (scaled)")
    ax.set_title(col, fontsize=8)
    ax.set_xlabel("Scaled value")
    ax.set_ylabel("Count")

plt.suptitle("Feature Distributions (Scaled Training Set)", fontweight="bold")
plt.tight_layout()
plt.savefig("../outputs/figures/02_feature_distributions.png", dpi=150)
plt.show()

# %% [markdown]
# ## 8. Save Processed Data

# %%
# Save the full cleaned + featured dataset as Parquet for fast re-loading
DATA_PROC.mkdir(parents=True, exist_ok=True)
df_clean.to_parquet(OAKVILLE_PARQUET, compression="snappy")
print(f"💾 Saved processed data → {OAKVILLE_PARQUET}")
print(f"   File size: {OAKVILLE_PARQUET.stat().st_size / 1024:.0f} KB")

# %% [markdown]
# ## 9. Summary

# %%
print("=" * 60)
print("PREPROCESSING SUMMARY")
print("=" * 60)
print(f"  Raw rows         : {len(df_raw):,}")
print(f"  After cleaning   : {len(df_clean):,}")
print(f"  Feature count    : {len(feature_cols)}")
print(f"  Train size       : {len(X_train):,}")
print(f"  Val size         : {len(X_val):,}")
print(f"  Test size        : {len(X_test):,}")
print(f"  Scaler           : StandardScaler (fit on train only)")
print("=" * 60)
print("\n✅ Ready for modelling — proceed to 03_baseline_models.py")
