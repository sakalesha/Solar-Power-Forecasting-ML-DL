"""
config.py — Central configuration for the Solar Power Forecasting project.

All paths, column names, sentinel values, hyperparameters and model settings
live here so every other module imports from a single source of truth.
"""

from pathlib import Path

import os

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent          # project root
DATA_RAW   = ROOT / "data" / "raw"
DATA_PROC  = ROOT / "data" / "processed"
DOCS       = ROOT / "docs"
OUTPUTS    = ROOT / "outputs"
MODELS_DIR = OUTPUTS / "models"
FIGS_DIR   = OUTPUTS / "figures"
RESULTS_DIR= OUTPUTS / "results"

# --- Google Colab / GDrive Compatibility ---
# If running in Colab, raw CSVs can be loaded directly from Google Drive
COLAB_DRIVE_DOCS = Path("/content/drive/MyDrive/Solar-Power-Forecasting-Data")
if "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ:
    if COLAB_DRIVE_DOCS.exists():
        DOCS = COLAB_DRIVE_DOCS

# Source CSVs (in docs/ — large files kept there, not in data/raw)
OAKVILLE_CSV   = DOCS / "FPV_Oakville_CA_data.csv"
ORLANDO_CSV    = DOCS / "FPV_Orlando_FL_data.csv"
METAFILE_XLSX  = DOCS / "FPV_metafile_mm.xlsx"


# Processed parquet outputs
OAKVILLE_PARQUET = DATA_PROC / "oakville_15min.parquet"

# ─────────────────────────────────────────────────────────────────────────────
# SITE SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
ACTIVE_SITE = "oakville"   # "oakville" | "orlando"

# ─────────────────────────────────────────────────────────────────────────────
# DATA SENTINELS  (values that represent sensor errors / missing data)
# ─────────────────────────────────────────────────────────────────────────────
SENTINEL_VALUES = [32767.0, 32766.0, -99.0]

# ─────────────────────────────────────────────────────────────────────────────
# TARGET VARIABLE
# ─────────────────────────────────────────────────────────────────────────────
TARGET = "INVPWR"          # Inverter AC Power Output (Watts)

# ─────────────────────────────────────────────────────────────────────────────
# COLUMN NAMES  (common across both sites — superset)
# ─────────────────────────────────────────────────────────────────────────────

# Oakville CA schema
OAKVILLE_COLS = {
    "irr_horiz":  "FHZIRR",    # Horizontal irradiance (W/m²)
    "irr_plane":  "FPAIRR",    # Plane-of-array irradiance (W/m²)
    "amb_temp":   "FPVDBT",    # Ambient dry-bulb temperature (°C)
    "rh":         "FPV_RH",    # Relative humidity (%)
    "wind_speed": "FWINDA",    # Average wind speed (m/s)
    "wind_max":   "FWINDM",    # Max wind speed (m/s)
    "pressure":   "FRAIRP",    # Atmospheric pressure (hPa)
    "rain":       "FPRECT",    # Precipitation total (mm)
    "target":     "INVPWR",    # AC inverter power (W)
    "mppt1":      "MPPT1P",    # MPPT string 1 power (W)
    "mppt2":      "MPPT2P",    # MPPT string 2 power (W)
    "voltage":    "GRIDVT",    # Grid voltage (V)
}

# Orlando FL schema  (slightly different column names)
ORLANDO_COLS = {
    "irr_horiz":  "FPHIRR",   # Floating PV horizontal irradiance
    "amb_temp":   "FAMBTM",   # Ambient temperature (°C)
    "rh":         "FPV_RH",
    "wind_speed": "FWINDS",
    "wind_max":   "FWINDM",
    "pressure":   "FVPRES",
    "target":     "INVPWR",
}

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SELECTION  (after engineering)
# ─────────────────────────────────────────────────────────────────────────────
# These are the final features fed to models (after EDA confirms importance)
FEATURE_COLS = [
    # Raw meteorological
    "irr_horiz",        # ← mapped from site-specific col
    "amb_temp",
    "rh",
    "wind_speed",
    "pressure",
    # Time-cyclical
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    # Lag & rolling (added by features.py)
    "irr_lag1",
    "irr_lag4",         # 4 × 15-min = 1 hour lag
    "irr_lag8",         # 2-hour lag
    "target_lag1",
    "target_lag4",
    "irr_roll4_mean",
    "irr_roll4_std",
]

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
RESAMPLE_RULE = "15min"       # Aggregate 1-min → 15-min
IRRADIANCE_THRESHOLD = 5.0    # W/m² — rows below this = nighttime (filtered)

# Train / Validation / Test split fractions (chronological)
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15

# Scaler: "standard" (zero-mean unit-variance) | "minmax" (0–1)
SCALER_TYPE = "standard"

# ─────────────────────────────────────────────────────────────────────────────
# BASELINE MODELS
# ─────────────────────────────────────────────────────────────────────────────
RF_PARAMS = dict(
    n_estimators   = 200,
    max_depth      = None,
    min_samples_leaf = 2,
    n_jobs         = -1,
    random_state   = 42,
)

XGB_PARAMS = dict(
    n_estimators   = 300,
    max_depth      = 6,
    learning_rate  = 0.05,
    subsample      = 0.8,
    colsample_bytree = 0.8,
    random_state   = 42,
    n_jobs         = -1,
)

SVR_PARAMS = dict(
    kernel = "rbf",
    C      = 10.0,
    gamma  = "scale",
    epsilon= 0.1,
)

# ─────────────────────────────────────────────────────────────────────────────
# DEEP LEARNING
# ─────────────────────────────────────────────────────────────────────────────
LOOKBACK      = 16            # 16 × 15-min = 4-hour history window
FORECAST_STEP = 1             # 1-step-ahead = next 15 minutes
BATCH_SIZE    = 256
EPOCHS        = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-5
PATIENCE      = 10            # Early stopping patience (epochs)
DEVICE        = "cpu"         # "cuda" if GPU available

LSTM_PARAMS = dict(
    hidden_size   = 64,
    num_layers    = 3,
    dropout       = 0.2,
)

CNN_PARAMS = dict(
    num_filters   = [64, 128, 64],
    kernel_sizes  = [3, 3, 3],
    dropout       = 0.2,
)

CNN_LSTM_PARAMS = dict(
    cnn_filters   = 64,
    cnn_kernel    = 3,
    lstm_hidden   = 64,
    lstm_layers   = 2,
    dropout       = 0.2,
)

# ─────────────────────────────────────────────────────────────────────────────
# RANDOM SEED  (reproducibility)
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
