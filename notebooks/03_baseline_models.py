# %% [markdown]
# # 03 - Baseline Models
# **Models**: Linear Regression, Random Forest, XGBoost
#
# Goals:
# 1. Train all three baseline models on the 15-min Oakville data
# 2. Evaluate on the held-out test set (RMSE, MAE, MAPE, R²)
# 3. Visualise actual vs predicted, residuals, feature importance
# 4. Save model artefacts and compile comparison table

# %% [markdown]
# ## 0. Setup

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path("..").resolve()))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from src.config       import TARGET, MODELS_DIR, RESULTS_DIR, FIGS_DIR, OAKVILLE_PARQUET
from src.data_loader  import load_and_standardise
from src.preprocessing import clean, split_chronological, get_xy, fit_scaler, apply_scaler
from src.features      import engineer_features
from src.evaluate      import compute_metrics, build_comparison_table, print_metrics
from src.visualize     import (
    plot_actual_vs_predicted, plot_scatter, plot_residuals,
    plot_feature_importance,
)
from src.models.baseline import LinearRegressionModel, RandomForestModel, XGBoostModel

for d in [MODELS_DIR, RESULTS_DIR, FIGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("✅ Setup complete")

# %% [markdown]
# ## 1. Load & Prepare Data

# %%
# Load from Parquet if available (fast), else raw CSV
if OAKVILLE_PARQUET.exists():
    print("📂 Loading from Parquet (fast path)...")
    df_clean = pd.read_parquet(OAKVILLE_PARQUET)
else:
    print("📥 Parquet not found — loading from CSV (run 02_preprocessing.py first)")
    df_raw   = load_and_standardise(site="oakville", verbose=True)
    df_feat  = engineer_features(df_raw, target=TARGET, irr_col="irr_horiz")
    df_clean = clean(df_feat, resample=True, filter_night=True)

print(f"Shape: {df_clean.shape}")

# %%
# Feature columns
candidate_features = [
    "irr_horiz", "amb_temp", "rh", "wind_speed", "pressure",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "month",
    "clearness_index",
    "irr_horiz_lag1", "irr_horiz_lag4", "irr_horiz_lag8",
    "target_lag1", "target_lag4",
    "irr_horiz_roll4_mean", "irr_horiz_roll4_std",
]
feature_cols = [c for c in candidate_features if c in df_clean.columns]

# Split
train_df, val_df, test_df = split_chronological(df_clean)
X_tr, y_tr = get_xy(train_df, feature_cols, TARGET)
X_va, y_va = get_xy(val_df,   feature_cols, TARGET)
X_te, y_te = get_xy(test_df,  feature_cols, TARGET)

# Scale
scaler_path = MODELS_DIR / "scaler.joblib"
if scaler_path.exists():
    scaler = joblib.load(scaler_path)
    print("📂 Loaded existing scaler")
else:
    scaler = fit_scaler(X_tr, save_path=scaler_path)

X_tr_sc = apply_scaler(X_tr, scaler)
X_va_sc = apply_scaler(X_va, scaler)
X_te_sc = apply_scaler(X_te, scaler)

print(f"\nTrain: {X_tr_sc.shape} | Val: {X_va_sc.shape} | Test: {X_te_sc.shape}")

# %% [markdown]
# ## 2. Linear Regression

# %%
lr_model = LinearRegressionModel(alpha=0.1)
lr_model.fit(X_tr_sc, y_tr)

y_pred_lr = lr_model.predict(X_te_sc)
metrics_lr = compute_metrics(y_te, y_pred_lr, label="Linear Regression")
print_metrics(metrics_lr)
lr_model.save()

# %%
fig = plot_actual_vs_predicted(y_te, y_pred_lr, label="Linear Regression",
                               save_path=FIGS_DIR / "03_lr_actual_vs_pred.png")
plt.show()

fig = plot_scatter(y_te, y_pred_lr, label="Linear Regression",
                   save_path=FIGS_DIR / "03_lr_scatter.png")
plt.show()

fig = plot_residuals(y_te, y_pred_lr, label="Linear Regression",
                     save_path=FIGS_DIR / "03_lr_residuals.png")
plt.show()

# %% [markdown]
# ## 3. Random Forest

# %%
rf_model = RandomForestModel(n_estimators=200)
rf_model.fit(X_tr_sc, y_tr)

y_pred_rf = rf_model.predict(X_te_sc)
metrics_rf = compute_metrics(y_te, y_pred_rf, label="Random Forest")
print_metrics(metrics_rf)
rf_model.save()

# %%
fig = plot_actual_vs_predicted(y_te, y_pred_rf, label="Random Forest",
                               save_path=FIGS_DIR / "03_rf_actual_vs_pred.png")
plt.show()

fig = plot_feature_importance(
    feature_names = X_tr_sc.columns.tolist(),
    importances   = rf_model.feature_importance(),
    label         = "Random Forest",
    save_path     = FIGS_DIR / "03_rf_feature_importance.png",
)
plt.show()

# %% [markdown]
# ## 4. XGBoost

# %%
xgb_model = XGBoostModel(n_estimators=300, learning_rate=0.05, max_depth=6)
xgb_model.fit(X_tr_sc, y_tr, X_val=X_va_sc, y_val=y_va)

y_pred_xgb = xgb_model.predict(X_te_sc)
metrics_xgb = compute_metrics(y_te, y_pred_xgb, label="XGBoost")
print_metrics(metrics_xgb)
xgb_model.save()

# %%
fig = plot_actual_vs_predicted(y_te, y_pred_xgb, label="XGBoost",
                               save_path=FIGS_DIR / "03_xgb_actual_vs_pred.png")
plt.show()

fig = plot_feature_importance(
    feature_names = X_tr_sc.columns.tolist(),
    importances   = xgb_model.feature_importance(),
    label         = "XGBoost",
    save_path     = FIGS_DIR / "03_xgb_feature_importance.png",
)
plt.show()

# %% [markdown]
# ## 5. Baseline Comparison Table

# %%
all_metrics = [metrics_lr, metrics_rf, metrics_xgb]
comparison  = build_comparison_table(all_metrics)

print("\n📊 Baseline Model Comparison:")
print(comparison.to_string())

comparison.to_csv(RESULTS_DIR / "03_baseline_comparison.csv")
print(f"\n💾 Saved → {RESULTS_DIR / '03_baseline_comparison.csv'}")

# %%
# Visual comparison
from src.visualize import plot_metrics_comparison

for metric in ["rmse", "mae", "mape"]:
    fig = plot_metrics_comparison(
        comparison, metric=metric,
        save_path=FIGS_DIR / f"03_baseline_{metric}_comparison.png",
    )
    plt.show()

# %% [markdown]
# ## 6. Error Analysis — Hourly & Seasonal

# %%
# Attach predictions to test index for temporal analysis
test_results = pd.DataFrame({
    "actual":     y_te.values,
    "lr_pred":    y_pred_lr,
    "rf_pred":    y_pred_rf,
    "xgb_pred":   y_pred_xgb,
    "lr_err":     np.abs(y_te.values - y_pred_lr),
    "rf_err":     np.abs(y_te.values - y_pred_rf),
    "xgb_err":    np.abs(y_te.values - y_pred_xgb),
}, index=y_te.index)

print("Mean absolute error by hour of day (XGBoost):")
hourly_err = test_results["xgb_err"].groupby(test_results.index.hour).mean()
print(hourly_err.to_string())

# %%
fig, ax = plt.subplots(figsize=(10, 4))
hourly_err.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Mean Absolute Error (W)")
ax.set_title("XGBoost — MAE by Hour of Day")
plt.tight_layout()
plt.savefig(FIGS_DIR / "03_xgb_hourly_error.png", dpi=150)
plt.show()

# %% [markdown]
# ## Summary

# %%
print("=" * 60)
print("BASELINE MODELS — SUMMARY")
print("=" * 60)
print(comparison[["rmse", "mae", "mape", "r2"]].to_string())
print("=" * 60)
print(f"\n🏆 Best baseline: {comparison['rmse'].idxmin()} "
      f"(RMSE = {comparison['rmse'].min():.2f} W)")
print("\n✅ Proceed to 04_deep_learning.py")
