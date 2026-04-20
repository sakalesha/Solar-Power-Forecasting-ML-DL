# %% [markdown]
# # 04 - Deep Learning Models
# **Models**: LSTM, 1D-CNN, CNN-LSTM (PyTorch)
#
# Goals:
# 1. Build sequence input tensors (sliding window = 16 × 15-min = 4 hours)
# 2. Train LSTM, 1D-CNN, CNN-LSTM with early stopping
# 3. Plot training loss curves
# 4. Evaluate all DL models on the test set
# 5. Add DL results to the comparison table

# %% [markdown]
# ## 0. Setup

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path("..").resolve()))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import joblib

from src.config import (
    TARGET, MODELS_DIR, RESULTS_DIR, FIGS_DIR, OAKVILLE_PARQUET,
    LOOKBACK, BATCH_SIZE, EPOCHS,
)
from src.data_loader   import load_and_standardise
from src.preprocessing import clean, split_chronological, get_xy, fit_scaler, apply_scaler
from src.features      import engineer_features, build_sequences
from src.evaluate      import compute_metrics, build_comparison_table, print_metrics
from src.visualize     import plot_actual_vs_predicted, plot_scatter, plot_training_history
from src.models.deep_learning import (
    LSTMForecaster, CNNForecaster, CNNLSTMForecaster,
    Trainer, make_dataloader,
)

for d in [MODELS_DIR, RESULTS_DIR, FIGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device    : {DEVICE}")

# %% [markdown]
# ## 1. Load & Prepare Data (same as Notebook 03)

# %%
if OAKVILLE_PARQUET.exists():
    df_clean = pd.read_parquet(OAKVILLE_PARQUET)
    print(f"📂 Loaded from Parquet: {df_clean.shape}")
else:
    df_raw   = load_and_standardise(site="oakville", verbose=True)
    df_feat  = engineer_features(df_raw, target=TARGET, irr_col="irr_horiz")
    df_clean = clean(df_feat, resample=True, filter_night=True)

candidate_features = [
    "irr_horiz", "amb_temp", "rh", "wind_speed", "pressure",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "month",
    "clearness_index",
    "irr_horiz_lag1", "irr_horiz_lag4", "irr_horiz_lag8",
    "target_lag1", "target_lag4",
    "irr_horiz_roll4_mean", "irr_horiz_roll4_std",
]
feature_cols = [c for c in candidate_features if c in df_clean.columns]
print(f"Features ({len(feature_cols)}): {feature_cols}")

train_df, val_df, test_df = split_chronological(df_clean)
X_tr, y_tr = get_xy(train_df, feature_cols, TARGET)
X_va, y_va = get_xy(val_df,   feature_cols, TARGET)
X_te, y_te = get_xy(test_df,  feature_cols, TARGET)

scaler_path = MODELS_DIR / "scaler.joblib"
scaler = joblib.load(scaler_path) if scaler_path.exists() else fit_scaler(X_tr, save_path=scaler_path)

X_tr_sc = apply_scaler(X_tr, scaler)
X_va_sc = apply_scaler(X_va, scaler)
X_te_sc = apply_scaler(X_te, scaler)

print(f"Shapes — Train: {X_tr_sc.shape} | Val: {X_va_sc.shape} | Test: {X_te_sc.shape}")

# %% [markdown]
# ## 2. Build Sequence Arrays

# %%
print(f"\nBuilding sequences (lookback={LOOKBACK} steps = {LOOKBACK*15} min)...")

X_tr_seq, y_tr_seq = build_sequences(X_tr_sc.values, y_tr.values, LOOKBACK)
X_va_seq, y_va_seq = build_sequences(X_va_sc.values, y_va.values, LOOKBACK)
X_te_seq, y_te_seq = build_sequences(X_te_sc.values, y_te.values, LOOKBACK)

print(f"X_train_seq : {X_tr_seq.shape}  (n_samples, lookback, n_features)")
print(f"X_val_seq   : {X_va_seq.shape}")
print(f"X_test_seq  : {X_te_seq.shape}")

n_features = X_tr_sc.shape[1]
print(f"\nModel input: ({LOOKBACK}, {n_features})")

# %%
# DataLoaders
tr_loader = make_dataloader(X_tr_seq, y_tr_seq, batch_size=BATCH_SIZE, shuffle=True)
va_loader = make_dataloader(X_va_seq, y_va_seq, batch_size=BATCH_SIZE, shuffle=False)
te_loader = make_dataloader(X_te_seq, y_te_seq, batch_size=BATCH_SIZE, shuffle=False)

# %% [markdown]
# ## 3. LSTM

# %%
lstm_model = LSTMForecaster(
    n_features  = n_features,
    hidden_size = 64,
    num_layers  = 3,
    dropout     = 0.2,
)
print(lstm_model)
print(f"Parameters: {sum(p.numel() for p in lstm_model.parameters() if p.requires_grad):,}")

# %%
print("\n🔵 Training LSTM...")
lstm_trainer = Trainer(lstm_model, lr=1e-3, patience=10, device=DEVICE)
lstm_trainer.fit(tr_loader, va_loader, epochs=EPOCHS, verbose=True)
lstm_trainer.save_model(MODELS_DIR / "lstm_best.pt")

# %%
# Loss curve
fig = plot_training_history(
    lstm_trainer.train_losses, lstm_trainer.val_losses,
    label="LSTM",
    save_path=FIGS_DIR / "04_lstm_training_history.png",
)
plt.show()

# %%
# Evaluate
y_pred_lstm = lstm_trainer.predict(te_loader)
metrics_lstm = compute_metrics(y_te_seq, y_pred_lstm, label="LSTM")
print_metrics(metrics_lstm)

# %%
fig = plot_actual_vs_predicted(y_te_seq, y_pred_lstm, label="LSTM",
                               save_path=FIGS_DIR / "04_lstm_actual_vs_pred.png")
plt.show()
fig = plot_scatter(y_te_seq, y_pred_lstm, label="LSTM",
                   save_path=FIGS_DIR / "04_lstm_scatter.png")
plt.show()

# %% [markdown]
# ## 4. 1D-CNN

# %%
cnn_model = CNNForecaster(
    n_features  = n_features,
    lookback    = LOOKBACK,
    num_filters = [64, 128, 64],
    kernel_sizes= [3, 3, 3],
    dropout     = 0.2,
)
print(cnn_model)
print(f"Parameters: {sum(p.numel() for p in cnn_model.parameters() if p.requires_grad):,}")

# %%
print("\n🟠 Training 1D-CNN...")
cnn_trainer = Trainer(cnn_model, lr=1e-3, patience=10, device=DEVICE)
cnn_trainer.fit(tr_loader, va_loader, epochs=EPOCHS, verbose=True)
cnn_trainer.save_model(MODELS_DIR / "cnn_best.pt")

# %%
fig = plot_training_history(
    cnn_trainer.train_losses, cnn_trainer.val_losses,
    label="1D-CNN",
    save_path=FIGS_DIR / "04_cnn_training_history.png",
)
plt.show()

# %%
y_pred_cnn = cnn_trainer.predict(te_loader)
metrics_cnn = compute_metrics(y_te_seq, y_pred_cnn, label="1D-CNN")
print_metrics(metrics_cnn)

fig = plot_actual_vs_predicted(y_te_seq, y_pred_cnn, label="1D-CNN",
                               save_path=FIGS_DIR / "04_cnn_actual_vs_pred.png")
plt.show()

# %% [markdown]
# ## 5. CNN-LSTM (Hybrid)

# %%
cnn_lstm_model = CNNLSTMForecaster(
    n_features  = n_features,
    lookback    = LOOKBACK,
    cnn_filters = 64,
    cnn_kernel  = 3,
    lstm_hidden = 64,
    lstm_layers = 2,
    dropout     = 0.2,
)
print(cnn_lstm_model)
print(f"Parameters: {sum(p.numel() for p in cnn_lstm_model.parameters() if p.requires_grad):,}")

# %%
print("\n🟢 Training CNN-LSTM...")
cnn_lstm_trainer = Trainer(cnn_lstm_model, lr=1e-3, patience=10, device=DEVICE)
cnn_lstm_trainer.fit(tr_loader, va_loader, epochs=EPOCHS, verbose=True)
cnn_lstm_trainer.save_model(MODELS_DIR / "cnn_lstm_best.pt")

# %%
fig = plot_training_history(
    cnn_lstm_trainer.train_losses, cnn_lstm_trainer.val_losses,
    label="CNN-LSTM",
    save_path=FIGS_DIR / "04_cnn_lstm_training_history.png",
)
plt.show()

# %%
y_pred_cnn_lstm = cnn_lstm_trainer.predict(te_loader)
metrics_cnn_lstm = compute_metrics(y_te_seq, y_pred_cnn_lstm, label="CNN-LSTM")
print_metrics(metrics_cnn_lstm)

fig = plot_actual_vs_predicted(y_te_seq, y_pred_cnn_lstm, label="CNN-LSTM",
                               save_path=FIGS_DIR / "04_cnn_lstm_actual_vs_pred.png")
plt.show()

# %% [markdown]
# ## 6. Deep Learning Comparison

# %%
dl_metrics = [metrics_lstm, metrics_cnn, metrics_cnn_lstm]
dl_table   = build_comparison_table(dl_metrics)

print("\n📊 Deep Learning Model Comparison:")
print(dl_table.to_string())
dl_table.to_csv(RESULTS_DIR / "04_dl_comparison.csv")
print(f"💾 Saved → {RESULTS_DIR / '04_dl_comparison.csv'}")

# %%
# Save loss histories
for name, trainer in [("lstm", lstm_trainer), ("cnn", cnn_trainer), ("cnn_lstm", cnn_lstm_trainer)]:
    pd.DataFrame({
        "train_loss": trainer.train_losses,
        "val_loss":   trainer.val_losses,
    }).to_csv(RESULTS_DIR / f"04_{name}_loss_history.csv", index=False)

print("✅ All DL models trained and evaluated. Proceed to 05_comparison.py")
