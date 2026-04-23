import json
import os
import re
from pathlib import Path

def create_notebook():
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    def add_markdown(text):
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.split("\n")]
        })

    def add_code(text):
        # Remove any src imports to make it self-contained
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            if re.match(r'^\s*from\s+src(\..*)?\s+import\s+.*', line) or re.match(r'^\s*import\s+src(\..*)?', line):
                continue
            cleaned_lines.append(line + "\n")
        
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": cleaned_lines
        })

    # Read all source files
    root = Path(".")
    
    add_markdown("# Solar Power Forecasting - Full Machine Learning Pipeline\nThis notebook is entirely self-contained. It includes configuration, data loading, preprocessing, model definitions (Baseline & Deep Learning), and the generation of paper assets.")

    def add_module(title, filepath):
        path = root / filepath
        if path.exists():
            add_markdown(f"## {title}")
            add_code(path.read_text(encoding='utf-8'))
        else:
            print(f"Warning: {filepath} not found.")

    add_module("1. Configuration", "src/config.py")
    add_module("2. Data Loader", "src/data_loader.py")
    add_module("3. Preprocessing", "src/preprocessing.py")
    add_module("4. Feature Engineering", "src/features.py")
    add_module("5. Evaluation Metrics", "src/evaluate.py")
    add_module("6. Visualization Utilities", "src/visualize.py")
    add_module("7. Baseline Models (Scikit-Learn / XGBoost)", "src/models/baseline.py")
    add_module("8. Deep Learning Models (PyTorch)", "src/models/deep_learning.py")
    add_module("9. Paper Assets Generation", "src/generate_paper_assets.py")

    # Now add the execution pipeline
    add_markdown("## 10. Execution Pipeline\nThis section executes the full pipeline using the components defined above.")
    
    execution_code = """
import warnings
warnings.filterwarnings('ignore')

# 1. Setup paths
for d in [MODELS_DIR, RESULTS_DIR, FIGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
print("✅ Setup complete")

# 2. Load & Prepare Data
print("\\n--- Data Loading & Preprocessing ---")
if OAKVILLE_PARQUET.exists():
    print("📂 Loading from Parquet (fast path)...")
    df_clean = pd.read_parquet(OAKVILLE_PARQUET)
else:
    print("📥 Loading from CSV...")
    df_raw = load_and_standardise(site="oakville", verbose=True)
    df_feat = engineer_features(df_raw, target=TARGET, irr_col="irr_horiz")
    df_clean = clean(df_feat, resample=True, filter_night=True)

print(f"Shape: {df_clean.shape}")

# 3. Features & Splitting
feature_cols = [c for c in FEATURE_COLS if c in df_clean.columns]
train_df, val_df, test_df = split_chronological(df_clean)
X_tr, y_tr = get_xy(train_df, feature_cols, TARGET)
X_va, y_va = get_xy(val_df, feature_cols, TARGET)
X_te, y_te = get_xy(test_df, feature_cols, TARGET)

scaler_path = MODELS_DIR / "scaler.joblib"
scaler = fit_scaler(X_tr, save_path=scaler_path)

X_tr_sc = apply_scaler(X_tr, scaler)
X_va_sc = apply_scaler(X_va, scaler)
X_te_sc = apply_scaler(X_te, scaler)
print(f"Train: {X_tr_sc.shape} | Val: {X_va_sc.shape} | Test: {X_te_sc.shape}")

# 4. Train Baseline Models
print("\\n--- Baseline Models ---")
lr_model = LinearRegressionModel(alpha=0.1)
lr_model.fit(X_tr_sc, y_tr)
y_pred_lr = lr_model.predict(X_te_sc)
metrics_lr = compute_metrics(y_te, y_pred_lr, label="Linear Regression")

rf_model = RandomForestModel(n_estimators=50) # Reduced for notebook speed
rf_model.fit(X_tr_sc, y_tr)
y_pred_rf = rf_model.predict(X_te_sc)
metrics_rf = compute_metrics(y_te, y_pred_rf, label="Random Forest")
rf_model.save()

xgb_model = XGBoostModel(n_estimators=100, learning_rate=0.05, max_depth=6)
xgb_model.fit(X_tr_sc, y_tr, X_val=X_va_sc, y_val=y_va)
y_pred_xgb = xgb_model.predict(X_te_sc)
metrics_xgb = compute_metrics(y_te, y_pred_xgb, label="XGBoost")
xgb_model.save()

baseline_comparison = build_comparison_table([metrics_lr, metrics_rf, metrics_xgb])
print("\\nBaseline Model Comparison:")
print(baseline_comparison)

# 5. Build Sequences for Deep Learning
print("\\n--- Deep Learning Models ---")
X_tr_seq, y_tr_seq = build_sequences(X_tr_sc.values, y_tr.values, LOOKBACK)
X_va_seq, y_va_seq = build_sequences(X_va_sc.values, y_va.values, LOOKBACK)
X_te_seq, y_te_seq = build_sequences(X_te_sc.values, y_te.values, LOOKBACK)

tr_loader = make_dataloader(X_tr_seq, y_tr_seq, batch_size=BATCH_SIZE, shuffle=True)
va_loader = make_dataloader(X_va_seq, y_va_seq, batch_size=BATCH_SIZE, shuffle=False)
te_loader = make_dataloader(X_te_seq, y_te_seq, batch_size=BATCH_SIZE, shuffle=False)
n_features = X_tr_sc.shape[1]

# 6. Train Deep Learning Models (Fewer epochs for demo)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

lstm_model = LSTMForecaster(n_features=n_features, hidden_size=64, num_layers=2, dropout=0.2)
lstm_trainer = Trainer(lstm_model, lr=1e-3, patience=5, device=DEVICE)
lstm_trainer.fit(tr_loader, va_loader, epochs=10, verbose=True)
y_pred_lstm = lstm_trainer.predict(te_loader)
metrics_lstm = compute_metrics(y_te_seq, y_pred_lstm, label="LSTM")

cnn_model = CNNForecaster(n_features=n_features, lookback=LOOKBACK, num_filters=[64, 64], kernel_sizes=[3, 3], dropout=0.2)
cnn_trainer = Trainer(cnn_model, lr=1e-3, patience=5, device=DEVICE)
cnn_trainer.fit(tr_loader, va_loader, epochs=10, verbose=True)
y_pred_cnn = cnn_trainer.predict(te_loader)
metrics_cnn = compute_metrics(y_te_seq, y_pred_cnn, label="1D-CNN")

dl_comparison = build_comparison_table([metrics_lstm, metrics_cnn])
print("\\nDeep Learning Comparison:")
print(dl_comparison)

# Combine all metrics
all_metrics = [metrics_lr, metrics_rf, metrics_xgb, metrics_lstm, metrics_cnn]
full_comparison = build_comparison_table(all_metrics)
full_comparison.to_csv(RESULTS_DIR / "model_comparison.csv")
print("\\nFull Model Comparison saved to outputs/results/model_comparison.csv")

# Save predictions for assets
pd.DataFrame({"actual": y_te, "predicted": y_pred_xgb}).to_csv(RESULTS_DIR / "xgboost_predictions.csv", index=False)
pd.DataFrame({"actual": y_te_seq, "predicted": y_pred_lstm}).to_csv(RESULTS_DIR / "lstm_predictions.csv", index=False)

# Save history for assets
pd.DataFrame({"train_loss": lstm_trainer.train_losses, "val_loss": lstm_trainer.val_losses}).to_csv(RESULTS_DIR / "lstm_loss_history.csv", index=False)
pd.DataFrame({"train_loss": cnn_trainer.train_losses, "val_loss": cnn_trainer.val_losses}).to_csv(RESULTS_DIR / "cnn_loss_history.csv", index=False)

# 7. Generate Paper Assets
print("\\n--- Generating Paper Assets ---")
# Load dataset for EDA figures
eda_df = load_dataset(nrows=50000)

fig1_introduction()
fig2_workflow()
fig3_timeseries(eda_df)
fig4_heatmap(eda_df)
fig5_loss_curves()
fig6_actual_vs_predicted()
fig7_performance_bar()
fig8_feature_importance()
fig9_seasonal(eda_df)
fig10_scatter()
fig11_time_horizons()

generate_tables(eda_df)

print("✅ Paper assets generated successfully in docs/paper_assets/")
"""
    add_code(execution_code)

    # Write notebook
    with open("Solar_Power_Forecasting_Full.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print("Created Solar_Power_Forecasting_Full.ipynb")

if __name__ == "__main__":
    create_notebook()
