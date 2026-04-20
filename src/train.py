"""
train.py — CLI training entry-point for all solar forecasting models.

Usage
-----
# Train everything on Oakville data (after preprocessing + features built)
python src/train.py --models all --site oakville

# Train only LSTM for 30 epochs
python src/train.py --models lstm --site oakville --epochs 30

# Train baseline models only
python src/train.py --models lr rf xgb --site oakville
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ─── adjust sys.path so imports work from project root ────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    OAKVILLE_PARQUET, RESULTS_DIR, MODELS_DIR,
    ACTIVE_SITE, SEED, LOOKBACK, BATCH_SIZE, EPOCHS as CFG_EPOCHS,
    FEATURE_COLS, TARGET,
)
from src.data_loader      import load_and_standardise
from src.preprocessing    import full_pipeline
from src.features         import engineer_features, build_sequences
from src.evaluate         import compute_metrics, build_comparison_table, print_metrics
from src.models.baseline  import LinearRegressionModel, RandomForestModel, XGBoostModel
from src.models.deep_learning import (
    LSTMForecaster, CNNForecaster, CNNLSTMForecaster,
    Trainer, make_dataloader, get_dl_model,
)


# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & PREP
# ─────────────────────────────────────────────────────────────────────────────

def load_data(site: str = ACTIVE_SITE) -> tuple:
    """
    Load→clean→feature-engineer the site data.
    Returns (data_dict, feature_cols) ready for modelling.
    """
    print("\n" + "="*60)
    print(f"  📥  Loading {site} data")
    print("="*60)

    # 1. Load raw CSV
    df_raw = load_and_standardise(site=site)

    # 2. Engineer features before pipeline (add time, lags, rolling)
    irr_col = "irr_horiz"
    df_feat = engineer_features(
        df_raw,
        target      = TARGET,
        irr_col     = irr_col,
        lags        = [1, 4, 8],
        roll_windows= [4, 8],
        add_clearness=True,
    )

    # 3. Determine feature columns available in this dataset
    candidate_features = FEATURE_COLS + [
        "clearness_index", "month",
        "irr_lag1", "irr_lag4", "irr_lag8",
        "target_lag1", "target_lag4",
        "irr_roll4_mean", "irr_roll4_std",
        "irr_roll8_mean", "irr_roll8_std",
        "irr_horiz", "amb_temp", "rh", "wind_speed", "pressure",
        "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    ]
    # Deduplicate and keep only columns that exist
    seen, feature_cols = set(), []
    for c in candidate_features:
        if c in df_feat.columns and c != TARGET and c not in seen:
            feature_cols.append(c)
            seen.add(c)

    # 4. Run full preprocessing pipeline (split + scale)
    data = full_pipeline(
        df_feat, feature_cols, target=TARGET,
        resample=False,        # already done as 15-min raw cols (1-min → resampled in loader)
        filter_night=True,
    )
    data["feature_cols"] = feature_cols
    return data, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_baseline(model_name: str, data: dict) -> dict:
    """Fit and evaluate a baseline model."""
    X_tr, y_tr = data["train"]
    X_va, y_va = data["val"]
    X_te, y_te = data["test"]

    model_map = {"lr": LinearRegressionModel, "rf": RandomForestModel, "xgb": XGBoostModel}
    ModelClass = model_map[model_name]
    model = ModelClass()

    if model_name == "xgb":
        model.fit(X_tr, y_tr, X_val=X_va, y_val=y_va)
    else:
        model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    metrics = compute_metrics(y_te, y_pred, label=model.name)
    print_metrics(metrics)

    model.save()
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# DL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_dl(model_name: str, data: dict, epochs: int = CFG_EPOCHS) -> dict:
    """Fit and evaluate a deep learning model."""
    X_tr, y_tr = data["train"]
    X_va, y_va = data["val"]
    X_te, y_te = data["test"]

    # Build sequences
    X_tr_seq, y_tr_seq = build_sequences(X_tr.values, y_tr.values, LOOKBACK)
    X_va_seq, y_va_seq = build_sequences(X_va.values, y_va.values, LOOKBACK)
    X_te_seq, y_te_seq = build_sequences(X_te.values, y_te.values, LOOKBACK)

    n_features = X_tr.shape[1]
    print(f"\n🧠 DL Input shape: ({LOOKBACK}, {n_features})")

    # Create DataLoaders
    tr_loader = make_dataloader(X_tr_seq, y_tr_seq, shuffle=True)
    va_loader = make_dataloader(X_va_seq, y_va_seq, shuffle=False)
    te_loader = make_dataloader(X_te_seq, y_te_seq, shuffle=False)

    # Instantiate model
    print(f"\n{'='*60}")
    print(f"  Training {model_name.upper()}")
    print("="*60)
    model = get_dl_model(model_name, n_features=n_features, lookback=LOOKBACK)
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    # Train
    trainer = Trainer(model, patience=10)
    trainer.fit(tr_loader, va_loader, epochs=epochs)
    trainer.save_model()

    # Save loss history
    history = pd.DataFrame({
        "train_loss": trainer.train_losses,
        "val_loss":   trainer.val_losses,
    })
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    history.to_csv(RESULTS_DIR / f"{model_name}_loss_history.csv", index=False)

    # Evaluate
    y_pred   = trainer.predict(te_loader)
    y_true   = y_te_seq
    metrics  = compute_metrics(y_true, y_pred, label=model_name.upper())
    print_metrics(metrics)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Solar Power Forecasting — Training CLI"
    )
    parser.add_argument(
        "--models", nargs="+",
        default=["all"],
        choices=["all", "lr", "rf", "xgb", "lstm", "cnn", "cnn_lstm"],
        help="Which models to train (default: all)"
    )
    parser.add_argument(
        "--site", type=str, default=ACTIVE_SITE,
        choices=["oakville", "orlando"],
        help=f"Which site dataset to use (default: {ACTIVE_SITE} from config.py)"
    )
    parser.add_argument(
        "--epochs", type=int, default=CFG_EPOCHS,
        help="Epochs for deep learning models"
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Expand 'all'
    if "all" in args.models:
        args.models = ["lr", "rf", "xgb", "lstm", "cnn", "cnn_lstm"]

    baseline_models = [m for m in args.models if m in ("lr", "rf", "xgb")]
    dl_models       = [m for m in args.models if m in ("lstm", "cnn", "cnn_lstm")]

    # Load data (common to all models)
    data, feature_cols = load_data(site=args.site)
    print(f"\n📐 Features used ({len(feature_cols)}): {feature_cols[:8]}...")

    all_metrics = []

    # ── Baselines ────────────────────────────────────────────────────────────
    for model_name in baseline_models:
        metrics = train_baseline(model_name, data)
        all_metrics.append(metrics)

    # ── Deep Learning ─────────────────────────────────────────────────────────
    for model_name in dl_models:
        metrics = train_dl(model_name, data, epochs=args.epochs)
        all_metrics.append(metrics)

    # ── Summary Table ─────────────────────────────────────────────────────────
    if all_metrics:
        print("\n" + "="*60)
        print("  📊  FINAL MODEL COMPARISON")
        print("="*60)
        table = build_comparison_table(all_metrics)
        print(table.to_string())

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = RESULTS_DIR / "model_comparison.csv"
        table.to_csv(csv_path)
        print(f"\n📁 Results saved → {csv_path}")


if __name__ == "__main__":
    main()
