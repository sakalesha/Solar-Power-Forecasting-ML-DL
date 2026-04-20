# ☀️ Solar Power Forecasting — ML & DL

Forecast solar PV inverter power output using the **High-Resolution Floating Solar PV Dataset** (OpenEI) with a full spectrum of models — from simple linear regression to deep LSTM/CNN networks.

---

## 📂 Project Structure

```
Solar-Power-Forecasting-ML-DL/
├── docs/           → Reference papers, raw CSVs, metafile
├── data/
│   ├── raw/        → Symlinks / copies of original site CSVs
│   └── processed/  → Cleaned 15-min Parquet files (generated)
├── notebooks/      → Step-by-step analysis notebooks (.py with # %% cells)
├── src/            → Reusable Python modules
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── evaluate.py
│   ├── visualize.py
│   ├── train.py
│   └── models/
│       ├── baseline.py      → Linear Regression, RF, XGBoost
│       └── deep_learning.py → PyTorch LSTM, 1D-CNN, CNN-LSTM
├── outputs/
│   ├── models/   → Saved model weights (.pt, .joblib)
│   ├── figures/  → Plots
│   └── results/  → Metric tables (CSV)
├── requirements.txt
└── README.md
```

---

## 🔧 Setup

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Link raw data (Windows — run as admin or copy manually)
# Copy docs/FPV_Orlando_FL_data.csv → data/raw/FPV_Orlando_FL_data.csv

# 4. Open notebooks in VS Code (Jupyter extension required)
# Each .py file in notebooks/ uses # %% cell separators
```

---

## ☁️ Running on Google Colab

This project is fully compatible with Google Colab and integrates with Google Drive for handling the large raw datasets (which cannot be stored directly on GitHub).

1. Upload the raw data (`FPV_Orlando_FL_data.csv`) into a folder in your Google Drive named `Solar-Power-Forecasting-Data`.
2. Open a new Colab Notebook and run the following setup commands:

```python
# 1. Clone the repository
!git clone https://github.com/sakalesha/Solar-Power-Forecasting-ML-DL.git

# 2. Change working directory
%cd Solar-Power-Forecasting-ML-DL

# 3. Install requirements
!pip install -r requirements.txt

# 4. Mount Google Drive so config.py can automatically find the raw data
from google.colab import drive
drive.mount('/content/drive')
```

After doing this, you can run `!python src/train.py --models all` or open the python notebooks directly!

---

## 🗂️ Dataset

- **Site**: orlando, CA (floating PV system)
- **Source**: [OpenEI High-Resolution Floating Solar PV Data](https://openei.org)
- **Raw resolution**: 1-minute intervals (~1M rows)
- **Processed resolution**: **15-minute** resampled (~65K rows)
- **Target (y)**: `INVPWR` — Inverter AC Power Output (Watts)
- **Sentinel values**: `32767.0`, `32766.0`, `-99.0` → replaced with `NaN`

---

## 🤖 Models

| # | Model | Type | Library |
|---|-------|------|---------|
| 1 | Linear Regression | Baseline | scikit-learn |
| 2 | Random Forest | Ensemble | scikit-learn |
| 3 | XGBoost | Gradient Boosting | xgboost |
| 4 | LSTM | Deep Learning | PyTorch |
| 5 | 1D-CNN | Deep Learning | PyTorch |
| 6 | CNN-LSTM | Hybrid DL | PyTorch |

---

## 📊 Evaluation Metrics

- **RMSE** — Root Mean Squared Error (primary ranking)
- **MAE** — Mean Absolute Error
- **MAPE** — Mean Absolute Percentage Error
- **R²** — Coefficient of Determination
- **Pearson r** — Linear correlation

---

## 🚀 Quick Start

```bash
# Run training for all baseline models on orlando data
python src/train.py --models all --site orlando

# Train only LSTM
python src/train.py --models lstm --site orlando --epochs 50
```

---

## 📚 References

- OpenEI Floating PV Dataset (orlando, CA & Orlando, FL)
- Deep Research Report: `docs/deep-research-report.md`
- Literature: Papers 1.a, 1.b, 1.c (see `docs/`)
