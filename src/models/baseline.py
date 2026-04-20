"""
baseline.py — Scikit-learn baseline models for solar power forecasting.

Models implemented:
  • LinearRegressionModel  — simple OLS linear regression
  • RandomForestModel      — ensemble of decision trees
  • XGBoostModel           — gradient-boosted trees

All models expose a unified interface:
  .fit(X_train, y_train)
  .predict(X)
  .save(path) / .load(path)
  .feature_importance()   (RF and XGBoost only)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble    import RandomForestRegressor
from xgboost             import XGBRegressor

from src.config import RF_PARAMS, XGB_PARAMS, MODELS_DIR


# ─────────────────────────────────────────────────────────────────────────────
# BASE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class BaseModel:
    """Common interface for all baseline models."""

    name: str = "BaseModel"

    def fit(self, X_train, y_train):
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: Path | None = None) -> Path:
        """Persist model to disk using joblib."""
        if path is None:
            path = MODELS_DIR / f"{self.name.lower().replace(' ', '_')}.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        print(f"💾 Saved {self.name} → {path}")
        return path

    @classmethod
    def load(cls, path: Path):
        """Load a saved model from disk."""
        instance = cls.__new__(cls)
        instance.model = joblib.load(path)
        print(f"📂 Loaded {cls.__name__} from {path}")
        return instance

    def feature_importance(self) -> np.ndarray | None:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# LINEAR REGRESSION
# ─────────────────────────────────────────────────────────────────────────────

class LinearRegressionModel(BaseModel):
    """
    Ordinary Least Squares (OLS) Linear Regression.
    Uses Ridge regularisation (default α=0.1) to avoid instability
    when features are highly correlated.
    """

    name = "Linear Regression"

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)

    def fit(self, X_train, y_train):
        print(f"🔵 Training {self.name}...")
        self.model.fit(X_train, y_train)
        print(f"   Done. Coefficients: {self.model.coef_.shape}")
        return self

    def predict(self, X) -> np.ndarray:
        preds = self.model.predict(X)
        return np.maximum(preds, 0)   # power cannot be negative

    @property
    def coefficients(self) -> pd.Series:
        """Named coefficients for interpretability."""
        if hasattr(X_train := self.model, "feature_names_in_"):
            return pd.Series(self.model.coef_,
                             index=self.model.feature_names_in_)
        return pd.Series(self.model.coef_)


# ─────────────────────────────────────────────────────────────────────────────
# RANDOM FOREST
# ─────────────────────────────────────────────────────────────────────────────

class RandomForestModel(BaseModel):
    """
    Random Forest Regressor — ensemble of decision trees.
    Provides feature_importance() via the built-in Gini importance.
    """

    name = "Random Forest"

    def __init__(self, **kwargs):
        params = {**RF_PARAMS, **kwargs}
        self.model = RandomForestRegressor(**params)

    def fit(self, X_train, y_train):
        print(f"🌲 Training {self.name} "
              f"({self.model.n_estimators} trees, max_depth={self.model.max_depth})...")
        self.model.fit(X_train, y_train)
        print(f"   Done.")
        return self

    def predict(self, X) -> np.ndarray:
        return np.maximum(self.model.predict(X), 0)

    def feature_importance(self) -> np.ndarray:
        return self.model.feature_importances_


# ─────────────────────────────────────────────────────────────────────────────
# XGBOOST
# ─────────────────────────────────────────────────────────────────────────────

class XGBoostModel(BaseModel):
    """
    Gradient-Boosted Decision Trees via XGBoost.
    Supports early stopping on a validation set.
    """

    name = "XGBoost"

    def __init__(self, **kwargs):
        params = {**XGB_PARAMS, **kwargs}
        self.model = XGBRegressor(**params)

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        early_stopping_rounds: int = 20,
        verbose: bool = True,
    ):
        print(f"🚀 Training {self.name} "
              f"({self.model.n_estimators} estimators)...")

        eval_set   = [(X_val, y_val)] if X_val is not None else None
        verbosity  = 50 if verbose else 0

        self.model.fit(
            X_train, y_train,
            eval_set = eval_set,
            early_stopping_rounds = early_stopping_rounds if eval_set else None,
            verbose  = verbosity,
        )
        best = getattr(self.model, "best_iteration", self.model.n_estimators)
        print(f"   Done. Best iteration: {best}")
        return self

    def predict(self, X) -> np.ndarray:
        return np.maximum(self.model.predict(X), 0)

    def feature_importance(self) -> np.ndarray:
        return self.model.feature_importances_

    def save(self, path: Path | None = None) -> Path:
        if path is None:
            path = MODELS_DIR / "xgboost.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        print(f"💾 Saved {self.name} → {path}")
        return path


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def get_baseline_model(name: str) -> BaseModel:
    """
    Instantiate a baseline model by name.

    Parameters
    ----------
    name : "lr" | "rf" | "xgb"

    Returns
    -------
    Instantiated (unfitted) model.
    """
    registry = {
        "lr":  LinearRegressionModel,
        "rf":  RandomForestModel,
        "xgb": XGBoostModel,
    }
    key = name.lower().strip()
    if key not in registry:
        raise ValueError(f"Unknown baseline model '{name}'. "
                         f"Choose from: {list(registry.keys())}")
    return registry[key]()
