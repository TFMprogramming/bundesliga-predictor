from __future__ import annotations
"""
LightGBM classifier for 1X2 outcome prediction.
Complements XGBoost in the ensemble with different bias-variance characteristics.
Uses the same feature set as XGBoost.
"""
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

from app.models.xgboost_model import FEATURE_COLS

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
MODEL_FILE = ARTIFACTS_DIR / "lgbm_model.pkl"


class LGBMPredictor:
    def __init__(self):
        self.model: CalibratedClassifierCV | None = None
        self.feature_cols: list[str] = FEATURE_COLS

    def fit(self, feature_df: pd.DataFrame) -> dict:
        """
        Train on feature matrix. Returns validation metrics.
        feature_df must contain FEATURE_COLS + 'outcome' column.
        """
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            logger.warning("LightGBM not installed. Skipping LGBM training.")
            return {"accuracy": None, "log_loss": None, "brier_score": None, "val_samples": 0}

        df = feature_df.dropna(subset=self.feature_cols + ["outcome"]).copy()
        if len(df) < 100:
            raise ValueError(f"Not enough training samples: {len(df)}")

        X = df[self.feature_cols].values
        y = df["outcome"].values

        tscv = TimeSeriesSplit(n_splits=5)

        base_model = LGBMClassifier(
            n_estimators=800,
            num_leaves=24,             # controls tree complexity (≈ max_depth 4–5)
            learning_rate=0.025,
            subsample=0.80,
            colsample_bytree=0.65,
            min_child_samples=15,      # min samples per leaf → regularisation
            reg_alpha=0.15,
            reg_lambda=1.2,
            objective="multiclass",
            num_class=3,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

        # Sigmoid calibration: less prone to overfitting than isotonic on small sets
        self.model = CalibratedClassifierCV(base_model, cv=tscv, method="sigmoid")
        self.model.fit(X, y)

        # Evaluate on last 20% (time-ordered)
        split = int(len(X) * 0.8)
        if split < len(X):
            from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
            X_val, y_val = X[split:], y[split:]
            proba = self.model.predict_proba(X_val)
            pred = np.argmax(proba, axis=1)
            acc = accuracy_score(y_val, pred)
            ll = log_loss(y_val, proba)
            y_bin = (y_val == 0).astype(int)
            bs = brier_score_loss(y_bin, proba[:, 0])
            logger.info(f"LGBM val — accuracy={acc:.3f}, log_loss={ll:.3f}, brier={bs:.3f}")
            return {"accuracy": acc, "log_loss": ll, "brier_score": bs, "val_samples": len(X_val)}

        return {"accuracy": None, "log_loss": None, "brier_score": None, "val_samples": 0}

    def predict_proba(self, features: dict) -> dict[str, float]:
        """
        Predict 1X2 probabilities from a feature dict.
        Returns {homeWin, draw, awayWin}.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        x = np.array([[features.get(c, 0.0) for c in self.feature_cols]])
        proba = self.model.predict_proba(x)[0]
        return {
            "homeWin": float(proba[0]),
            "draw": float(proba[1]),
            "awayWin": float(proba[2]),
        }

    def save(self):
        ARTIFACTS_DIR.mkdir(exist_ok=True)
        joblib.dump(self.model, MODEL_FILE)
        logger.info(f"LGBM model saved to {MODEL_FILE}")

    @classmethod
    def load(cls) -> "LGBMPredictor":
        if not MODEL_FILE.exists():
            raise FileNotFoundError(f"No saved LGBM model at {MODEL_FILE}.")
        predictor = cls()
        predictor.model = joblib.load(MODEL_FILE)
        return predictor
