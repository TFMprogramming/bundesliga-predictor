from __future__ import annotations
"""
XGBoost classifier for 1X2 outcome prediction.
Trained on engineered features, complements the Dixon-Coles Poisson model.
"""
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
MODEL_FILE = ARTIFACTS_DIR / "xgboost_model.pkl"

# Features used by XGBoost / LightGBM (must match builder.py output)
FEATURE_COLS = [
    # Elo ratings (strongest signal)
    "h_elo", "a_elo", "elo_diff",
    # Recent form (5 matches)
    "h_form_pts", "h_form_gf", "h_form_ga", "h_form_gd", "h_win_streak", "h_unbeaten",
    "a_form_pts", "a_form_gf", "a_form_ga", "a_form_gd", "a_win_streak", "a_unbeaten",
    # Short-term momentum (3 matches)
    "h_short_form_pts", "h_short_form_gf", "h_short_form_ga", "h_short_form_gd",
    "h_short_win_streak", "h_short_unbeaten",
    "a_short_form_pts", "a_short_form_gf", "a_short_form_ga", "a_short_form_gd",
    "a_short_win_streak", "a_short_unbeaten",
    # Long-term form (10 matches)
    "h_long_form_pts", "a_long_form_pts",
    # Venue-specific form
    "h_home_pts", "h_home_gf", "h_home_ga",
    "a_away_pts", "a_away_gf", "a_away_ga",
    # Season stats
    "h_season_pts_pg", "h_season_gf_pg", "h_season_ga_pg", "h_season_matches",
    "a_season_pts_pg", "a_season_gf_pg", "a_season_ga_pg", "a_season_matches",
    # Head-to-head
    "h2h_home_wins", "h2h_draws", "h2h_away_wins", "h2h_home_gf", "h2h_away_gf", "h2h_matches",
    # Table position
    "h_position", "a_position", "position_diff",
    # Rest / fatigue
    "h_days_rest", "a_days_rest", "rest_advantage",
    # Explicit differential features
    "form_pts_diff", "season_pts_pg_diff", "season_gd_diff",
    "home_away_pts_diff", "short_form_diff",
    # Dixon-Coles model-derived team strength (time-decay-weighted attack/defense)
    "h_dc_attack", "h_dc_defense", "a_dc_attack", "a_dc_defense",
    "dc_mu", "dc_lam", "dc_mu_lam_ratio",
    # Form trend: short vs. long momentum direction (improving / declining)
    "h_form_trend", "a_form_trend", "form_trend_diff",
    # Defensive consistency (clean sheet rate)
    "h_clean_sheet_rate", "a_clean_sheet_rate",
    # Scoring / conceding consistency (std dev)
    "h_goals_var", "a_goals_var", "h_conceded_var", "a_conceded_var",
    # Pythagorean win expectation (GF^1.83 / (GF^1.83 + GA^1.83))
    "h_pythagorean", "a_pythagorean", "pythagorean_diff",
]


class XGBoostPredictor:
    def __init__(self):
        self.model: CalibratedClassifierCV | None = None
        self.feature_cols: list[str] = FEATURE_COLS

    def fit(self, feature_df: pd.DataFrame) -> dict:
        """
        Train on feature matrix. Returns validation metrics.
        feature_df must contain FEATURE_COLS + 'outcome' column.
        """
        df = feature_df.dropna(subset=self.feature_cols + ["outcome"]).copy()
        if len(df) < 100:
            raise ValueError(f"Not enough training samples: {len(df)}")

        X = df[self.feature_cols].values
        y = df["outcome"].values

        # Time-series cross-validation (no future data leak)
        tscv = TimeSeriesSplit(n_splits=5)

        base_model = XGBClassifier(
            n_estimators=700,
            max_depth=4,
            learning_rate=0.025,       # slower learning → better generalisation
            subsample=0.75,
            colsample_bytree=0.65,
            colsample_bylevel=0.65,    # additional column subsampling per level
            min_child_weight=10,       # stricter leaf requirements → less overfitting
            gamma=0.3,
            reg_alpha=0.3,
            reg_lambda=2.0,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

        # Calibrate probabilities for better confidence estimates
        self.model = CalibratedClassifierCV(base_model, cv=tscv, method="isotonic")
        self.model.fit(X, y)

        # Evaluate on last 20% (time-ordered)
        split = int(len(X) * 0.8)
        if split < len(X):
            X_val, y_val = X[split:], y[split:]
            from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

            proba = self.model.predict_proba(X_val)
            pred = np.argmax(proba, axis=1)
            acc = accuracy_score(y_val, pred)
            ll = log_loss(y_val, proba)

            # Brier score for home win (class 0)
            y_bin = (y_val == 0).astype(int)
            bs = brier_score_loss(y_bin, proba[:, 0])

            logger.info(f"XGBoost val — accuracy={acc:.3f}, log_loss={ll:.3f}, brier={bs:.3f}")
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
        proba = self.model.predict_proba(x)[0]  # [home_win, draw, away_win]
        return {
            "homeWin": float(proba[0]),
            "draw": float(proba[1]),
            "awayWin": float(proba[2]),
        }

    def save(self):
        ARTIFACTS_DIR.mkdir(exist_ok=True)
        joblib.dump(self.model, MODEL_FILE)
        logger.info(f"XGBoost model saved to {MODEL_FILE}")

    @classmethod
    def load(cls) -> "XGBoostPredictor":
        if not MODEL_FILE.exists():
            raise FileNotFoundError(f"No saved XGBoost model at {MODEL_FILE}.")
        predictor = cls()
        predictor.model = joblib.load(MODEL_FILE)
        return predictor
