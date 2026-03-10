from __future__ import annotations
"""
Ensemble: combines Dixon-Coles (Poisson) + XGBoost predictions.
Weights are optimised via cross-validation in optimize_weights().
"""
import json
import logging
import math
from pathlib import Path

import numpy as np

from app.models.dixon_coles import DixonColesModel
from app.models.xgboost_model import XGBoostPredictor

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
WEIGHTS_FILE = ARTIFACTS_DIR / "ensemble_weights.json"

# Defaults – overridden if weights file exists or optimize_weights() is called
DC_WEIGHT = 0.40
XGB_WEIGHT = 0.60


def _load_weights() -> tuple[float, float]:
    """Load saved ensemble weights, falling back to defaults."""
    if WEIGHTS_FILE.exists():
        data = json.loads(WEIGHTS_FILE.read_text())
        return float(data["dc_weight"]), float(data["xgb_weight"])
    return DC_WEIGHT, XGB_WEIGHT


def optimize_weights(
    dc_probs_list: list[dict],
    xgb_probs_list: list[dict],
    outcomes: list[int],
) -> tuple[float, float]:
    """
    Find the DC/XGB blend weight that minimises log-loss on held-out predictions.

    dc_probs_list / xgb_probs_list: list of {homeWin, draw, awayWin} dicts
    outcomes: list of ints (0=home win, 1=draw, 2=away win)

    Returns (dc_weight, xgb_weight) and saves to disk.
    """
    from scipy.optimize import minimize_scalar

    keys = ["homeWin", "draw", "awayWin"]

    dc_arr = np.array([[p[k] for k in keys] for p in dc_probs_list])
    xgb_arr = np.array([[p[k] for k in keys] for p in xgb_probs_list])
    y = np.array(outcomes)

    def log_loss(alpha: float) -> float:
        blended = alpha * dc_arr + (1 - alpha) * xgb_arr
        # Normalise rows
        blended = blended / blended.sum(axis=1, keepdims=True)
        # Clip to avoid log(0)
        blended = np.clip(blended, 1e-10, 1.0)
        ll = -np.mean(np.log(blended[np.arange(len(y)), y]))
        return ll

    result = minimize_scalar(log_loss, bounds=(0.0, 1.0), method="bounded")
    best_alpha = float(result.x)
    dc_w = round(best_alpha, 3)
    xgb_w = round(1.0 - dc_w, 3)

    logger.info(f"Optimised ensemble weights: DC={dc_w:.3f}, XGB={xgb_w:.3f} (log-loss={result.fun:.4f})")

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    WEIGHTS_FILE.write_text(json.dumps({"dc_weight": dc_w, "xgb_weight": xgb_w}, indent=2))

    return dc_w, xgb_w


class EnsemblePredictor:
    def __init__(self, dc_model: DixonColesModel, xgb_model: XGBoostPredictor):
        self.dc = dc_model
        self.xgb = xgb_model
        self.dc_weight, self.xgb_weight = _load_weights()
        logger.info(f"Ensemble weights: DC={self.dc_weight:.2f}, XGB={self.xgb_weight:.2f}")

    def predict(
        self,
        home_team: str,
        away_team: str,
        features: dict,
        max_goals: int = 8,
    ) -> dict:
        """
        Full prediction for one match.

        Returns:
            probabilities: {homeWin, draw, awayWin}
            predictedScore: {home, away}
            expectedGoals: {home, away}
            scoreMatrix: 2D list (max_goals+1 x max_goals+1) of joint probabilities
            confidenceLevel: "high" | "medium" | "low"
            modelAgreement: bool
            dcProbabilities: raw Dixon-Coles probs
            xgbProbabilities: raw XGBoost probs
        """
        # --- Dixon-Coles ---
        dc_probs = self.dc.predict_1x2(home_team, away_team)
        mu, lam = self.dc.expected_goals(home_team, away_team)
        score_matrix = self.dc.score_matrix(home_team, away_team, max_goals).tolist()
        dc_score = self.dc.predict_score(home_team, away_team)

        # --- XGBoost ---
        try:
            xgb_probs = self.xgb.predict_proba(features)
            xgb_weight = self.xgb_weight
            dc_weight = self.dc_weight
        except Exception as e:
            logger.warning(f"XGBoost prediction failed, using DC only: {e}")
            xgb_probs = dc_probs
            xgb_weight = 0.0
            dc_weight = 1.0

        # --- Ensemble blend ---
        blended = {
            "homeWin": dc_weight * dc_probs["homeWin"] + xgb_weight * xgb_probs["homeWin"],
            "draw":    dc_weight * dc_probs["draw"]    + xgb_weight * xgb_probs["draw"],
            "awayWin": dc_weight * dc_probs["awayWin"] + xgb_weight * xgb_probs["awayWin"],
        }
        # Renormalise
        total = sum(blended.values())
        blended = {k: v / total for k, v in blended.items()}

        # --- Confidence (Shannon entropy, normalised) ---
        probs = list(blended.values())
        entropy = -sum(p * math.log(p + 1e-10) for p in probs)
        max_entropy = math.log(3)
        normalised_entropy = entropy / max_entropy  # 0 = confident, 1 = uniform

        if normalised_entropy < 0.35:
            confidence = "high"
        elif normalised_entropy < 0.65:
            confidence = "medium"
        else:
            confidence = "low"

        # --- Model agreement (both predict same winner) ---
        dc_winner = max(dc_probs, key=dc_probs.get)
        xgb_winner = max(xgb_probs, key=xgb_probs.get)
        agreement = dc_winner == xgb_winner

        return {
            "probabilities": {k: round(v, 4) for k, v in blended.items()},
            "predictedScore": {"home": int(dc_score[0]), "away": int(dc_score[1])},
            "expectedGoals": {"home": round(mu, 2), "away": round(lam, 2)},
            "scoreMatrix": [[round(p, 5) for p in row] for row in score_matrix],
            "confidenceLevel": confidence,
            "modelAgreement": agreement,
            "dcProbabilities": {k: round(v, 4) for k, v in dc_probs.items()},
            "xgbProbabilities": {k: round(v, 4) for k, v in xgb_probs.items()},
        }


def load_ensemble() -> EnsemblePredictor:
    dc = DixonColesModel.load()
    xgb = XGBoostPredictor.load()
    return EnsemblePredictor(dc, xgb)
