from __future__ import annotations
"""
Ensemble: combines Dixon-Coles (Poisson) + XGBoost predictions.
Weights learned empirically; defaults to 60/40 DC/XGB.
"""
import logging
import math

import numpy as np

from app.models.dixon_coles import DixonColesModel
from app.models.xgboost_model import XGBoostPredictor

logger = logging.getLogger(__name__)

# Ensemble weights (can be tuned via backtest)
DC_WEIGHT = 0.60
XGB_WEIGHT = 0.40


class EnsemblePredictor:
    def __init__(self, dc_model: DixonColesModel, xgb_model: XGBoostPredictor):
        self.dc = dc_model
        self.xgb = xgb_model

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
        except Exception as e:
            logger.warning(f"XGBoost prediction failed, using DC only: {e}")
            xgb_probs = dc_probs
            xgb_weight = 0.0
            dc_weight = 1.0
        else:
            xgb_weight = XGB_WEIGHT
            dc_weight = DC_WEIGHT

        # --- Ensemble blend ---
        blended = {
            "homeWin": dc_weight * dc_probs["homeWin"] + xgb_weight * xgb_probs["homeWin"],
            "draw": dc_weight * dc_probs["draw"] + xgb_weight * xgb_probs["draw"],
            "awayWin": dc_weight * dc_probs["awayWin"] + xgb_weight * xgb_probs["awayWin"],
        }
        # Renormalise
        total = sum(blended.values())
        blended = {k: v / total for k, v in blended.items()}

        # --- Confidence (Shannon entropy, normalised) ---
        probs = list(blended.values())
        entropy = -sum(p * math.log(p + 1e-10) for p in probs)
        max_entropy = math.log(3)  # max entropy for 3 outcomes
        normalised_entropy = entropy / max_entropy  # 0 = very confident, 1 = uniform

        if normalised_entropy < 0.35:
            confidence = "high"
        elif normalised_entropy < 0.65:
            confidence = "medium"
        else:
            confidence = "low"

        # --- Model agreement (both agree on winner) ---
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
