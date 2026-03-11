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


def _poisson_pmf(k: int, mu: float) -> float:
    """Poisson PMF, stable for k ≤ 10 and mu ≤ 10."""
    if mu < 1e-10:
        return 1.0 if k == 0 else 0.0
    try:
        return math.exp(-mu) * (mu ** k) / math.factorial(k)
    except (OverflowError, ValueError):
        return 0.0


def _calibrate_goals(mu: float, lam: float, target_hw: float, max_goals: int = 8) -> tuple[float, float]:
    """
    Adjust (mu, lam) so that P(home wins) matches target_hw,
    while preserving total expected goals (mu + lam).
    Uses binary search on the ratio r = mu/lam.
    """
    total = mu + lam
    if total < 0.1 or not (0.05 < target_hw < 0.95):
        return mu, lam

    def hw_at_ratio(r: float) -> float:
        m = total * r / (1.0 + r)
        la = total / (1.0 + r)
        hw = 0.0
        for h in range(1, max_goals + 1):
            ph = _poisson_pmf(h, m)
            if ph < 1e-10:
                continue
            for a in range(h):
                hw += ph * _poisson_pmf(a, la)
        return hw

    lo, hi = 0.05, 20.0
    for _ in range(35):
        mid = (lo + hi) / 2.0
        if hw_at_ratio(mid) < target_hw:
            lo = mid
        else:
            hi = mid
    r_opt = (lo + hi) / 2.0
    return total * r_opt / (1.0 + r_opt), total / (1.0 + r_opt)

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

        # --- Calibrated score prediction (aligned with ensemble probability) ---
        try:
            cal_mu, cal_lam = _calibrate_goals(mu, lam, blended["homeWin"])
            size = max_goals + 1
            cal_flat = [
                _poisson_pmf(h, cal_mu) * _poisson_pmf(a, cal_lam)
                for h in range(size) for a in range(size)
            ]
            total_p = sum(cal_flat) or 1.0
            cal_flat = [p / total_p for p in cal_flat]

            top_indices = sorted(range(len(cal_flat)), key=lambda i: cal_flat[i], reverse=True)[:5]
            top_scorelines = [
                {
                    "home": i // size,
                    "away": i % size,
                    "probability": round(cal_flat[i], 4),
                }
                for i in top_indices
            ]
            best = top_indices[0]
            predicted_score = {"home": best // size, "away": best % size}
        except Exception:
            top_scorelines = []
            predicted_score = {"home": int(dc_score[0]), "away": int(dc_score[1])}

        return {
            "probabilities": {k: round(v, 4) for k, v in blended.items()},
            "predictedScore": predicted_score,
            "expectedGoals": {"home": round(mu, 2), "away": round(lam, 2)},
            "scoreMatrix": [[round(p, 5) for p in row] for row in score_matrix],
            "confidenceLevel": confidence,
            "modelAgreement": agreement,
            "dcProbabilities": {k: round(v, 4) for k, v in dc_probs.items()},
            "xgbProbabilities": {k: round(v, 4) for k, v in xgb_probs.items()},
            "topScorelines": top_scorelines,
        }


def load_ensemble() -> EnsemblePredictor:
    dc = DixonColesModel.load()
    xgb = XGBoostPredictor.load()
    return EnsemblePredictor(dc, xgb)
