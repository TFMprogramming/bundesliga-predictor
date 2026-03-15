from __future__ import annotations
"""
Ensemble: combines Dixon-Coles (Poisson) + XGBoost + LightGBM predictions.
Weights are optimised via cross-validation using log-loss + Brier score.
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
from app.models.lgbm_model import LGBMPredictor

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
WEIGHTS_FILE = ARTIFACTS_DIR / "ensemble_weights.json"

# Defaults — overridden if weights file exists or optimize_weights() is called
DC_WEIGHT_DEFAULT = 0.30
XGB_WEIGHT_DEFAULT = 0.45
LGBM_WEIGHT_DEFAULT = 0.25


def _load_weights() -> tuple[float, float, float]:
    """Load saved ensemble weights, falling back to defaults."""
    if WEIGHTS_FILE.exists():
        data = json.loads(WEIGHTS_FILE.read_text())
        dc_w = float(data.get("dc_weight", DC_WEIGHT_DEFAULT))
        xgb_w = float(data.get("xgb_weight", XGB_WEIGHT_DEFAULT))
        lgbm_w = float(data.get("lgbm_weight", LGBM_WEIGHT_DEFAULT))
        # Renormalise in case of old 2-model weight file (lgbm_weight missing)
        total = dc_w + xgb_w + lgbm_w
        return dc_w / total, xgb_w / total, lgbm_w / total
    return DC_WEIGHT_DEFAULT, XGB_WEIGHT_DEFAULT, LGBM_WEIGHT_DEFAULT


def optimize_weights(
    dc_probs_list: list[dict],
    xgb_probs_list: list[dict],
    lgbm_probs_list: list[dict] | None,
    outcomes: list[int],
) -> tuple[float, float, float]:
    """
    Find the DC/XGB/LGBM blend weights that minimise a combined
    log-loss + Brier score objective on held-out predictions.

    Returns (dc_weight, xgb_weight, lgbm_weight) and saves to disk.
    """
    from scipy.optimize import minimize

    keys = ["homeWin", "draw", "awayWin"]
    dc_arr = np.array([[p[k] for k in keys] for p in dc_probs_list])
    xgb_arr = np.array([[p[k] for k in keys] for p in xgb_probs_list])
    y = np.array(outcomes)

    has_lgbm = lgbm_probs_list is not None and len(lgbm_probs_list) == len(outcomes)
    lgbm_arr = np.array([[p[k] for k in keys] for p in lgbm_probs_list]) if has_lgbm else None

    def _blend(w_dc: float, w_xgb: float, w_lgbm: float) -> np.ndarray:
        b = w_dc * dc_arr + w_xgb * xgb_arr
        if has_lgbm and lgbm_arr is not None:
            b = b + w_lgbm * lgbm_arr
        b = b / b.sum(axis=1, keepdims=True)
        return np.clip(b, 1e-10, 1.0)

    def objective(log_w: np.ndarray) -> float:
        # Softmax parameterisation: weights always positive & sum to 1
        w = np.exp(log_w - log_w.max())
        w = w / w.sum()
        blended = _blend(w[0], w[1], w[2])
        # Log-loss
        ll = -np.mean(np.log(blended[np.arange(len(y)), y]))
        # Brier score (home win as reference class, generalised)
        brier = float(np.mean(np.sum((blended - np.eye(3)[y]) ** 2, axis=1)))
        return 0.7 * ll + 0.3 * brier  # combined objective

    n_models = 3 if has_lgbm else 2
    x0 = np.zeros(n_models)  # softmax([0,0,0]) = [1/3, 1/3, 1/3]
    if n_models == 2:
        # Pad to 3 dims but fix lgbm to near-zero
        x0 = np.array([0.0, 0.0, -10.0])

    result = minimize(objective, x0, method="Nelder-Mead",
                      options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-6})

    w_opt = np.exp(result.x - result.x.max())
    w_opt = w_opt / w_opt.sum()
    dc_w = round(float(w_opt[0]), 3)
    xgb_w = round(float(w_opt[1]), 3)
    lgbm_w = round(float(w_opt[2]), 3)

    logger.info(
        f"Optimised ensemble weights: DC={dc_w:.3f}, XGB={xgb_w:.3f}, "
        f"LGBM={lgbm_w:.3f} (objective={result.fun:.4f})"
    )

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    WEIGHTS_FILE.write_text(json.dumps(
        {"dc_weight": dc_w, "xgb_weight": xgb_w, "lgbm_weight": lgbm_w}, indent=2
    ))

    return dc_w, xgb_w, lgbm_w


class EnsemblePredictor:
    def __init__(
        self,
        dc_model: DixonColesModel,
        xgb_model: XGBoostPredictor,
        lgbm_model: LGBMPredictor | None = None,
    ):
        self.dc = dc_model
        self.xgb = xgb_model
        self.lgbm = lgbm_model
        self.dc_weight, self.xgb_weight, self.lgbm_weight = _load_weights()
        if lgbm_model is None:
            # Redistribute LGBM weight to XGB if LGBM not available
            total = self.dc_weight + self.xgb_weight
            if total > 0:
                self.dc_weight /= total
                self.xgb_weight /= total
            self.lgbm_weight = 0.0
        logger.info(
            f"Ensemble weights: DC={self.dc_weight:.2f}, "
            f"XGB={self.xgb_weight:.2f}, LGBM={self.lgbm_weight:.2f}"
        )

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
            lgbmProbabilities: raw LightGBM probs (or same as XGB if unavailable)
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
            logger.warning(f"XGBoost prediction failed: {e}")
            xgb_probs = None

        # --- LightGBM ---
        lgbm_probs = None
        if self.lgbm is not None and self.lgbm_weight > 0:
            try:
                lgbm_probs = self.lgbm.predict_proba(features)
            except Exception as e:
                logger.warning(f"LightGBM prediction failed: {e}")

        # --- Ensemble blend ---
        # Determine effective weights based on model availability
        dc_w = self.dc_weight
        xgb_w = self.xgb_weight if xgb_probs is not None else 0.0
        lgbm_w = self.lgbm_weight if lgbm_probs is not None else 0.0

        if xgb_probs is None and lgbm_probs is None:
            xgb_probs = dc_probs
            lgbm_probs = dc_probs
            dc_w, xgb_w, lgbm_w = 1.0, 0.0, 0.0
        elif xgb_probs is None:
            xgb_probs = lgbm_probs
            dc_w += xgb_w
            xgb_w = 0.0
        elif lgbm_probs is None:
            lgbm_probs = xgb_probs
            dc_w += lgbm_w
            lgbm_w = 0.0

        total_w = dc_w + xgb_w + lgbm_w
        if total_w < 1e-10:
            dc_w, total_w = 1.0, 1.0

        keys = ["homeWin", "draw", "awayWin"]
        blended = {
            k: (dc_w * dc_probs[k] + xgb_w * xgb_probs[k] + lgbm_w * lgbm_probs[k]) / total_w
            for k in keys
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

        # --- Model agreement (DC, XGB and LGBM all agree on winner) ---
        winners = [max(dc_probs, key=dc_probs.get), max(xgb_probs, key=xgb_probs.get)]
        if lgbm_probs is not None:
            winners.append(max(lgbm_probs, key=lgbm_probs.get))
        agreement = len(set(winners)) == 1

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
            "lgbmProbabilities": {k: round(v, 4) for k, v in (lgbm_probs or xgb_probs).items()},
            "topScorelines": top_scorelines,
        }


def load_ensemble() -> EnsemblePredictor:
    dc = DixonColesModel.load()
    xgb = XGBoostPredictor.load()
    try:
        lgbm = LGBMPredictor.load()
    except (FileNotFoundError, Exception) as e:
        logger.info(f"LightGBM model not available, using DC+XGB only: {e}")
        lgbm = None
    return EnsemblePredictor(dc, xgb, lgbm)
