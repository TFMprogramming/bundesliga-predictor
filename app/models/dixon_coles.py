from __future__ import annotations
"""
Dixon-Coles Poisson model for Bundesliga score prediction.

Dixon & Coles (1997): "Modelling Association Football Scores and Inefficiencies
in the Football Betting Market", Applied Statistics.

Key parameters per team:
  alpha_i  — attack strength
  beta_j   — defense strength
  gamma    — home advantage multiplier
  rho      — low-score correction (handles 0-0, 1-0, 0-1, 1-1 overdispersion)
  xi       — time decay rate (recent matches weighted higher)
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
PARAMS_FILE = ARTIFACTS_DIR / "dixon_coles_params.json"


# ---------------------------------------------------------------------------
# Low-score correction (Dixon-Coles tau function)
# ---------------------------------------------------------------------------

def tau(x: int, y: int, mu: float, lam: float, rho: float) -> float:
    """Correction factor for low-scoring outcomes."""
    if x == 0 and y == 0:
        return 1 - mu * lam * rho
    elif x == 0 and y == 1:
        return 1 + mu * rho
    elif x == 1 and y == 0:
        return 1 + lam * rho
    elif x == 1 and y == 1:
        return 1 - rho
    return 1.0


# ---------------------------------------------------------------------------
# Time weighting
# ---------------------------------------------------------------------------

def time_weight(match_dt: datetime, reference_dt: datetime, xi: float) -> float:
    """Exponential decay: w = exp(-xi * days_elapsed)."""
    days = max((reference_dt - match_dt).days, 0)
    return np.exp(-xi * days)


# ---------------------------------------------------------------------------
# Dixon-Coles model
# ---------------------------------------------------------------------------

class DixonColesModel:
    def __init__(self, xi: float = 0.006):  # half-life ~115 days (was 230)
        self.xi = xi
        self.teams: list[str] = []
        self.alpha: dict[str, float] = {}   # attack
        self.beta: dict[str, float] = {}    # defense
        self.gamma: float = 1.0             # home advantage
        self.rho: float = 0.0              # low-score correction
        self.fitted: bool = False
        self.fitted_at: datetime | None = None

    # -----------------------------------------------------------------------
    # Fitting
    # -----------------------------------------------------------------------

    def fit(self, matches: list[dict], reference_dt: datetime | None = None):
        """
        Fit model parameters via Maximum Likelihood Estimation.

        matches: list of dicts with keys:
            home_team, away_team, home_goals, away_goals, match_datetime
        """
        if reference_dt is None:
            reference_dt = datetime.utcnow()

        # Filter only finished matches
        matches = [m for m in matches if m["home_goals"] is not None and m["away_goals"] is not None]
        if len(matches) < 50:
            raise ValueError(f"Not enough matches to fit model: {len(matches)}")

        self.teams = sorted(set(
            t for m in matches for t in [m["home_team"], m["away_team"]]
        ))
        n_teams = len(self.teams)
        team_idx = {t: i for i, t in enumerate(self.teams)}

        # Compute time weights
        weights = np.array([
            time_weight(m["match_datetime"], reference_dt, self.xi)
            for m in matches
        ])

        home_goals = np.array([m["home_goals"] for m in matches])
        away_goals = np.array([m["away_goals"] for m in matches])
        home_idx = np.array([team_idx[m["home_team"]] for m in matches])
        away_idx = np.array([team_idx[m["away_team"]] for m in matches])

        # Initial parameters: [alpha_0..n, beta_0..n, log_gamma, rho]
        # Constraint: alpha[0] = 1 (reference team, so alpha[0] is fixed via log scale)
        # We optimise log(alpha), log(beta), log(gamma), rho
        x0 = np.zeros(2 * n_teams + 2)
        x0[n_teams] = 0.0    # log(beta) = 0 → beta = 1
        x0[-2] = np.log(1.3)  # log(gamma) ≈ home advantage
        x0[-1] = -0.1         # rho

        def neg_log_likelihood(params: np.ndarray) -> float:
            log_alpha = params[:n_teams]
            log_beta = params[n_teams:2 * n_teams]
            log_gamma = params[-2]
            rho = params[-1]

            # Clamp rho to valid range
            rho = np.clip(rho, -0.99, 0.99)

            alpha = np.exp(log_alpha)
            beta = np.exp(log_beta)
            gamma = np.exp(log_gamma)

            mu = alpha[home_idx] * beta[away_idx] * gamma   # home expected goals
            lam = alpha[away_idx] * beta[home_idx]           # away expected goals

            # Poisson log-likelihood
            ll = (
                home_goals * np.log(mu + 1e-10) - mu
                + away_goals * np.log(lam + 1e-10) - lam
            )

            # Dixon-Coles low-score correction (only for 0-0, 1-0, 0-1, 1-1)
            correction = np.zeros(len(matches))
            mask_00 = (home_goals == 0) & (away_goals == 0)
            mask_10 = (home_goals == 1) & (away_goals == 0)
            mask_01 = (home_goals == 0) & (away_goals == 1)
            mask_11 = (home_goals == 1) & (away_goals == 1)

            correction[mask_00] = np.log(np.maximum(1 - mu[mask_00] * lam[mask_00] * rho, 1e-10))
            correction[mask_10] = np.log(np.maximum(1 + lam[mask_10] * rho, 1e-10))
            correction[mask_01] = np.log(np.maximum(1 + mu[mask_01] * rho, 1e-10))
            correction[mask_11] = np.log(np.maximum(1 - rho, 1e-10))

            total = np.sum(weights * (ll + correction))
            return -total  # minimise negative log-likelihood

        # Constraint: sum of log_alpha = 0 (identifiability)
        constraints = [
            {"type": "eq", "fun": lambda p: np.sum(p[:n_teams])}
        ]

        bounds = (
            [(-3, 3)] * n_teams          # log_alpha
            + [(-3, 3)] * n_teams        # log_beta
            + [(np.log(0.5), np.log(3))] # log_gamma
            + [(-0.99, 0.99)]            # rho
        )

        logger.info(f"Fitting Dixon-Coles on {len(matches)} matches, {n_teams} teams...")
        result = minimize(
            neg_log_likelihood,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 2000, "ftol": 1e-10},
        )

        if not result.success:
            logger.warning(f"Optimisation warning: {result.message}")

        params = result.x
        log_alpha = params[:n_teams]
        log_beta = params[n_teams:2 * n_teams]

        self.alpha = {t: float(np.exp(log_alpha[i])) for i, t in enumerate(self.teams)}
        self.beta = {t: float(np.exp(log_beta[i])) for i, t in enumerate(self.teams)}
        self.gamma = float(np.exp(params[-2]))
        self.rho = float(np.clip(params[-1], -0.99, 0.99))
        self.fitted = True
        self.fitted_at = reference_dt

        logger.info(
            f"Dixon-Coles fitted. gamma={self.gamma:.3f}, rho={self.rho:.3f}, "
            f"nll={result.fun:.2f}"
        )

    # -----------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------

    def expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        """Returns (mu_home, lambda_away) expected goals."""
        alpha_h = self.alpha.get(home_team, self._league_avg_alpha())
        beta_h = self.beta.get(home_team, self._league_avg_beta())
        alpha_a = self.alpha.get(away_team, self._league_avg_alpha())
        beta_a = self.beta.get(away_team, self._league_avg_beta())

        mu = alpha_h * beta_a * self.gamma
        lam = alpha_a * beta_h
        return mu, lam

    def score_matrix(self, home_team: str, away_team: str, max_goals: int = 8) -> np.ndarray:
        """
        Joint probability matrix P(home=i, away=j).
        Shape: (max_goals+1, max_goals+1)
        """
        mu, lam = self.expected_goals(home_team, away_team)
        matrix = np.zeros((max_goals + 1, max_goals + 1))

        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                p = poisson.pmf(i, mu) * poisson.pmf(j, lam)
                p *= tau(i, j, mu, lam, self.rho)
                matrix[i, j] = max(p, 0)

        # Normalise (tau can slightly shift probabilities off 1)
        matrix /= matrix.sum()
        return matrix

    def predict_1x2(self, home_team: str, away_team: str) -> dict[str, float]:
        """Returns dict with homeWin, draw, awayWin probabilities."""
        m = self.score_matrix(home_team, away_team)
        home_win = float(np.sum(np.tril(m, -1)))
        draw = float(np.trace(m))
        away_win = float(np.sum(np.triu(m, 1)))
        total = home_win + draw + away_win
        return {
            "homeWin": home_win / total,
            "draw": draw / total,
            "awayWin": away_win / total,
        }

    def predict_score(self, home_team: str, away_team: str) -> tuple[int, int]:
        """Returns most probable scoreline."""
        m = self.score_matrix(home_team, away_team)
        idx = np.unravel_index(np.argmax(m), m.shape)
        return int(idx[0]), int(idx[1])

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save(self):
        ARTIFACTS_DIR.mkdir(exist_ok=True)
        data = {
            "teams": self.teams,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "rho": self.rho,
            "xi": self.xi,
            "fitted_at": self.fitted_at.isoformat() if self.fitted_at else None,
        }
        PARAMS_FILE.write_text(json.dumps(data, indent=2))
        logger.info(f"Dixon-Coles params saved to {PARAMS_FILE}")

    @classmethod
    def load(cls) -> "DixonColesModel":
        if not PARAMS_FILE.exists():
            raise FileNotFoundError(f"No saved model at {PARAMS_FILE}. Run training first.")
        data = json.loads(PARAMS_FILE.read_text())
        model = cls(xi=data["xi"])
        model.teams = data["teams"]
        model.alpha = data["alpha"]
        model.beta = data["beta"]
        model.gamma = data["gamma"]
        model.rho = data["rho"]
        model.fitted = True
        model.fitted_at = datetime.fromisoformat(data["fitted_at"]) if data["fitted_at"] else None
        return model

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _league_avg_alpha(self) -> float:
        return float(np.mean(list(self.alpha.values()))) if self.alpha else 1.0

    def _league_avg_beta(self) -> float:
        return float(np.mean(list(self.beta.values()))) if self.beta else 1.0

    def get_team_ratings(self) -> list[dict]:
        """Attack/defense ratings for all teams, sorted by attack desc."""
        return sorted(
            [
                {
                    "team": t,
                    "attack": round(self.alpha.get(t, 1.0), 3),
                    "defense": round(self.beta.get(t, 1.0), 3),
                }
                for t in self.teams
            ],
            key=lambda x: x["attack"],
            reverse=True,
        )
