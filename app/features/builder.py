from __future__ import annotations
"""
Feature engineering for the XGBoost / LightGBM models.
All features are computed from match data available BEFORE the match
(no data leakage).

New vs. original:
  - dc_model features: h_dc_attack / defense, dc_mu / dc_lam / dc_mu_lam_ratio
  - form_trend:  short-term vs. long-term momentum direction
  - clean_sheet_rate: defensive consistency
  - goals_var / conceded_var: scoring consistency
  - pythagorean: Pythagorean win expectation
  - Elo season-start regression toward the mean
"""
import logging
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from app.models.dixon_coles import DixonColesModel

logger = logging.getLogger(__name__)

FORM_WINDOW = 5      # last N matches for form
FORM_SHORT = 3       # last N matches for momentum
FORM_LONG = 10       # last N matches for long-term form
H2H_WINDOW = 8       # last N head-to-head meetings
PYTH_WINDOW = 20     # last N matches for Pythagorean expectation

ELO_K = 20.0               # Elo K-factor
ELO_HOME_ADVANTAGE = 80.0  # Elo home advantage in rating points
ELO_DEFAULT = 1500.0       # Starting Elo for new teams
ELO_SEASON_REGRESSION = 0.30  # How much to regress toward mean each season start


def build_feature_matrix(matches: list[dict], dc_model: "DixonColesModel | None" = None) -> pd.DataFrame:
    """
    Build a feature matrix from a list of match dicts (sorted by date ascending).
    Each row represents one match with features computed from prior matches only.

    match dict keys: match_id, home_team, away_team, home_goals, away_goals,
                     match_datetime, season, matchday

    dc_model: optional fitted DixonColesModel; when provided its attack/defense
              parameters and expected goals are added as features.
    """
    df = pd.DataFrame(matches)
    df = df[df["home_goals"].notna() & df["away_goals"].notna()].copy()
    df["match_datetime"] = pd.to_datetime(df["match_datetime"])
    df = df.sort_values("match_datetime").reset_index(drop=True)

    elo_before: dict[str, float] = {}
    last_season = None

    records = []
    for idx, row in df.iterrows():
        # Apply Elo season-start regression at season boundaries
        season = row.get("season")
        if last_season is not None and season != last_season:
            _apply_season_elo_regression(elo_before)
        last_season = season

        prior = df.iloc[:idx]
        h_elo = elo_before.get(row["home_team"], ELO_DEFAULT)
        a_elo = elo_before.get(row["away_team"], ELO_DEFAULT)

        features = _compute_features(row, prior, h_elo=h_elo, a_elo=a_elo, dc_model=dc_model)
        features["match_id"] = row["match_id"]
        features["home_goals"] = int(row["home_goals"])
        features["away_goals"] = int(row["away_goals"])
        features["outcome"] = _outcome(row["home_goals"], row["away_goals"])
        records.append(features)

        _update_elo(elo_before, row["home_team"], row["away_team"],
                    int(row["home_goals"]), int(row["away_goals"]))

    return pd.DataFrame(records)


def compute_prediction_features(
    home_team: str,
    away_team: str,
    all_matches: list[dict],
    match_datetime: datetime | None = None,
    dc_model: "DixonColesModel | None" = None,
) -> dict:
    """
    Compute features for an upcoming match (no result available).
    all_matches: all historical finished matches.
    dc_model: optional fitted DixonColesModel for DC-based features.
    """
    df = pd.DataFrame(all_matches)
    df = df[df["home_goals"].notna() & df["away_goals"].notna()].copy()
    df["match_datetime"] = pd.to_datetime(df["match_datetime"])
    df = df.sort_values("match_datetime")

    # Replay all Elo updates (with season-start regression) to get current ratings
    elo: dict[str, float] = {}
    last_season = None
    for _, row in df.iterrows():
        season = row.get("season")
        if last_season is not None and season != last_season:
            _apply_season_elo_regression(elo)
        last_season = season
        _update_elo(elo, row["home_team"], row["away_team"],
                    int(row["home_goals"]), int(row["away_goals"]))

    h_elo = elo.get(home_team, ELO_DEFAULT)
    a_elo = elo.get(away_team, ELO_DEFAULT)

    fake_row = {
        "home_team": home_team,
        "away_team": away_team,
        "match_datetime": match_datetime or datetime.utcnow(),
        "season": df["season"].max() if len(df) else 2025,
        "matchday": 99,
    }
    return _compute_features(fake_row, df, h_elo=h_elo, a_elo=a_elo, dc_model=dc_model)


# ---------------------------------------------------------------------------
# Elo helpers
# ---------------------------------------------------------------------------

def _apply_season_elo_regression(elo: dict[str, float]) -> None:
    """Regress all Elo ratings toward the mean at season start.
    This prevents compounding errors and better reflects squad changes."""
    if not elo:
        return
    mean = float(np.mean(list(elo.values())))
    for team in elo:
        elo[team] = elo[team] * (1 - ELO_SEASON_REGRESSION) + mean * ELO_SEASON_REGRESSION


def _update_elo(
    elo: dict[str, float],
    home_team: str,
    away_team: str,
    home_goals: int,
    away_goals: int,
) -> None:
    """Update Elo ratings in-place after a match result."""
    h = elo.get(home_team, ELO_DEFAULT)
    a = elo.get(away_team, ELO_DEFAULT)

    # Expected score with home advantage
    h_exp = 1.0 / (1.0 + 10.0 ** ((a - h - ELO_HOME_ADVANTAGE) / 400.0))
    a_exp = 1.0 - h_exp

    # Actual outcome
    if home_goals > away_goals:
        h_score, a_score = 1.0, 0.0
    elif home_goals == away_goals:
        h_score, a_score = 0.5, 0.5
    else:
        h_score, a_score = 0.0, 1.0

    # Goal margin multiplier (capped at 3 goals)
    margin = 1.0 + 0.1 * min(abs(home_goals - away_goals), 3)

    elo[home_team] = h + ELO_K * margin * (h_score - h_exp)
    elo[away_team] = a + ELO_K * margin * (a_score - a_exp)


# ---------------------------------------------------------------------------
# Core feature computation
# ---------------------------------------------------------------------------

def _compute_features(
    row: dict | pd.Series,
    prior: pd.DataFrame,
    h_elo: float = ELO_DEFAULT,
    a_elo: float = ELO_DEFAULT,
    dc_model: "DixonColesModel | None" = None,
) -> dict:
    home = row["home_team"]
    away = row["away_team"]
    dt = row["match_datetime"]
    season = row.get("season", 2025)

    f = {}

    # --- Elo ratings ---
    f["h_elo"] = h_elo
    f["a_elo"] = a_elo
    f["elo_diff"] = h_elo - a_elo  # positive = home stronger

    # --- Form features ---
    f.update(_form_features(home, prior, prefix="h_", window=FORM_WINDOW))
    f.update(_form_features(away, prior, prefix="a_", window=FORM_WINDOW))

    # --- Short-term momentum (last 3) ---
    f.update(_form_features(home, prior, prefix="h_short_", window=FORM_SHORT))
    f.update(_form_features(away, prior, prefix="a_short_", window=FORM_SHORT))

    # --- Long-term form (last 10) ---
    h_long = _form_pts_only(home, prior, window=FORM_LONG)
    a_long = _form_pts_only(away, prior, window=FORM_LONG)
    f["h_long_form_pts"] = h_long
    f["a_long_form_pts"] = a_long

    # --- Home/away specific form ---
    f.update(_venue_form(home, prior, venue="home", prefix="h_home_"))
    f.update(_venue_form(away, prior, venue="away", prefix="a_away_"))

    # --- Season stats ---
    f.update(_season_stats(home, prior, season, prefix="h_"))
    f.update(_season_stats(away, prior, season, prefix="a_"))

    # --- Head-to-head ---
    f.update(_head_to_head(home, away, prior))

    # --- Position / table context ---
    f.update(_table_position(home, away, prior, season))

    # --- Match context ---
    f.update(_match_context(home, away, prior, dt))

    # --- Form trend (short vs. long momentum direction) ---
    f["h_form_trend"] = f["h_short_form_pts"] - f["h_long_form_pts"]
    f["a_form_trend"] = f["a_short_form_pts"] - f["a_long_form_pts"]
    f["form_trend_diff"] = f["h_form_trend"] - f["a_form_trend"]

    # --- Defensive consistency (clean sheet rate) ---
    f.update(_clean_sheet_rate(home, prior, prefix="h_"))
    f.update(_clean_sheet_rate(away, prior, prefix="a_"))

    # --- Scoring consistency (variance) ---
    f.update(_goals_consistency(home, prior, prefix="h_"))
    f.update(_goals_consistency(away, prior, prefix="a_"))

    # --- Pythagorean win expectation ---
    h_pyth = _pythagorean(home, prior)
    a_pyth = _pythagorean(away, prior)
    f["h_pythagorean"] = h_pyth
    f["a_pythagorean"] = a_pyth
    f["pythagorean_diff"] = h_pyth - a_pyth

    # --- Dixon-Coles model-derived features ---
    f.update(_dc_features(home, away, dc_model))

    # --- Differential features (explicit relative strength) ---
    f["form_pts_diff"] = f["h_form_pts"] - f["a_form_pts"]
    f["season_pts_pg_diff"] = f["h_season_pts_pg"] - f["a_season_pts_pg"]
    f["season_gd_diff"] = (f["h_season_gf_pg"] - f["h_season_ga_pg"]) - (f["a_season_gf_pg"] - f["a_season_ga_pg"])
    f["home_away_pts_diff"] = f["h_home_pts"] - f["a_away_pts"]
    f["short_form_diff"] = f["h_short_form_pts"] - f["a_short_form_pts"]

    return f


def _form_features(team: str, prior: pd.DataFrame, prefix: str, window: int) -> dict:
    """Rolling form over last `window` matches regardless of venue."""
    team_matches = prior[
        (prior["home_team"] == team) | (prior["away_team"] == team)
    ].tail(window)

    if len(team_matches) == 0:
        return {
            f"{prefix}form_pts": 1.0, f"{prefix}form_gf": 1.3,
            f"{prefix}form_ga": 1.3, f"{prefix}form_gd": 0.0,
            f"{prefix}win_streak": 0, f"{prefix}unbeaten": 0,
        }

    pts, gf, ga = [], [], []
    for _, m in team_matches.iterrows():
        is_home = m["home_team"] == team
        g_for = m["home_goals"] if is_home else m["away_goals"]
        g_ag = m["away_goals"] if is_home else m["home_goals"]
        gf.append(g_for)
        ga.append(g_ag)
        if g_for > g_ag:
            pts.append(3)
        elif g_for == g_ag:
            pts.append(1)
        else:
            pts.append(0)

    # Win streak (from most recent)
    streak = 0
    for p in reversed(pts):
        if p == 3:
            streak += 1
        else:
            break

    unbeaten = 0
    for p in reversed(pts):
        if p > 0:
            unbeaten += 1
        else:
            break

    return {
        f"{prefix}form_pts": float(np.mean(pts)),
        f"{prefix}form_gf": float(np.mean(gf)),
        f"{prefix}form_ga": float(np.mean(ga)),
        f"{prefix}form_gd": float(np.mean(gf) - np.mean(ga)),
        f"{prefix}win_streak": float(streak),
        f"{prefix}unbeaten": float(unbeaten),
    }


def _form_pts_only(team: str, prior: pd.DataFrame, window: int) -> float:
    """Average points over last `window` matches."""
    team_matches = prior[
        (prior["home_team"] == team) | (prior["away_team"] == team)
    ].tail(window)

    if len(team_matches) == 0:
        return 1.0

    pts = []
    for _, m in team_matches.iterrows():
        is_home = m["home_team"] == team
        gf = m["home_goals"] if is_home else m["away_goals"]
        ga = m["away_goals"] if is_home else m["home_goals"]
        pts.append(3 if gf > ga else (1 if gf == ga else 0))

    return float(np.mean(pts))


def _venue_form(team: str, prior: pd.DataFrame, venue: str, prefix: str) -> dict:
    """Form only at home / only away over last FORM_WINDOW such matches."""
    if venue == "home":
        matches = prior[prior["home_team"] == team].tail(FORM_WINDOW)
        gf_col, ga_col = "home_goals", "away_goals"
    else:
        matches = prior[prior["away_team"] == team].tail(FORM_WINDOW)
        gf_col, ga_col = "away_goals", "home_goals"

    if len(matches) == 0:
        return {f"{prefix}pts": 1.0, f"{prefix}gf": 1.3, f"{prefix}ga": 1.3}

    pts = []
    for _, m in matches.iterrows():
        gf, ga = m[gf_col], m[ga_col]
        pts.append(3 if gf > ga else (1 if gf == ga else 0))

    return {
        f"{prefix}pts": float(np.mean(pts)),
        f"{prefix}gf": float(matches[gf_col].mean()),
        f"{prefix}ga": float(matches[ga_col].mean()),
    }


def _season_stats(team: str, prior: pd.DataFrame, season: int, prefix: str) -> dict:
    """Current season stats up to this point."""
    season_matches = prior[
        (prior["season"] == season)
        & ((prior["home_team"] == team) | (prior["away_team"] == team))
    ]

    if len(season_matches) == 0:
        return {
            f"{prefix}season_pts_pg": 1.0, f"{prefix}season_gf_pg": 1.3,
            f"{prefix}season_ga_pg": 1.3, f"{prefix}season_matches": 0,
        }

    pts, gf, ga = [], [], []
    for _, m in season_matches.iterrows():
        is_home = m["home_team"] == team
        g_for = m["home_goals"] if is_home else m["away_goals"]
        g_ag = m["away_goals"] if is_home else m["home_goals"]
        gf.append(g_for)
        ga.append(g_ag)
        pts.append(3 if g_for > g_ag else (1 if g_for == g_ag else 0))

    n = len(pts)
    return {
        f"{prefix}season_pts_pg": float(np.sum(pts) / n),
        f"{prefix}season_gf_pg": float(np.mean(gf)),
        f"{prefix}season_ga_pg": float(np.mean(ga)),
        f"{prefix}season_matches": float(n),
    }


def _head_to_head(home: str, away: str, prior: pd.DataFrame) -> dict:
    """Head-to-head record between these two teams."""
    h2h = prior[
        ((prior["home_team"] == home) & (prior["away_team"] == away))
        | ((prior["home_team"] == away) & (prior["away_team"] == home))
    ].tail(H2H_WINDOW)

    if len(h2h) == 0:
        return {
            "h2h_home_wins": 0.33, "h2h_draws": 0.33, "h2h_away_wins": 0.33,
            "h2h_home_gf": 1.3, "h2h_away_gf": 1.3, "h2h_matches": 0,
        }

    home_wins, draws, away_wins = 0, 0, 0
    home_gf, away_gf = [], []

    for _, m in h2h.iterrows():
        if m["home_team"] == home:
            hg, ag = m["home_goals"], m["away_goals"]
        else:
            hg, ag = m["away_goals"], m["home_goals"]
        home_gf.append(hg)
        away_gf.append(ag)
        if hg > ag:
            home_wins += 1
        elif hg == ag:
            draws += 1
        else:
            away_wins += 1

    n = len(h2h)
    return {
        "h2h_home_wins": home_wins / n,
        "h2h_draws": draws / n,
        "h2h_away_wins": away_wins / n,
        "h2h_home_gf": float(np.mean(home_gf)),
        "h2h_away_gf": float(np.mean(away_gf)),
        "h2h_matches": float(n),
    }


def _table_position(home: str, away: str, prior: pd.DataFrame, season: int) -> dict:
    """Approximate table positions from season stats."""
    season_data = prior[prior["season"] == season]
    teams = set(season_data["home_team"]) | set(season_data["away_team"])

    points: dict[str, int] = {}
    for team in teams:
        tm = season_data[(season_data["home_team"] == team) | (season_data["away_team"] == team)]
        pts = 0
        for _, m in tm.iterrows():
            is_home = m["home_team"] == team
            gf = m["home_goals"] if is_home else m["away_goals"]
            ga = m["away_goals"] if is_home else m["home_goals"]
            pts += 3 if gf > ga else (1 if gf == ga else 0)
        points[team] = pts

    if not points:
        return {"h_position": 9.0, "a_position": 9.0, "position_diff": 0.0}

    sorted_teams = sorted(points, key=lambda t: points[t], reverse=True)
    pos = {t: i + 1 for i, t in enumerate(sorted_teams)}

    h_pos = float(pos.get(home, 9))
    a_pos = float(pos.get(away, 9))
    return {
        "h_position": h_pos,
        "a_position": a_pos,
        "position_diff": h_pos - a_pos,  # negative = home team higher
    }


def _match_context(home: str, away: str, prior: pd.DataFrame, dt: datetime) -> dict:
    """Fatigue and fixture density features."""
    def days_since_last(team: str) -> float:
        tm = prior[(prior["home_team"] == team) | (prior["away_team"] == team)]
        if len(tm) == 0:
            return 7.0
        last = pd.to_datetime(tm["match_datetime"]).max()
        return max((dt - last).days, 0) if pd.notna(last) else 7.0

    h_rest = days_since_last(home)
    a_rest = days_since_last(away)

    return {
        "h_days_rest": float(min(h_rest, 21)),
        "a_days_rest": float(min(a_rest, 21)),
        "rest_advantage": float(min(h_rest, 21) - min(a_rest, 21)),
    }


def _clean_sheet_rate(team: str, prior: pd.DataFrame, prefix: str, window: int = FORM_WINDOW) -> dict:
    """Fraction of recent matches where team kept a clean sheet."""
    matches = prior[(prior["home_team"] == team) | (prior["away_team"] == team)].tail(window)

    if len(matches) == 0:
        return {f"{prefix}clean_sheet_rate": 0.3}

    cs = 0
    for _, m in matches.iterrows():
        ga = m["away_goals"] if m["home_team"] == team else m["home_goals"]
        if ga == 0:
            cs += 1

    return {f"{prefix}clean_sheet_rate": float(cs / len(matches))}


def _goals_consistency(team: str, prior: pd.DataFrame, prefix: str, window: int = FORM_WINDOW) -> dict:
    """Standard deviation of goals scored and conceded in last `window` matches.
    Lower variance = more predictable team."""
    matches = prior[(prior["home_team"] == team) | (prior["away_team"] == team)].tail(window)

    if len(matches) < 2:
        return {f"{prefix}goals_var": 1.0, f"{prefix}conceded_var": 1.0}

    gf, ga = [], []
    for _, m in matches.iterrows():
        is_home = m["home_team"] == team
        gf.append(float(m["home_goals"] if is_home else m["away_goals"]))
        ga.append(float(m["away_goals"] if is_home else m["home_goals"]))

    return {
        f"{prefix}goals_var": float(np.std(gf)),
        f"{prefix}conceded_var": float(np.std(ga)),
    }


def _pythagorean(team: str, prior: pd.DataFrame, window: int = PYTH_WINDOW) -> float:
    """Pythagorean win expectation: GF^exp / (GF^exp + GA^exp).
    Exponent 1.83 empirically calibrated for football."""
    matches = prior[(prior["home_team"] == team) | (prior["away_team"] == team)].tail(window)

    if len(matches) == 0:
        return 0.5

    total_gf, total_ga = 0.0, 0.0
    for _, m in matches.iterrows():
        is_home = m["home_team"] == team
        total_gf += float(m["home_goals"] if is_home else m["away_goals"])
        total_ga += float(m["away_goals"] if is_home else m["home_goals"])

    if total_gf + total_ga == 0:
        return 0.5

    exp = 1.83
    return float(total_gf ** exp / (total_gf ** exp + total_ga ** exp))


def _dc_features(home: str, away: str, dc_model: "DixonColesModel | None") -> dict:
    """
    Dixon-Coles attack/defense ratings and expected goals as features.
    These represent time-decay-weighted team strength from all historical data —
    the strongest signal available for predicting match outcomes.
    """
    if dc_model is None or not dc_model.fitted:
        # Neutral defaults (model not available)
        return {
            "h_dc_attack": 1.0, "h_dc_defense": 1.0,
            "a_dc_attack": 1.0, "a_dc_defense": 1.0,
            "dc_mu": 1.5, "dc_lam": 1.2,
            "dc_mu_lam_ratio": 1.25,
        }

    avg_alpha = dc_model._league_avg_alpha()
    avg_beta = dc_model._league_avg_beta()

    h_atk = dc_model.alpha.get(home, avg_alpha)
    h_def = dc_model.beta.get(home, avg_beta)
    a_atk = dc_model.alpha.get(away, avg_alpha)
    a_def = dc_model.beta.get(away, avg_beta)

    mu, lam = dc_model.expected_goals(home, away)

    return {
        "h_dc_attack": float(h_atk),
        "h_dc_defense": float(h_def),
        "a_dc_attack": float(a_atk),
        "a_dc_defense": float(a_def),
        "dc_mu": float(mu),
        "dc_lam": float(lam),
        "dc_mu_lam_ratio": float(mu / (lam + 1e-10)),
    }


def _outcome(home_goals: float, away_goals: float) -> int:
    """0 = home win, 1 = draw, 2 = away win."""
    if home_goals > away_goals:
        return 0
    elif home_goals == away_goals:
        return 1
    return 2
