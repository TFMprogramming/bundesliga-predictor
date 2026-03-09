from __future__ import annotations
"""
Feature engineering for the XGBoost model.
All features are computed from match data available BEFORE the match
(no data leakage).
"""
import logging
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FORM_WINDOW = 5      # last N matches for form
FORM_LONG = 10       # last N matches for long-term form
H2H_WINDOW = 8       # last N head-to-head meetings


def build_feature_matrix(matches: list[dict]) -> pd.DataFrame:
    """
    Build a feature matrix from a list of match dicts (sorted by date ascending).
    Each row represents one match with features computed from prior matches only.

    match dict keys: match_id, home_team, away_team, home_goals, away_goals,
                     match_datetime, season, matchday
    """
    df = pd.DataFrame(matches)
    df = df[df["home_goals"].notna() & df["away_goals"].notna()].copy()
    df["match_datetime"] = pd.to_datetime(df["match_datetime"])
    df = df.sort_values("match_datetime").reset_index(drop=True)

    records = []
    for idx, row in df.iterrows():
        prior = df.iloc[:idx]  # Only matches BEFORE this one
        features = _compute_features(row, prior)
        features["match_id"] = row["match_id"]
        features["home_goals"] = int(row["home_goals"])
        features["away_goals"] = int(row["away_goals"])
        features["outcome"] = _outcome(row["home_goals"], row["away_goals"])
        records.append(features)

    return pd.DataFrame(records)


def compute_prediction_features(
    home_team: str,
    away_team: str,
    all_matches: list[dict],
    match_datetime: datetime | None = None,
) -> dict:
    """
    Compute features for an upcoming match (no result available).
    all_matches: all historical finished matches.
    """
    df = pd.DataFrame(all_matches)
    df = df[df["home_goals"].notna() & df["away_goals"].notna()].copy()
    df["match_datetime"] = pd.to_datetime(df["match_datetime"])
    df = df.sort_values("match_datetime")

    fake_row = {
        "home_team": home_team,
        "away_team": away_team,
        "match_datetime": match_datetime or datetime.utcnow(),
        "season": df["season"].max() if len(df) else 2025,
        "matchday": 99,
    }
    return _compute_features(fake_row, df)


# ---------------------------------------------------------------------------
# Core feature computation
# ---------------------------------------------------------------------------

def _compute_features(row: dict | pd.Series, prior: pd.DataFrame) -> dict:
    home = row["home_team"]
    away = row["away_team"]
    dt = row["match_datetime"]
    season = row.get("season", 2025)

    f = {}

    # --- Form features ---
    f.update(_form_features(home, prior, "home", prefix="h_"))
    f.update(_form_features(away, prior, "away", prefix="a_"))

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

    return f


def _form_features(team: str, prior: pd.DataFrame, venue: str, prefix: str) -> dict:
    """Rolling form over last FORM_WINDOW matches regardless of venue."""
    team_matches = prior[
        (prior["home_team"] == team) | (prior["away_team"] == team)
    ].tail(FORM_WINDOW)

    if len(team_matches) == 0:
        return {f"{prefix}form_pts": 1.0, f"{prefix}form_gf": 1.3, f"{prefix}form_ga": 1.3,
                f"{prefix}form_gd": 0.0, f"{prefix}win_streak": 0, f"{prefix}unbeaten": 0}

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
        return {f"{prefix}season_pts_pg": 1.0, f"{prefix}season_gf_pg": 1.3,
                f"{prefix}season_ga_pg": 1.3, f"{prefix}season_matches": 0}

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
        return {"h2h_home_wins": 0.33, "h2h_draws": 0.33, "h2h_away_wins": 0.33,
                "h2h_home_gf": 1.3, "h2h_away_gf": 1.3, "h2h_matches": 0}

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


def _outcome(home_goals: float, away_goals: float) -> int:
    """0 = home win, 1 = draw, 2 = away win."""
    if home_goals > away_goals:
        return 0
    elif home_goals == away_goals:
        return 1
    return 2
