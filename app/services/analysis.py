from __future__ import annotations
"""
Match highlights generator — produces emoji bullet points per match.
Each highlight surfaces a genuinely notable stat (good form, weak away record, etc.)
Only shown if the value is actually remarkable, not for average situations.
"""
import json
import logging
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.models import MatchAnalysis

logger = logging.getLogger(__name__)

# Stored as JSON in DB; prefix distinguishes from old plain-text format
_JSON_PREFIX = "["


def generate_highlights(
    home_team: str,
    away_team: str,
    prediction: dict,
    features: dict,
    dc_alpha: dict,
    dc_beta: dict,
) -> list[dict]:
    """
    Return a list of {emoji, text} dicts for the most notable match stats.
    Only includes a highlight if the value is genuinely remarkable.
    """
    probs = prediction["probabilities"]
    xg = prediction["expectedGoals"]
    xg_home, xg_away = xg["home"], xg["away"]

    elo_diff        = features.get("elo_diff", 0)
    h_win_streak    = int(features.get("h_win_streak", 0))
    a_win_streak    = int(features.get("a_win_streak", 0))
    h_unbeaten      = int(features.get("h_unbeaten", 0))
    a_unbeaten      = int(features.get("a_unbeaten", 0))
    h_short_pts     = features.get("h_short_form_pts", 1.0)
    a_short_pts     = features.get("a_short_form_pts", 1.0)
    h_short_streak  = int(features.get("h_short_win_streak", 0))
    a_short_streak  = int(features.get("a_short_win_streak", 0))
    h_home_pts      = features.get("h_home_pts", 1.0)
    a_away_pts      = features.get("a_away_pts", 1.0)
    h_home_gf       = features.get("h_home_gf", 1.3)
    h_home_ga       = features.get("h_home_ga", 1.3)
    a_away_gf       = features.get("a_away_gf", 1.3)
    a_away_ga       = features.get("a_away_ga", 1.3)
    h2h_h           = features.get("h2h_home_wins", 0.33)
    h2h_a           = features.get("h2h_away_wins", 0.33)
    h2h_n           = int(features.get("h2h_matches", 0))
    h_rest          = int(features.get("h_days_rest", 7))
    a_rest          = int(features.get("a_days_rest", 7))
    h_pos           = int(features.get("h_position", 9))
    a_pos           = int(features.get("a_position", 9))
    h_atk           = dc_alpha.get(home_team, 1.0)
    a_atk           = dc_alpha.get(away_team, 1.0)
    h_def           = dc_beta.get(home_team, 1.0)
    a_def           = dc_beta.get(away_team, 1.0)

    # scored candidates: (score, emoji, text)
    candidates: list[tuple[float, str, str]] = []

    def add(score: float, emoji: str, text: str):
        candidates.append((score, emoji, text))

    # ── Win streaks ──────────────────────────────────────────────────────────
    if h_win_streak >= 5:
        add(h_win_streak, "🔥", f"{home_team}: {h_win_streak} Siege in Serie")
    elif h_short_streak == 3:
        add(3.0, "🔥", f"{home_team}: Letzte 3 Spiele alle gewonnen")
    elif h_win_streak >= 3:
        add(h_win_streak, "📈", f"{home_team}: {h_win_streak} Siege in Folge")

    if a_win_streak >= 5:
        add(a_win_streak, "🔥", f"{away_team}: {a_win_streak} Siege in Serie")
    elif a_short_streak == 3:
        add(3.0, "🔥", f"{away_team}: Letzte 3 Spiele alle gewonnen")
    elif a_win_streak >= 3:
        add(a_win_streak, "📈", f"{away_team}: {a_win_streak} Siege in Folge")

    # ── Unbeaten streaks (only if no win streak already added) ───────────────
    if h_unbeaten >= 5 and h_win_streak < 3:
        add(h_unbeaten * 0.8, "🛡️", f"{home_team}: {h_unbeaten} Spiele ungeschlagen")
    if a_unbeaten >= 5 and a_win_streak < 3:
        add(a_unbeaten * 0.8, "🛡️", f"{away_team}: {a_unbeaten} Spiele ungeschlagen")

    # ── Recent slump ─────────────────────────────────────────────────────────
    if h_short_pts == 0.0:
        add(2.5, "📉", f"{home_team}: 3 Spiele in Folge ohne Sieg")
    elif h_short_pts <= 0.5 and h_win_streak == 0:
        add(2.0, "📉", f"{home_team}: Schwache Kurzform ({h_short_pts:.1f} Pkt/Sp)")

    if a_short_pts == 0.0:
        add(2.5, "📉", f"{away_team}: 3 Spiele in Folge ohne Sieg")
    elif a_short_pts <= 0.5 and a_win_streak == 0:
        add(2.0, "📉", f"{away_team}: Schwache Kurzform ({a_short_pts:.1f} Pkt/Sp)")

    # ── Home form ────────────────────────────────────────────────────────────
    if h_home_pts >= 2.6:
        add(h_home_pts, "🏠", f"{home_team} zuhause stark: {h_home_pts:.1f} Pkt/Sp")
    elif h_home_pts <= 0.6:
        add(2.5 - h_home_pts, "⚠️", f"{home_team} zuhause anfällig: {h_home_pts:.1f} Pkt/Sp")

    if h_home_ga <= 0.6:
        add(2.0, "🔒", f"{home_team}: Kaum Gegentore zuhause ({h_home_ga:.1f}/Sp)")
    elif h_home_gf >= 2.5:
        add(h_home_gf * 0.7, "⚽", f"{home_team}: Treffsicher zuhause ({h_home_gf:.1f} Tore/Sp)")

    # ── Away form ────────────────────────────────────────────────────────────
    if a_away_pts >= 2.0:
        add(a_away_pts * 1.2, "✈️", f"{away_team} auswärts stark: {a_away_pts:.1f} Pkt/Sp")
    elif a_away_pts <= 0.5:
        add(2.5 - a_away_pts, "⚠️", f"{away_team} auswärts schwach: {a_away_pts:.1f} Pkt/Sp")

    if a_away_gf >= 2.2:
        add(a_away_gf * 0.6, "⚽", f"{away_team}: Treffsicher auswärts ({a_away_gf:.1f} Tore/Sp)")
    elif a_away_ga <= 0.6:
        add(1.8, "🔒", f"{away_team}: Wenig Gegentore auswärts ({a_away_ga:.1f}/Sp)")

    # ── xG highlights ────────────────────────────────────────────────────────
    if xg_home >= 2.2:
        add(xg_home * 0.8, "⚽", f"{home_team}: {xg_home:.2f} xG erwartet – offensiv dominant")
    if xg_away <= 0.7:
        add(1.5, "🔒", f"{away_team}: Nur {xg_away:.2f} xG – kaum Chancen erwartet")
    elif xg_away >= 2.0:
        add(xg_away * 0.8, "⚽", f"{away_team}: {xg_away:.2f} xG – gefährlich für die Abwehr")

    # ── H2H dominance ────────────────────────────────────────────────────────
    if h2h_n >= 5 and h2h_h >= 0.6:
        add((h2h_h - 0.33) * 4, "📊",
            f"Direktvergleich: {home_team} gewinnt {h2h_h*100:.0f}% der letzten {h2h_n} Duelle")
    elif h2h_n >= 5 and h2h_a >= 0.6:
        add((h2h_a - 0.33) * 4, "📊",
            f"Direktvergleich: {away_team} gewinnt {h2h_a*100:.0f}% der letzten {h2h_n} Duelle")

    # ── Elo gap ──────────────────────────────────────────────────────────────
    if abs(elo_diff) >= 200:
        stronger = home_team if elo_diff > 0 else away_team
        add(abs(elo_diff) / 80, "💪",
            f"{stronger} klar favorisiert: Elo-Vorsprung von {abs(int(elo_diff))} Punkten")

    # ── Table gap ────────────────────────────────────────────────────────────
    pos_diff = abs(h_pos - a_pos)
    if pos_diff >= 12:
        higher = home_team if h_pos < a_pos else away_team
        lower = away_team if h_pos < a_pos else home_team
        add(pos_diff / 4, "📋",
            f"{pos_diff} Tabellenplätze trennen {higher} (#{min(h_pos, a_pos)}) und {lower} (#{max(h_pos, a_pos)})")

    # ── Rest / fatigue ───────────────────────────────────────────────────────
    rest_diff = h_rest - a_rest
    if rest_diff <= -3:
        add(abs(rest_diff) * 0.4, "😴",
            f"Frischevorteil {away_team}: {a_rest} vs. {h_rest} Ruhetage")
    elif rest_diff >= 3:
        add(rest_diff * 0.4, "😴",
            f"Frischevorteil {home_team}: {h_rest} vs. {a_rest} Ruhetage")

    # ── Model agreement ──────────────────────────────────────────────────────
    if prediction.get("modelAgreement") and prediction.get("confidenceLevel") == "high":
        add(1.2, "🤝", "Beide KI-Modelle kommen unabhängig zum gleichen Tipp")

    # ── Select top highlights (max 5, diverse) ───────────────────────────────
    candidates.sort(key=lambda x: x[0], reverse=True)

    # Avoid showing two highlights about the same team's same aspect
    seen_teams: dict[str, int] = {}
    result = []
    for _, emoji, text in candidates:
        # Count how many times this team already appears
        team_in_text = home_team if home_team in text else (away_team if away_team in text else None)
        if team_in_text:
            if seen_teams.get(team_in_text, 0) >= 2:
                continue
            seen_teams[team_in_text] = seen_teams.get(team_in_text, 0) + 1
        result.append({"emoji": emoji, "text": text})
        if len(result) >= 5:
            break

    return result


async def generate_all_analyses(
    matches_data: list[dict],
    matchday: int,
    session: AsyncSession,
) -> dict[int, list[dict]]:
    """Return highlights for all matches. DB-cached, generated locally (free)."""
    match_ids = [m["match_id"] for m in matches_data]

    result = await session.execute(
        select(MatchAnalysis).where(MatchAnalysis.match_id.in_(match_ids))
    )
    # Only use cached entries that are JSON (new format)
    cached: dict[int, list[dict]] = {}
    for row in result.scalars().all():
        if row.analysis_text.startswith(_JSON_PREFIX):
            try:
                parsed = json.loads(row.analysis_text)
                if parsed:  # skip empty cached arrays — regenerate them
                    cached[row.match_id] = parsed
            except Exception:
                pass

    missing = [m for m in matches_data if m["match_id"] not in cached]
    if not missing:
        return cached

    logger.info(f"Generating highlights for {len(missing)} matches (matchday {matchday})")

    for m in missing:
        highlights = generate_highlights(
            m["home_team"], m["away_team"],
            m["prediction"], m["features"],
            m["dc_alpha"], m["dc_beta"],
        )
        json_str = json.dumps(highlights, ensure_ascii=False)
        # Upsert
        existing = await session.execute(
            select(MatchAnalysis).where(MatchAnalysis.match_id == m["match_id"])
        )
        row = existing.scalar_one_or_none()
        if row:
            row.analysis_text = json_str
        else:
            session.add(MatchAnalysis(
                match_id=m["match_id"],
                matchday=matchday,
                analysis_text=json_str,
            ))
        cached[m["match_id"]] = highlights

    await session.commit()
    return cached
