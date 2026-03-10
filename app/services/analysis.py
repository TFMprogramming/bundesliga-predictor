from __future__ import annotations
"""
AI-generated match analysis using Claude.
Each match gets a unique, natural-language explanation of why the model predicts what it does.
"""
import asyncio
import logging
from functools import lru_cache

import anthropic

from app.config import settings

logger = logging.getLogger(__name__)

# In-memory cache: (matchday, matchId) -> analysis text
_analysis_cache: dict[tuple[int, int], str] = {}


@lru_cache(maxsize=1)
def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)


def _build_prompt(
    home_team: str,
    away_team: str,
    prediction: dict,
    features: dict,
    dc_alpha: dict,
    dc_beta: dict,
) -> str:
    probs = prediction["probabilities"]
    score = prediction["predictedScore"]
    xg = prediction["expectedGoals"]

    h_pos = int(features.get("h_position", 9))
    a_pos = int(features.get("a_position", 9))
    h_elo = int(features.get("h_elo", 1500))
    a_elo = int(features.get("a_elo", 1500))
    elo_diff = int(features.get("elo_diff", 0))
    h_form = features.get("h_form_pts", 1.0)
    a_form = features.get("a_form_pts", 1.0)
    h_short = features.get("h_short_form_pts", 1.0)
    a_short = features.get("a_short_form_pts", 1.0)
    h_home = features.get("h_home_pts", 1.0)
    a_away = features.get("a_away_pts", 1.0)
    h2h_h = features.get("h2h_home_wins", 0.33)
    h2h_d = features.get("h2h_draws", 0.33)
    h2h_a = features.get("h2h_away_wins", 0.33)
    h_rest = int(features.get("h_days_rest", 7))
    a_rest = int(features.get("a_days_rest", 7))
    h_season = features.get("h_season_pts_pg", 1.0)
    a_season = features.get("a_season_pts_pg", 1.0)
    h_atk = dc_alpha.get(home_team, 1.0)
    a_atk = dc_alpha.get(away_team, 1.0)
    h_def = dc_beta.get(home_team, 1.0)
    a_def = dc_beta.get(away_team, 1.0)

    winner_label = (
        f"{home_team}-Sieg" if probs["homeWin"] > probs["awayWin"] and probs["homeWin"] > probs["draw"]
        else f"{away_team}-Sieg" if probs["awayWin"] > probs["homeWin"] and probs["awayWin"] > probs["draw"]
        else "Unentschieden"
    )

    return f"""Du bist ein Fußball-Analyst. Erkläre in 3 prägnanten Sätzen auf Deutsch, warum das KI-Modell für dieses Bundesliga-Spiel so tippt.

Spiel: {home_team} vs. {away_team} (Saison 2025/26)

Modell-Prognose:
- Wahrscheinlichstes Ergebnis: {score['home']}:{score['away']} ({winner_label})
- Wahrscheinlichkeiten: Heimsieg {probs['homeWin']*100:.0f}% | X {probs['draw']*100:.0f}% | Auswärtssieg {probs['awayWin']*100:.0f}%
- Expected Goals: {home_team} {xg['home']:.2f} | {away_team} {xg['away']:.2f}

Schlüsselkennzahlen:
- Tabelle: {home_team} Platz {h_pos} | {away_team} Platz {a_pos}
- Elo-Stärke: {home_team} {h_elo} | {away_team} {a_elo} (Differenz: {elo_diff:+d})
- Form letzte 5: {home_team} {h_form:.1f} Pkt/Sp | {away_team} {a_form:.1f} Pkt/Sp
- Trend letzte 3: {home_team} {h_short:.1f} | {away_team} {a_short:.1f}
- Heimstärke {home_team}: {h_home:.1f} Pkt/Sp | Auswärtsstärke {away_team}: {a_away:.1f} Pkt/Sp
- Saisonschnitt: {home_team} {h_season:.2f} | {away_team} {a_season:.2f} Pkt/Sp
- Direktvergleich: {h2h_h*100:.0f}% {home_team} | {h2h_d*100:.0f}% X | {h2h_a*100:.0f}% {away_team}
- Ruhetage: {home_team} {h_rest}d | {away_team} {a_rest}d
- Angriffsrating: {home_team} {h_atk:.2f} | {away_team} {a_atk:.2f}
- Defensivrating: {home_team} {h_def:.2f} | {away_team} {a_def:.2f}

Regeln:
- Nenne die 2-3 wichtigsten Datenpunkte die den Tipp treiben (konkrete Zahlen verwenden)
- Kein Intro, keine Überschrift, direkt mit der Analyse starten
- Natürlicher Ton, wie ein Experte der seinen Tipp erklärt
- Wenn ein Team klar stärker ist, das direkt benennen
- Wenn es ein enges Spiel ist (wenig Differenz in den Metriken), das erwähnen"""


async def generate_analysis(
    match_id: int,
    matchday: int,
    home_team: str,
    away_team: str,
    prediction: dict,
    features: dict,
    dc_alpha: dict,
    dc_beta: dict,
) -> str:
    """Generate AI analysis for one match. Returns cached result if available."""
    cache_key = (matchday, match_id)
    if cache_key in _analysis_cache:
        return _analysis_cache[cache_key]

    if not settings.ANTHROPIC_API_KEY:
        return ""

    try:
        prompt = _build_prompt(home_team, away_team, prediction, features, dc_alpha, dc_beta)

        loop = asyncio.get_event_loop()

        def _call() -> str:
            response = _client().messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=250,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()

        text = await loop.run_in_executor(None, _call)
        _analysis_cache[cache_key] = text
        return text

    except Exception as e:
        logger.warning(f"Analysis generation failed for {home_team} vs {away_team}: {e}")
        return ""


async def generate_all_analyses(
    matches_data: list[dict],
    matchday: int,
) -> dict[int, str]:
    """
    Generate analyses for all matches in parallel.
    matches_data: list of dicts with match_id, home_team, away_team, prediction, features, dc_alpha, dc_beta
    Returns: {match_id: analysis_text}
    """
    tasks = [
        generate_analysis(
            match_id=m["match_id"],
            matchday=matchday,
            home_team=m["home_team"],
            away_team=m["away_team"],
            prediction=m["prediction"],
            features=m["features"],
            dc_alpha=m["dc_alpha"],
            dc_beta=m["dc_beta"],
        )
        for m in matches_data
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return {
        m["match_id"]: (r if isinstance(r, str) else "")
        for m, r in zip(matches_data, results)
    }
