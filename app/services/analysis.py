from __future__ import annotations
"""
AI-generated match analysis using Claude.
Analyses are cached in SQLite — generated once per match, never regenerated.
"""
import asyncio
import logging
from functools import lru_cache

import anthropic
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.data.models import MatchAnalysis

logger = logging.getLogger(__name__)


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

    winner_label = (
        f"{home_team}-Sieg" if probs["homeWin"] > probs["awayWin"] and probs["homeWin"] > probs["draw"]
        else f"{away_team}-Sieg" if probs["awayWin"] > probs["homeWin"] and probs["awayWin"] > probs["draw"]
        else "Unentschieden"
    )

    h_atk = dc_alpha.get(home_team, 1.0)
    a_atk = dc_alpha.get(away_team, 1.0)
    h_def = dc_beta.get(home_team, 1.0)
    a_def = dc_beta.get(away_team, 1.0)

    return f"""Du bist ein Fußball-Analyst. Erkläre in 3 prägnanten Sätzen auf Deutsch, warum das KI-Modell für dieses Bundesliga-Spiel so tippt.

Spiel: {home_team} vs. {away_team} (Saison 2025/26)

Modell-Prognose:
- Wahrscheinlichstes Ergebnis: {score['home']}:{score['away']} ({winner_label})
- Wahrscheinlichkeiten: Heimsieg {probs['homeWin']*100:.0f}% | X {probs['draw']*100:.0f}% | Auswärtssieg {probs['awayWin']*100:.0f}%
- Expected Goals: {home_team} {xg['home']:.2f} | {away_team} {xg['away']:.2f}

Schlüsselkennzahlen:
- Tabelle: {home_team} Platz {int(features.get('h_position', 9))} | {away_team} Platz {int(features.get('a_position', 9))}
- Elo-Stärke: {home_team} {int(features.get('h_elo', 1500))} | {away_team} {int(features.get('a_elo', 1500))} (Diff: {int(features.get('elo_diff', 0)):+d})
- Form letzte 5: {home_team} {features.get('h_form_pts', 1.0):.1f} Pkt/Sp | {away_team} {features.get('a_form_pts', 1.0):.1f} Pkt/Sp
- Trend letzte 3: {home_team} {features.get('h_short_form_pts', 1.0):.1f} | {away_team} {features.get('a_short_form_pts', 1.0):.1f}
- Heimstärke: {home_team} {features.get('h_home_pts', 1.0):.1f} Pkt/Sp | Auswärtsstärke: {away_team} {features.get('a_away_pts', 1.0):.1f} Pkt/Sp
- Saisonschnitt: {home_team} {features.get('h_season_pts_pg', 1.0):.2f} | {away_team} {features.get('a_season_pts_pg', 1.0):.2f} Pkt/Sp
- Direktvergleich: {features.get('h2h_home_wins', 0.33)*100:.0f}% {home_team} | {features.get('h2h_draws', 0.33)*100:.0f}% X | {features.get('h2h_away_wins', 0.33)*100:.0f}% {away_team}
- Ruhetage: {home_team} {int(features.get('h_days_rest', 7))}d | {away_team} {int(features.get('a_days_rest', 7))}d
- Angriffsrating: {home_team} {h_atk:.2f} | {away_team} {a_atk:.2f}
- Defensivrating: {home_team} {h_def:.2f} | {away_team} {a_def:.2f}

Regeln:
- Nenne die 2-3 wichtigsten Datenpunkte die den Tipp treiben (konkrete Zahlen verwenden)
- Kein Intro, keine Überschrift, direkt mit der Analyse starten
- Natürlicher Ton wie ein Experte der seinen Tipp erklärt
- Wenn ein Team klar stärker ist, das direkt benennen; bei engem Spiel die Ausgeglichenheit erwähnen"""


def _generate_sync(prompt: str) -> str:
    response = _client().messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=250,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


async def generate_all_analyses(
    matches_data: list[dict],
    matchday: int,
    session: AsyncSession,
) -> dict[int, str]:
    """
    Return analysis texts for all matches.
    Loads from DB cache where available; generates missing ones in parallel.
    """
    if not settings.ANTHROPIC_API_KEY:
        return {}

    match_ids = [m["match_id"] for m in matches_data]

    # Load existing cached analyses from DB
    result = await session.execute(
        select(MatchAnalysis).where(MatchAnalysis.match_id.in_(match_ids))
    )
    cached = {row.match_id: row.analysis_text for row in result.scalars().all()}

    # Find matches that need generation
    missing = [m for m in matches_data if m["match_id"] not in cached]

    if not missing:
        return cached

    logger.info(f"Generating analysis for {len(missing)} matches (matchday {matchday})")

    # Generate missing analyses in parallel
    loop = asyncio.get_event_loop()

    async def _generate_one(m: dict) -> tuple[int, str]:
        prompt = _build_prompt(
            m["home_team"], m["away_team"],
            m["prediction"], m["features"],
            m["dc_alpha"], m["dc_beta"],
        )
        try:
            text = await loop.run_in_executor(None, _generate_sync, prompt)
        except Exception as e:
            logger.warning(f"Analysis failed for match {m['match_id']}: {e}")
            text = ""
        return m["match_id"], text

    new_results = await asyncio.gather(*[_generate_one(m) for m in missing])

    # Persist to DB
    for match_id, text in new_results:
        if text:
            session.add(MatchAnalysis(
                match_id=match_id,
                matchday=matchday,
                analysis_text=text,
            ))
    await session.commit()

    cached.update({mid: txt for mid, txt in new_results if txt})
    return cached
