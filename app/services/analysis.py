from __future__ import annotations
"""
Rule-based match analysis generator — no API costs.
Generates natural German analysis text based on the actual prediction data.
"""
import logging
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.models import MatchAnalysis

logger = logging.getLogger(__name__)


def generate_analysis_text(
    home_team: str,
    away_team: str,
    prediction: dict,
    features: dict,
    dc_alpha: dict,
    dc_beta: dict,
) -> str:
    """Generate a data-driven analysis without any external API calls."""
    probs = prediction["probabilities"]
    score = prediction["predictedScore"]
    xg = prediction["expectedGoals"]
    confidence = prediction.get("confidenceLevel", "medium")

    elo_diff = features.get("elo_diff", 0)
    h_form = features.get("h_form_pts", 1.0)
    a_form = features.get("a_form_pts", 1.0)
    h_short = features.get("h_short_form_pts", 1.0)
    a_short = features.get("a_short_form_pts", 1.0)
    h_home = features.get("h_home_pts", 1.0)
    a_away = features.get("a_away_pts", 1.0)
    h_pos = int(features.get("h_position", 9))
    a_pos = int(features.get("a_position", 9))
    h2h_h = features.get("h2h_home_wins", 0.33)
    h2h_a = features.get("h2h_away_wins", 0.33)
    h2h_n = int(features.get("h2h_matches", 0))
    h_rest = int(features.get("h_days_rest", 7))
    a_rest = int(features.get("a_days_rest", 7))
    h_season = features.get("h_season_pts_pg", 1.0)
    a_season = features.get("a_season_pts_pg", 1.0)
    h_atk = dc_alpha.get(home_team, 1.0)
    a_atk = dc_alpha.get(away_team, 1.0)
    h_def = dc_beta.get(home_team, 1.0)
    a_def = dc_beta.get(away_team, 1.0)

    home_win = probs["homeWin"]
    away_win = probs["awayWin"]
    draw = probs["draw"]
    is_home_fav = home_win > away_win and home_win > draw
    is_away_fav = away_win > home_win and away_win > draw
    is_draw_fav = draw >= home_win and draw >= away_win
    fav = home_team if is_home_fav else (away_team if is_away_fav else None)
    fav_prob = max(home_win, away_win, draw)
    underdog = away_team if is_home_fav else (home_team if is_away_fav else None)

    sentences = []

    # ── Sentence 1: Main strength driver ────────────────────────────────────
    if abs(elo_diff) >= 120:
        stronger = home_team if elo_diff > 0 else away_team
        weaker = away_team if elo_diff > 0 else home_team
        diff = abs(int(elo_diff))
        sentences.append(
            f"{stronger} geht als klarer Favorit in dieses Spiel: "
            f"Der Elo-Vorsprung von {diff} Punkten gegenüber {weaker} ist einer der größten in diesem Spieltag."
        )
    elif abs(h_pos - a_pos) >= 8:
        higher = home_team if h_pos < a_pos else away_team
        lower = away_team if h_pos < a_pos else home_team
        higher_pos = h_pos if h_pos < a_pos else a_pos
        lower_pos = a_pos if h_pos < a_pos else h_pos
        sentences.append(
            f"Der Tabellenunterschied spricht eine deutliche Sprache: "
            f"{higher} (Platz {higher_pos}) trifft auf {lower} (Platz {lower_pos})."
        )
    elif abs(h_season - a_season) >= 0.5:
        better = home_team if h_season > a_season else away_team
        worse = away_team if h_season > a_season else home_team
        better_pts = h_season if h_season > a_season else a_season
        worse_pts = a_season if h_season > a_season else h_season
        sentences.append(
            f"{better} ist in dieser Saison das konstantere Team mit "
            f"{better_pts:.2f} Punkten pro Spiel gegenüber {worse_pts:.2f} bei {worse}."
        )
    elif is_draw_fav:
        sentences.append(
            f"Die Kennzahlen zeigen ein ausgeglichenes Duell: "
            f"Beide Teams liegen mit Elo-Rating {int(features.get('h_elo', 1500))} vs. "
            f"{int(features.get('a_elo', 1500))} und ähnlicher Saisonform nah beieinander."
        )
    else:
        sentences.append(
            f"Das Modell sieht {fav} leicht vorn ({fav_prob*100:.0f}%) – "
            f"der Unterschied in den Stärkekennzahlen ist real, aber nicht dominant."
        )

    # ── Sentence 2: Form / momentum / H2H ───────────────────────────────────
    form_diff = h_form - a_form
    short_diff = h_short - a_short

    # Strong recent momentum swing
    if abs(short_diff) >= 1.0:
        in_form = home_team if short_diff > 0 else away_team
        out_form = away_team if short_diff > 0 else home_team
        in_pts = h_short if short_diff > 0 else a_short
        sentences.append(
            f"Besonders der aktuelle Trend gibt den Ausschlag: "
            f"{in_form} holte in den letzten 3 Spielen {in_pts:.1f} Punkte pro Spiel, "
            f"{out_form} kommt hier deutlich schlechter weg."
        )
    # Good H2H record
    elif h2h_n >= 4 and (h2h_h >= 0.6 or h2h_a >= 0.6):
        dominant = home_team if h2h_h >= 0.6 else away_team
        rate = h2h_h if h2h_h >= 0.6 else h2h_a
        sentences.append(
            f"Auch der Direktvergleich stützt den Tipp: "
            f"{dominant} gewann {rate*100:.0f}% der letzten {h2h_n} Duelle zwischen diesen Teams."
        )
    # Venue-specific form
    elif abs(h_home - a_away) >= 0.7:
        if h_home > a_away:
            sentences.append(
                f"{home_team} ist zu Hause mit {h_home:.1f} Punkten pro Spiel stark, "
                f"während {away_team} auswärts mit {a_away:.1f} Punkten/Sp. "
                f"{'kaum Punkte mitnimmt' if a_away < 1.0 else 'schwächer aufritt'}."
            )
        else:
            sentences.append(
                f"{away_team} reist in guter Auswärtsform an ({a_away:.1f} Pkt/Sp), "
                f"während {home_team} zuhause mit {h_home:.1f} Punkten pro Spiel "
                f"{'enttäuscht' if h_home < 1.0 else 'solide, aber nicht beeindruckend ist'}."
            )
    # General form diff
    elif abs(form_diff) >= 0.6:
        better_form = home_team if form_diff > 0 else away_team
        pts = h_form if form_diff > 0 else a_form
        sentences.append(
            f"Die Form der letzten 5 Spiele spricht für {better_form}: "
            f"{pts:.1f} Punkte pro Spiel – klar besser als die Konkurrenz in dieser Phase."
        )
    else:
        sentences.append(
            f"Formtechnisch liegen beide Teams eng beieinander "
            f"({h_form:.1f} vs. {a_form:.1f} Pkt/Sp in den letzten 5 Spielen), "
            f"was das Duell zusätzlich schwer einschätzbar macht."
        )

    # ── Sentence 3: xG / rest / confidence context ──────────────────────────
    xg_diff = xg["home"] - xg["away"]
    rest_diff = h_rest - a_rest

    if abs(rest_diff) >= 3 and rest_diff < 0:
        sentences.append(
            f"Ein Warnsignal für {home_team}: nur {h_rest} Tage Pause, "
            f"{away_team} hatte {a_rest} Tage Erholung – Frische könnte heute ein Faktor sein."
        )
    elif abs(rest_diff) >= 3 and rest_diff > 0:
        sentences.append(
            f"{home_team} profitiert von {h_rest} Ruhetagen, "
            f"{away_team} war zuletzt häufiger gefordert ({a_rest} Tage Pause)."
        )
    elif abs(xg_diff) >= 0.7:
        more_xg = home_team if xg_diff > 0 else away_team
        xg_val = xg["home"] if xg_diff > 0 else xg["away"]
        sentences.append(
            f"Das Torchancen-Modell (xG) unterstreicht die Prognose: "
            f"{more_xg} kommt auf {xg_val:.2f} erwartete Tore, was das vorhergesagte Ergebnis {score['home']}:{score['away']} gut erklärt."
        )
    elif confidence == "high":
        sentences.append(
            f"Beide KI-Modelle sind sich einig und die Wahrscheinlichkeitsverteilung zeigt "
            f"eine überdurchschnittlich klare Tendenz – das Modell bewertet diese Prognose mit hoher Konfidenz."
        )
    elif confidence == "low":
        sentences.append(
            f"Trotz der Tendenz bleibt das Spiel schwer einschätzbar: "
            f"Die Wahrscheinlichkeiten von Heimsieg ({home_win*100:.0f}%), "
            f"Unentschieden ({draw*100:.0f}%) und Auswärtssieg ({away_win*100:.0f}%) liegen nah beieinander."
        )
    else:
        sentences.append(
            f"Das vorhergesagte Ergebnis {score['home']}:{score['away']} spiegelt die "
            f"Torerwartungswerte ({xg['home']:.2f} vs. {xg['away']:.2f} xG) wider."
        )

    return " ".join(sentences)


async def generate_all_analyses(
    matches_data: list[dict],
    matchday: int,
    session: AsyncSession,
) -> dict[int, str]:
    """
    Return analysis texts for all matches.
    Loads from DB cache where available; generates missing ones locally.
    """
    match_ids = [m["match_id"] for m in matches_data]

    # Load cached analyses from DB
    result = await session.execute(
        select(MatchAnalysis).where(MatchAnalysis.match_id.in_(match_ids))
    )
    cached = {row.match_id: row.analysis_text for row in result.scalars().all()}

    missing = [m for m in matches_data if m["match_id"] not in cached]
    if not missing:
        return cached

    logger.info(f"Generating analysis for {len(missing)} matches (matchday {matchday})")

    for m in missing:
        text = generate_analysis_text(
            m["home_team"], m["away_team"],
            m["prediction"], m["features"],
            m["dc_alpha"], m["dc_beta"],
        )
        if text:
            session.add(MatchAnalysis(
                match_id=m["match_id"],
                matchday=matchday,
                analysis_text=text,
            ))
            cached[m["match_id"]] = text

    await session.commit()
    return cached
