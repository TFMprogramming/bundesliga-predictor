from __future__ import annotations
"""
Match analysis generator with deterministic variation.
Uses match_id as seed → same match always gets same text,
but each match picks from a large template pool based on its data.
"""
import logging
import random
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.models import MatchAnalysis

logger = logging.getLogger(__name__)


def _rng(match_id: int, salt: int = 0) -> random.Random:
    """Deterministic RNG seeded by match_id — reproducible per match."""
    return random.Random(match_id * 31 + salt)


def _pick(rng: random.Random, options: list[str]) -> str:
    return rng.choice(options)


def generate_analysis_text(
    home_team: str,
    away_team: str,
    prediction: dict,
    features: dict,
    dc_alpha: dict,
    dc_beta: dict,
    match_id: int = 0,
) -> str:
    rng = _rng(match_id)

    probs = prediction["probabilities"]
    score = prediction["predictedScore"]
    xg = prediction["expectedGoals"]
    confidence = prediction.get("confidenceLevel", "medium")
    model_agreement = prediction.get("modelAgreement", False)

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
    underdog = away_team if is_home_fav else (home_team if is_away_fav else None)
    xg_home, xg_away = xg["home"], xg["away"]
    xg_diff = xg_home - xg_away

    # Score all potential narrative angles by their "interestingness"
    angles: list[tuple[float, str]] = []

    # ── Elo dominance ────────────────────────────────────────────────────────
    if abs(elo_diff) >= 180:
        stronger = home_team if elo_diff > 0 else away_team
        weaker = away_team if elo_diff > 0 else home_team
        diff = abs(int(elo_diff))
        s = _pick(rng, [
            f"{stronger} ist laut Elo-Modell um {diff} Punkte stärker als {weaker} – ein Unterschied, der sich selten nicht im Ergebnis niederschlägt.",
            f"Mit einem Elo-Vorsprung von {diff} Punkten ist {stronger} in einer anderen Leistungsklasse als {weaker} an diesem Spieltag.",
            f"Der KI-Stärkeindex stuft {stronger} ({int(features.get('h_elo' if elo_diff > 0 else 'a_elo', 1500))}) deutlich über {weaker} ({int(features.get('a_elo' if elo_diff > 0 else 'h_elo', 1500))}) ein.",
        ])
        angles.append((abs(elo_diff) / 50, s))
    elif abs(elo_diff) >= 80:
        stronger = home_team if elo_diff > 0 else away_team
        diff = abs(int(elo_diff))
        s = _pick(rng, [
            f"{stronger} hat einen spürbaren Elo-Vorteil von {diff} Punkten – nicht überwältigend, aber konsistent genug, um den Ausschlag zu geben.",
            f"Im Stärke-Rating liegt {stronger} mit {diff} Punkten Vorsprung vorn, was sich über eine Saison als verlässlicher Indikator erwiesen hat.",
        ])
        angles.append((abs(elo_diff) / 80, s))

    # ── Table position ───────────────────────────────────────────────────────
    if abs(h_pos - a_pos) >= 10:
        higher = home_team if h_pos < a_pos else away_team
        lower = away_team if h_pos < a_pos else home_team
        hp = h_pos if h_pos < a_pos else a_pos
        lp = a_pos if h_pos < a_pos else h_pos
        s = _pick(rng, [
            f"Platz {hp} trifft auf Platz {lp} – der Tabellenabstand von {lp - hp} Rängen ist in der Bundesliga eine erhebliche Lücke.",
            f"Zwischen {higher} (Platz {hp}) und {lower} (Platz {lp}) liegen {lp - hp} Tabellenplätze, was die unterschiedliche Saisonkonstanz unterstreicht.",
        ])
        angles.append((abs(h_pos - a_pos) / 5, s))

    # ── Recent momentum (last 3) ─────────────────────────────────────────────
    short_diff = h_short - a_short
    if abs(short_diff) >= 1.2:
        in_form = home_team if short_diff > 0 else away_team
        out_form = away_team if short_diff > 0 else home_team
        in_pts = h_short if short_diff > 0 else a_short
        out_pts = a_short if short_diff > 0 else h_short
        s = _pick(rng, [
            f"{in_form} zeigt gerade Topform: {in_pts:.1f} Punkte pro Spiel in den letzten drei Partien, während {out_form} auf nur {out_pts:.1f} kommt.",
            f"Der Trendunterschied ist markant – {in_form} holte zuletzt {in_pts:.1f} Pkt/Sp, {out_form} enttäuschte mit {out_pts:.1f}.",
            f"Aktuell hat {in_form} Rückenwind: In der Kurzform (letzte 3) liegt der Abstand zu {out_form} bei {abs(short_diff):.1f} Punkten pro Spiel.",
        ])
        angles.append((abs(short_diff) * 1.2, s))
    elif abs(short_diff) >= 0.7:
        in_form = home_team if short_diff > 0 else away_team
        in_pts = h_short if short_diff > 0 else a_short
        s = _pick(rng, [
            f"Kurzfristig hat {in_form} die bessere Form – {in_pts:.1f} Punkte pro Spiel in den letzten drei Partien.",
        ])
        angles.append((abs(short_diff), s))

    # ── 5-match form ─────────────────────────────────────────────────────────
    form_diff = h_form - a_form
    if abs(form_diff) >= 0.8:
        better = home_team if form_diff > 0 else away_team
        worse = away_team if form_diff > 0 else home_team
        b_pts = h_form if form_diff > 0 else a_form
        w_pts = a_form if form_diff > 0 else h_form
        s = _pick(rng, [
            f"Die Formkurve über die letzten fünf Spiele zeigt: {better} mit {b_pts:.1f} Punkten pro Spiel ist derzeit einfach beständiger als {worse} ({w_pts:.1f}).",
            f"{better} hat die deutlich bessere Fünf-Spiele-Form im Gepäck ({b_pts:.1f} vs. {w_pts:.1f} Pkt/Sp).",
        ])
        angles.append((abs(form_diff) * 0.9, s))

    # ── Venue-specific form ──────────────────────────────────────────────────
    venue_diff = h_home - a_away
    if abs(venue_diff) >= 0.8:
        if venue_diff > 0:
            s = _pick(rng, [
                f"{home_team} ist auf eigenem Platz schwer zu schlagen ({h_home:.1f} Pkt/Sp zuhause), {away_team} tut sich auswärts dagegen schwer ({a_away:.1f}).",
                f"Die Heimbilanz von {home_team} ({h_home:.1f} Pkt/Sp) trifft auf die schwache Auswärtsbilanz von {away_team} ({a_away:.1f}) – ein klares strukturelles Ungleichgewicht.",
            ])
        else:
            s = _pick(rng, [
                f"{away_team} reist als starkes Auswärtsteam an ({a_away:.1f} Pkt/Sp), während {home_team} zuhause zuletzt enttäuschte ({h_home:.1f}).",
                f"Interessante Konstellation: {away_team} ist auswärts ({a_away:.1f} Pkt/Sp) tatsächlich stärker als {home_team} auf eigenem Platz ({h_home:.1f}).",
            ])
        angles.append((abs(venue_diff) * 0.85, s))

    # ── Head-to-head ─────────────────────────────────────────────────────────
    if h2h_n >= 5 and max(h2h_h, h2h_a) >= 0.55:
        dominant = home_team if h2h_h > h2h_a else away_team
        rate = max(h2h_h, h2h_a)
        s = _pick(rng, [
            f"Die Geschichte dieses Duells spricht für {dominant}: {rate*100:.0f}% der letzten {h2h_n} Begegnungen gingen an sie.",
            f"Im direkten Vergleich dominiert {dominant} mit {rate*100:.0f}% Siegen aus {h2h_n} Duellen.",
            f"Historisch hat {dominant} in diesem Matchup eindeutig die Nase vorn – {rate*100:.0f}% Gewinnquote aus {h2h_n} Spielen.",
        ])
        angles.append((rate - 0.33, s))

    # ── xG / attacking model ─────────────────────────────────────────────────
    if abs(xg_diff) >= 0.8:
        more = home_team if xg_diff > 0 else away_team
        less = away_team if xg_diff > 0 else home_team
        more_xg = xg_home if xg_diff > 0 else xg_away
        s = _pick(rng, [
            f"Das Torchancen-Modell sieht {more} klar vorn: {more_xg:.2f} erwartete Tore gegen {less} – das prognostizierte Ergebnis {score['home']}:{score['away']} passt dazu.",
            f"Mit {xg_home:.2f} xG für {home_team} und {xg_away:.2f} für {away_team} spiegelt das erwartete Ergebnis {score['home']}:{score['away']} die Chancenstruktur klar wider.",
            f"Angriffsstärke {home_team}: {h_atk:.2f}, Defensivrating {away_team}: {a_def:.2f} – das ergibt {xg_home:.2f} erwartete Tore für die Heimseite.",
        ])
        angles.append((abs(xg_diff) * 0.7, s))

    # ── Rest / fatigue ───────────────────────────────────────────────────────
    rest_diff = h_rest - a_rest
    if abs(rest_diff) >= 3:
        fresher = home_team if rest_diff > 0 else away_team
        tired = away_team if rest_diff > 0 else home_team
        fresh_days = max(h_rest, a_rest)
        tired_days = min(h_rest, a_rest)
        s = _pick(rng, [
            f"Konditionell hat {fresher} einen Vorteil: {fresh_days} Tage Pause gegenüber nur {tired_days} bei {tired}.",
            f"{tired} kommt unter Umständen mit müden Beinen – nur {tired_days} Ruhetage, während {fresher} {fresh_days} Tage regenerieren konnte.",
        ])
        angles.append((abs(rest_diff) / 3, s))

    # ── Season consistency ───────────────────────────────────────────────────
    if abs(h_season - a_season) >= 0.6:
        better = home_team if h_season > a_season else away_team
        worse = away_team if h_season > a_season else home_team
        b_pts = max(h_season, a_season)
        w_pts = min(h_season, a_season)
        s = _pick(rng, [
            f"Saisonal ist {better} mit {b_pts:.2f} Punkten pro Spiel das konstantere Team gegenüber {worse} ({w_pts:.2f}).",
        ])
        angles.append((abs(h_season - a_season) * 0.7, s))

    # ── Close match / draw scenarios ─────────────────────────────────────────
    if is_draw_fav or (not is_home_fav and not is_away_fav):
        s = _pick(rng, [
            f"Beide Teams sind so nah beieinander, dass das Modell ein Unentschieden als wahrscheinlichstes Einzelergebnis sieht – kleine Zufälligkeiten entscheiden hier.",
            f"Die Kennzahlen ergeben ein klassisches Fifty-fifty-Duell: Weder Elo, Form noch Direktvergleich liefern ein klares Signal.",
            f"Dieses Spiel ist für das Modell eine echte Gratwanderung – die Wahrscheinlichkeiten für alle drei Ausgänge liegen relativ eng beieinander.",
        ])
        angles.append((0.5, s))

    # ── Model agreement bonus sentence ──────────────────────────────────────
    agreement_sentence = ""
    if model_agreement and confidence == "high":
        agreement_sentence = _pick(rng, [
            f"Bemerkenswert: Beide KI-Modelle (Dixon-Coles und XGBoost) zeigen in die gleiche Richtung – das erhöht die Verlässlichkeit dieser Prognose.",
            f"Alle Modelle sind sich einig, was in dieser Konstellation nicht selbstverständlich ist.",
        ])
    elif not model_agreement:
        agreement_sentence = _pick(rng, [
            f"Die beiden Teilmodelle kommen hier zu leicht unterschiedlichen Einschätzungen – das Ensemble-Ergebnis ist ein gewichteter Kompromiss.",
        ])

    # ── Select the best angles ───────────────────────────────────────────────
    angles.sort(key=lambda x: x[0], reverse=True)

    # Vary number of core sentences (2 or 3) based on match_id
    n_main = rng.choice([2, 2, 3])
    selected = [text for _, text in angles[:n_main]]

    if not selected:
        # Fallback for very average matches
        selected = [_pick(rng, [
            f"Das Modell sieht {fav or home_team} leicht vorn – die Unterschiede in den Kennzahlen sind vorhanden, aber nicht dramatisch.",
            f"Ein schwer einschätzbares Spiel: Die meisten Kennzahlen zeigen keine klare Dominanz einer Seite.",
        ])]

    if agreement_sentence:
        selected.append(agreement_sentence)

    return " ".join(selected)


async def generate_all_analyses(
    matches_data: list[dict],
    matchday: int,
    session: AsyncSession,
) -> dict[int, str]:
    """Return analyses for all matches. DB-cached, generated locally."""
    match_ids = [m["match_id"] for m in matches_data]

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
            match_id=m["match_id"],
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
