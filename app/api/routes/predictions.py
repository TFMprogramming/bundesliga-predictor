from __future__ import annotations
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import MatchPrediction, NextMatchdayResponse, Probabilities, Score, ExpectedGoals, TeamSummary
from app.config import settings
from app.data.database import get_db
from app.data.models import Match, Team
from app.data.openligadb_client import OpenLigaDBClient
from app.features.builder import compute_prediction_features
from app.models.ensemble import load_ensemble
from app.services.analysis import generate_all_analyses

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predictions", tags=["predictions"])


async def _get_all_matches(session: AsyncSession) -> list[dict]:
    """Load all finished matches from DB for feature computation."""
    result = await session.execute(
        select(Match).where(Match.is_finished == True).order_by(Match.match_datetime)
    )
    matches = result.scalars().all()

    teams_result = await session.execute(select(Team))
    teams = {t.team_id: t for t in teams_result.scalars().all()}

    return [
        {
            "match_id": m.match_id,
            "home_team": teams[m.home_team_id].team_name if m.home_team_id in teams else str(m.home_team_id),
            "away_team": teams[m.away_team_id].team_name if m.away_team_id in teams else str(m.away_team_id),
            "home_goals": m.home_goals,
            "away_goals": m.away_goals,
            "match_datetime": m.match_datetime,
            "season": m.season,
            "matchday": m.matchday,
        }
        for m in matches
    ]


def _team_summary(team: Team) -> TeamSummary:
    return TeamSummary(
        teamId=team.team_id,
        teamName=team.team_name,
        shortName=team.short_name or team.team_name[:3].upper(),
        teamIconUrl=team.icon_url or "",
    )


@router.get("/next-matchday", response_model=NextMatchdayResponse)
async def get_next_matchday_predictions(session: AsyncSession = Depends(get_db)):
    """Fetch upcoming fixtures and return ML predictions for each match."""
    # Load ensemble model
    try:
        ensemble = load_ensemble()
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run POST /api/v1/model/train first.",
        )

    # Fetch upcoming fixtures from OpenLigaDB
    client = OpenLigaDBClient()
    try:
        raw_matches = await client.get_next_matchday_matches()
    except Exception as e:
        logger.error(f"Failed to fetch next matchday: {e}")
        raise HTTPException(status_code=502, detail=f"OpenLigaDB unavailable: {e}")
    finally:
        await client.close()

    if not raw_matches:
        raise HTTPException(status_code=404, detail="No upcoming matches found.")

    matchday = raw_matches[0].get("matchday", 0)

    # Load all historical matches for feature computation
    all_matches = await _get_all_matches(session)

    # Load teams from DB for icons
    teams_result = await session.execute(select(Team))
    teams_by_id = {t.team_id: t for t in teams_result.scalars().all()}
    teams_by_name = {t.team_name: t for t in teams_by_id.values()}

    # Build predictions + collect data for parallel analysis generation
    match_results = []
    for raw in raw_matches:
        home_name = raw["homeTeamName"]
        away_name = raw["awayTeamName"]
        match_dt = raw["matchDatetime"]

        features = compute_prediction_features(home_name, away_name, all_matches, match_dt)
        result = ensemble.predict(home_name, away_name, features)

        match_results.append({
            "raw": raw,
            "home_name": home_name,
            "away_name": away_name,
            "match_dt": match_dt,
            "features": features,
            "result": result,
        })

    # Generate AI analyses in parallel for all matches
    analyses_input = [
        {
            "match_id": m["raw"]["matchId"],
            "home_team": m["home_name"],
            "away_team": m["away_name"],
            "prediction": m["result"],
            "features": m["features"],
            "dc_alpha": ensemble.dc.alpha,
            "dc_beta": ensemble.dc.beta,
        }
        for m in match_results
    ]
    analyses = await generate_all_analyses(analyses_input, matchday, session)

    # Assemble final predictions
    predictions = []
    for m in match_results:
        raw = m["raw"]
        home_name = m["home_name"]
        away_name = m["away_name"]
        result = m["result"]

        home_team = teams_by_name.get(home_name)
        away_team = teams_by_name.get(away_name)

        home_summary = TeamSummary(
            teamId=raw["homeTeamId"] or 0,
            teamName=home_name,
            shortName=raw.get("homeTeamShort") or home_name[:3].upper(),
            teamIconUrl=(home_team.icon_url if home_team else None) or raw.get("homeTeamIcon", ""),
        )
        away_summary = TeamSummary(
            teamId=raw["awayTeamId"] or 0,
            teamName=away_name,
            shortName=raw.get("awayTeamShort") or away_name[:3].upper(),
            teamIconUrl=(away_team.icon_url if away_team else None) or raw.get("awayTeamIcon", ""),
        )

        p = result["probabilities"]
        dc_p = result["dcProbabilities"]
        xgb_p = result["xgbProbabilities"]

        predictions.append(
            MatchPrediction(
                matchId=raw["matchId"],
                matchDateTime=m["match_dt"],
                homeTeam=home_summary,
                awayTeam=away_summary,
                probabilities=Probabilities(**p),
                predictedScore=Score(**result["predictedScore"]),
                expectedGoals=ExpectedGoals(**result["expectedGoals"]),
                scoreMatrix=result["scoreMatrix"],
                confidenceLevel=result["confidenceLevel"],
                modelAgreement=result["modelAgreement"],
                dcProbabilities=Probabilities(**dc_p),
                xgbProbabilities=Probabilities(**xgb_p),
                analysisText=analyses.get(raw["matchId"]) or None,
            )
        )

    return NextMatchdayResponse(
        matchday=matchday,
        season=settings.CURRENT_SEASON,
        generatedAt=datetime.utcnow(),
        matches=predictions,
    )
