from __future__ import annotations
"""
Fetch all historical match data from OpenLigaDB and store in the local DB.
"""
import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.data.models import Goal, Match, Team
from app.data.openligadb_client import OpenLigaDBClient

logger = logging.getLogger(__name__)


async def upsert_team(session: AsyncSession, team_id: int, name: str, short: str, icon: str) -> Team:
    result = await session.execute(select(Team).where(Team.team_id == team_id))
    team = result.scalar_one_or_none()
    if team is None:
        team = Team(team_id=team_id, team_name=name, short_name=short, icon_url=icon)
        session.add(team)
    else:
        team.team_name = name
        team.short_name = short
        team.icon_url = icon or team.icon_url
    return team


async def upsert_match(session: AsyncSession, raw: dict, season: int) -> Match:
    result = await session.execute(select(Match).where(Match.match_id == raw["matchId"]))
    match = result.scalar_one_or_none()

    if match is None:
        match = Match(
            match_id=raw["matchId"],
            season=season,
            matchday=raw["matchday"],
            match_datetime=raw["matchDatetime"],
            home_team_id=raw["homeTeamId"],
            away_team_id=raw["awayTeamId"],
            home_goals=raw["homeGoals"],
            away_goals=raw["awayGoals"],
            is_finished=raw["isFinished"],
        )
        session.add(match)
    else:
        match.home_goals = raw["homeGoals"]
        match.away_goals = raw["awayGoals"]
        match.is_finished = raw["isFinished"]
        match.match_datetime = raw["matchDatetime"] or match.match_datetime

    return match


async def upsert_goals(session: AsyncSession, match_id: int, goals: list[dict]):
    # Delete existing goals for this match, then re-insert
    from sqlalchemy import delete
    await session.execute(delete(Goal).where(Goal.match_id == match_id))
    for g in goals:
        session.add(
            Goal(
                match_id=match_id,
                scorer_name=g["scorerName"],
                minute=g["minute"],
                score_home=g["scoreHome"],
                score_away=g["scoreAway"],
                is_penalty=g["isPenalty"],
                is_own_goal=g["isOwnGoal"],
                is_overtime=g["isOvertime"],
            )
        )


async def ingest_season(session: AsyncSession, client: OpenLigaDBClient, season: int):
    logger.info(f"Ingesting season {season}...")
    matches = await client.get_season_matches(season)
    teams_seen: set[int] = set()

    for raw in matches:
        # Upsert teams
        for tid, tname, tshort, ticon in [
            (raw["homeTeamId"], raw["homeTeamName"], raw["homeTeamShort"], raw["homeTeamIcon"]),
            (raw["awayTeamId"], raw["awayTeamName"], raw["awayTeamShort"], raw["awayTeamIcon"]),
        ]:
            if tid and tid not in teams_seen:
                await upsert_team(session, tid, tname, tshort, ticon)
                teams_seen.add(tid)

        # Upsert match
        if raw["homeTeamId"] and raw["awayTeamId"]:
            await upsert_match(session, raw, season)
            if raw["isFinished"] and raw["goals"]:
                await upsert_goals(session, raw["matchId"], raw["goals"])

    await session.commit()
    finished = sum(1 for m in matches if m["isFinished"])
    logger.info(f"Season {season}: {len(matches)} matches ({finished} finished)")


async def ingest_all_seasons(session: AsyncSession):
    client = OpenLigaDBClient()
    try:
        for season in settings.SEASONS:
            await ingest_season(session, client, season)
    finally:
        await client.close()
