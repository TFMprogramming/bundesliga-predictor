from __future__ import annotations
"""
Async client for the OpenLigaDB REST API.
Docs: https://api.openligadb.de/swagger/index.html
"""
import asyncio
import logging
from datetime import datetime

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

BASE = settings.OPENLIGADB_BASE_URL
LEAGUE = settings.LEAGUE


def _parse_dt(raw: str | None) -> datetime | None:
    if not raw:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.strptime(raw[:19], fmt[:len(fmt)])
        except ValueError:
            continue
    return None


class OpenLigaDBClient:
    def __init__(self):
        self._client = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        await self._client.aclose()

    async def _get(self, path: str) -> dict | list:
        url = f"{BASE}/{path.lstrip('/')}"
        logger.debug(f"GET {url}")
        resp = await self._client.get(url)
        resp.raise_for_status()
        await asyncio.sleep(settings.REQUEST_DELAY_SECONDS)
        return resp.json()

    # -------------------------------------------------------------------------
    # Season data
    # -------------------------------------------------------------------------

    async def get_season_matches(self, season: int) -> list[dict]:
        """All matches for one Bundesliga season."""
        raw = await self._get(f"/getmatchdata/{LEAGUE}/{season}")
        return [self._parse_match(m) for m in raw]

    async def get_matchday_matches(self, season: int, matchday: int) -> list[dict]:
        """9 matches for a specific matchday."""
        raw = await self._get(f"/getmatchdata/{LEAGUE}/{season}/{matchday}")
        return [self._parse_match(m) for m in raw]

    async def get_current_group(self) -> dict:
        """Returns info about the current matchday."""
        raw = await self._get(f"/getcurrentgroup/{LEAGUE}")
        return {
            "groupName": raw.get("groupName", ""),
            "groupOrderID": raw.get("groupOrderID", 0),
            "groupID": raw.get("groupID", 0),
        }

    async def get_available_teams(self, season: int) -> list[dict]:
        raw = await self._get(f"/getavailableteams/{LEAGUE}/{season}")
        return [
            {
                "teamId": t["teamId"],
                "teamName": t["teamName"],
                "shortName": t.get("shortName", ""),
                "teamIconUrl": t.get("teamIconUrl", ""),
            }
            for t in raw
        ]

    async def get_table(self, season: int) -> list[dict]:
        raw = await self._get(f"/getbltable/{LEAGUE}/{season}")
        return [
            {
                "teamId": r["teamInfoId"],
                "teamName": r["teamName"],
                "points": r["points"],
                "won": r["won"],
                "draw": r["draw"],
                "lost": r["lost"],
                "goals": r["goals"],
                "opponentGoals": r["opponentGoals"],
                "goalDiff": r["goalDiff"],
                "matches": r["matches"],
            }
            for r in raw
        ]

    async def get_next_matchday_matches(self) -> list[dict]:
        """Fetch next matchday fixtures (may include unfinished matches)."""
        group = await self.get_current_group()
        season = settings.CURRENT_SEASON
        matchday = group["groupOrderID"]
        matches = await self.get_matchday_matches(season, matchday)
        # If all finished, get next matchday
        if all(m["isFinished"] for m in matches):
            matches = await self.get_matchday_matches(season, matchday + 1)
        return matches

    # -------------------------------------------------------------------------
    # Parse helpers
    # -------------------------------------------------------------------------

    def _parse_match(self, raw: dict) -> dict:
        home = raw.get("team1", {})
        away = raw.get("team2", {})
        results = raw.get("matchResults", [])
        goals = raw.get("goals", [])

        home_goals, away_goals, is_finished = None, None, raw.get("matchIsFinished", False)

        # matchResults: list with pointsTeam1/pointsTeam2, resultTypeID 2 = final
        for r in results:
            if r.get("resultTypeID") == 2:
                home_goals = r.get("pointsTeam1")
                away_goals = r.get("pointsTeam2")
                break
        if home_goals is None and is_finished and results:
            # fallback: take last result
            last = results[-1]
            home_goals = last.get("pointsTeam1")
            away_goals = last.get("pointsTeam2")

        return {
            "matchId": raw["matchID"],
            "matchDatetime": _parse_dt(raw.get("matchDateTime")),
            "matchday": raw.get("group", {}).get("groupOrderID", 0),
            "homeTeamId": home.get("teamId"),
            "homeTeamName": home.get("teamName", ""),
            "homeTeamShort": home.get("shortName", ""),
            "homeTeamIcon": home.get("teamIconUrl", ""),
            "awayTeamId": away.get("teamId"),
            "awayTeamName": away.get("teamName", ""),
            "awayTeamShort": away.get("shortName", ""),
            "awayTeamIcon": away.get("teamIconUrl", ""),
            "homeGoals": home_goals,
            "awayGoals": away_goals,
            "isFinished": is_finished,
            "goals": [
                {
                    "scorerName": g.get("goalGetterName", ""),
                    "minute": g.get("matchMinute"),
                    "scoreHome": g.get("scoreTeam1"),
                    "scoreAway": g.get("scoreTeam2"),
                    "isPenalty": g.get("isPenalty", False),
                    "isOwnGoal": g.get("isOwnGoal", False),
                    "isOvertime": g.get("isOvertime", False),
                }
                for g in goals
            ],
        }
