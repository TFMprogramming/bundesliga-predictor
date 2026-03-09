from __future__ import annotations
import logging

from fastapi import APIRouter, HTTPException

from app.api.schemas import StandingEntry, StandingsResponse
from app.config import settings
from app.data.openligadb_client import OpenLigaDBClient

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/standings", tags=["standings"])


@router.get("/current", response_model=StandingsResponse)
async def get_current_standings():
    client = OpenLigaDBClient()
    try:
        table = await client.get_table(settings.CURRENT_SEASON)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenLigaDB unavailable: {e}")
    finally:
        await client.close()

    entries = [
        StandingEntry(
            position=i + 1,
            teamId=row["teamId"],
            teamName=row["teamName"],
            teamIconUrl="",
            matches=row["matches"],
            won=row["won"],
            draw=row["draw"],
            lost=row["lost"],
            goals=row["goals"],
            opponentGoals=row["opponentGoals"],
            goalDiff=row["goalDiff"],
            points=row["points"],
        )
        for i, row in enumerate(table)
    ]

    return StandingsResponse(season=settings.CURRENT_SEASON, table=entries)
