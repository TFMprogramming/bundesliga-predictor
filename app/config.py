from __future__ import annotations
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    OPENLIGADB_BASE_URL: str = "https://api.openligadb.de"
    LEAGUE: str = "bl1"
    SEASONS: list[int] = list(range(2017, 2026))  # 2017 through 2025
    CURRENT_SEASON: int = 2025

    DATABASE_URL: str = "sqlite+aiosqlite:///./bundesliga.db"

    # Dixon-Coles time decay (half-life ~230 days)
    DC_XI: float = 0.003
    # Minimum matches per team before trusting parameters
    MIN_MATCHES: int = 5

    CACHE_TTL_SECONDS: int = 3600  # 1h during matchdays
    REQUEST_DELAY_SECONDS: float = 0.1

    # API
    API_PREFIX: str = "/api/v1"

    model_config = {"env_file": ".env"}


settings = Settings()
