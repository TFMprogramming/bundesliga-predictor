import asyncio
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select

from app.config import settings
from app.data.database import async_session_factory, init_db
from app.api.routes import predictions, standings, model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Bundesliga Predictor API",
    description="ML-powered Bundesliga match predictions using Dixon-Coles + XGBoost ensemble",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predictions.router, prefix=settings.API_PREFIX)
app.include_router(standings.router, prefix=settings.API_PREFIX)
app.include_router(model.router, prefix=settings.API_PREFIX)


async def _auto_ingest_and_train():
    """If DB is empty, fetch all seasons and train models automatically."""
    from app.data.models import Match
    from app.etl.ingest import ingest_all_seasons
    from app.api.routes.model import _run_training

    async with async_session_factory() as session:
        result = await session.execute(select(func.count()).select_from(Match))
        count = result.scalar()

    if count == 0:
        logger.info("Database is empty — starting automatic data ingestion...")
        async with async_session_factory() as session:
            await ingest_all_seasons(session)
        logger.info("Ingestion complete — training models...")
        async with async_session_factory() as session:
            await _run_training(session)
        logger.info("Auto-setup complete.")
    else:
        logger.info(f"Database has {count} matches — skipping ingestion.")


@app.on_event("startup")
async def startup():
    await init_db()
    logger.info("Database initialised.")
    asyncio.create_task(_auto_ingest_and_train())


@app.get("/health")
async def health():
    return {"status": "ok"}
