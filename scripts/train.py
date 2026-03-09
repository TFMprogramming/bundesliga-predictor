"""
One-shot training script:
  1. Ingests all historical seasons from OpenLigaDB
  2. Trains Dixon-Coles + XGBoost
  3. Saves artifacts

Usage:
    cd backend
    python -m scripts.train
"""
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.data.database import async_session_factory, init_db
from app.etl.ingest import ingest_all_seasons
from app.api.routes.model import _run_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


async def main():
    logger.info("=== Bundesliga Predictor — Training Pipeline ===")

    # 1. Init DB
    await init_db()

    async with async_session_factory() as session:
        # 2. Ingest all seasons
        logger.info("Step 1/2: Ingesting match data from OpenLigaDB...")
        await ingest_all_seasons(session)

        # 3. Train models
        logger.info("Step 2/2: Training models...")
        await _run_training(session)

    logger.info("=== Training complete! Start the API with: uvicorn app.main:app --reload ===")


if __name__ == "__main__":
    asyncio.run(main())
