import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.data.database import init_db
from app.api.routes import predictions, standings, model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

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


@app.on_event("startup")
async def startup():
    await init_db()
    logging.getLogger(__name__).info("Database initialised.")


@app.get("/health")
async def health():
    return {"status": "ok"}
