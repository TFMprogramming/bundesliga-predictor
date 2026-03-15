from __future__ import annotations
import logging
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import ModelInfoResponse, TeamRating
from app.config import settings
from app.data.database import get_db
from app.data.models import ModelArtifact
from app.models.dixon_coles import DixonColesModel
from app.models.ensemble import EnsemblePredictor, optimize_weights
from app.models.xgboost_model import XGBoostPredictor
from app.models.lgbm_model import LGBMPredictor
from app.features.builder import build_feature_matrix
from sqlalchemy import select

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/model", tags=["model"])


async def _load_matches_for_training(session: AsyncSession) -> list[dict]:
    from sqlalchemy import select
    from app.data.models import Match, Team

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
        if m.home_team_id in teams and m.away_team_id in teams
    ]


async def _run_training(session: AsyncSession):
    logger.info("Starting model training...")
    matches = await _load_matches_for_training(session)

    if len(matches) < 100:
        raise ValueError(f"Not enough training data: {len(matches)} matches")

    # 1. Fit Dixon-Coles
    dc = DixonColesModel(xi=settings.DC_XI)
    dc.fit(matches, reference_dt=datetime.utcnow())
    dc.save()

    # 2. Build feature matrix (pass dc_model for DC-derived features)
    feature_df = build_feature_matrix(matches, dc_model=dc)

    # 3. Fit XGBoost
    xgb = XGBoostPredictor()
    metrics = xgb.fit(feature_df)
    xgb.save()

    # 4. Fit LightGBM
    lgbm = LGBMPredictor()
    try:
        lgbm_metrics = lgbm.fit(feature_df)
        lgbm.save()
        logger.info(f"LGBM metrics: {lgbm_metrics}")
    except Exception as e:
        logger.warning(f"LightGBM training skipped: {e}")
        lgbm = None

    # 5. Optimise ensemble weights on last 20% of matches (time-ordered)
    try:
        val_df = feature_df.dropna(subset=xgb.feature_cols + ["outcome"])
        split = int(len(val_df) * 0.8)
        val_df = val_df.iloc[split:].reset_index(drop=True)
        val_matches = matches[int(len(matches) * 0.8):]  # same time slice

        if len(val_matches) >= 30:
            dc_probs_list, xgb_probs_list, lgbm_probs_list, outcomes = [], [], [], []
            for m, (_, row) in zip(val_matches, val_df.iterrows()):
                try:
                    feat = row[xgb.feature_cols].to_dict()
                    dc_probs_list.append(dc.predict_1x2(m["home_team"], m["away_team"]))
                    xgb_probs_list.append(xgb.predict_proba(feat))
                    if lgbm is not None:
                        lgbm_probs_list.append(lgbm.predict_proba(feat))
                    outcomes.append(int(row["outcome"]))
                except Exception:
                    continue
            if len(outcomes) >= 30:
                optimize_weights(
                    dc_probs_list, xgb_probs_list,
                    lgbm_probs_list if lgbm is not None and lgbm_probs_list else None,
                    outcomes,
                )
    except Exception as e:
        logger.warning(f"Weight optimisation skipped: {e}")

    # 4. Save artifact metadata
    artifact_result = await session.execute(
        select(ModelArtifact).where(ModelArtifact.name == "ensemble")
    )
    artifact = artifact_result.scalar_one_or_none()
    if artifact is None:
        artifact = ModelArtifact(name="ensemble")
        session.add(artifact)

    artifact.trained_at = datetime.utcnow()
    artifact.seasons_used = ",".join(str(s) for s in settings.SEASONS)
    artifact.training_samples = len(matches)
    artifact.accuracy = metrics.get("accuracy")
    artifact.log_loss = metrics.get("log_loss")
    artifact.brier_score = metrics.get("brier_score")
    await session.commit()

    logger.info(f"Training complete. {len(matches)} matches, accuracy={metrics.get('accuracy'):.3f}")


@router.post("/train", status_code=202)
async def trigger_training(
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db),
):
    """Trigger a full model retrain in the background."""
    matches = await _load_matches_for_training(session)
    if len(matches) < 100:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough data ({len(matches)} matches). Run data ingestion first.",
        )
    background_tasks.add_task(_run_training, session)
    return {"message": "Training started in background.", "matches": len(matches)}


@router.get("/info", response_model=ModelInfoResponse)
async def get_model_info(session: AsyncSession = Depends(get_db)):
    artifact_result = await session.execute(
        select(ModelArtifact).where(ModelArtifact.name == "ensemble")
    )
    artifact = artifact_result.scalar_one_or_none()

    try:
        dc = DixonColesModel.load()
        gamma = dc.gamma
        rho = dc.rho
        fitted_at = dc.fitted_at
    except FileNotFoundError:
        gamma, rho, fitted_at = 0.0, 0.0, None

    matches = await _load_matches_for_training(session)

    return ModelInfoResponse(
        trainedAt=artifact.trained_at if artifact else fitted_at,
        seasonsUsed=settings.SEASONS,
        trainingMatches=artifact.training_samples if artifact else len(matches),
        dcGamma=round(gamma, 4),
        dcRho=round(rho, 4),
        accuracy=artifact.accuracy if artifact else None,
        logLoss=artifact.log_loss if artifact else None,
        brierScore=artifact.brier_score if artifact else None,
    )


@router.get("/ratings", response_model=list[TeamRating])
async def get_team_ratings():
    """Attack/defense ratings from Dixon-Coles model."""
    try:
        dc = DixonColesModel.load()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not trained yet.")
    return dc.get_team_ratings()
