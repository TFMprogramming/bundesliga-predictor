from __future__ import annotations
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.data.database import Base


class Team(Base):
    __tablename__ = "teams"

    team_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    team_name: Mapped[str] = mapped_column(String, nullable=False)
    short_name: Mapped[str] = mapped_column(String, nullable=True)
    icon_url: Mapped[str] = mapped_column(String, nullable=True)

    home_matches: Mapped[list["Match"]] = relationship(
        "Match", foreign_keys="Match.home_team_id", back_populates="home_team"
    )
    away_matches: Mapped[list["Match"]] = relationship(
        "Match", foreign_keys="Match.away_team_id", back_populates="away_team"
    )


class Match(Base):
    __tablename__ = "matches"

    match_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    matchday: Mapped[int] = mapped_column(Integer, nullable=False)
    match_datetime: Mapped[datetime] = mapped_column(DateTime, nullable=True)

    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.team_id"), nullable=False)
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.team_id"), nullable=False)

    home_goals: Mapped[int] = mapped_column(Integer, nullable=True)
    away_goals: Mapped[int] = mapped_column(Integer, nullable=True)
    is_finished: Mapped[bool] = mapped_column(Boolean, default=False)

    home_team: Mapped["Team"] = relationship("Team", foreign_keys=[home_team_id], back_populates="home_matches")
    away_team: Mapped["Team"] = relationship("Team", foreign_keys=[away_team_id], back_populates="away_matches")

    __table_args__ = (
        Index("ix_matches_season_matchday", "season", "matchday"),
        Index("ix_matches_datetime", "match_datetime"),
    )


class Goal(Base):
    __tablename__ = "goals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.match_id"), nullable=False)
    scorer_name: Mapped[str] = mapped_column(String, nullable=True)
    minute: Mapped[int] = mapped_column(Integer, nullable=True)
    score_home: Mapped[int] = mapped_column(Integer, nullable=True)
    score_away: Mapped[int] = mapped_column(Integer, nullable=True)
    is_penalty: Mapped[bool] = mapped_column(Boolean, default=False)
    is_own_goal: Mapped[bool] = mapped_column(Boolean, default=False)
    is_overtime: Mapped[bool] = mapped_column(Boolean, default=False)


class MatchAnalysis(Base):
    """Cached AI-generated analysis texts per match."""
    __tablename__ = "match_analyses"

    match_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    matchday: Mapped[int] = mapped_column(Integer, nullable=False)
    analysis_text: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ModelArtifact(Base):
    __tablename__ = "model_artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    trained_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    seasons_used: Mapped[str] = mapped_column(String, nullable=True)
    brier_score: Mapped[float] = mapped_column(Float, nullable=True)
    log_loss: Mapped[float] = mapped_column(Float, nullable=True)
    accuracy: Mapped[float] = mapped_column(Float, nullable=True)
    training_samples: Mapped[int] = mapped_column(Integer, nullable=True)
