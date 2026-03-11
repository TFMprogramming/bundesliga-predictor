from __future__ import annotations
from datetime import datetime

from pydantic import BaseModel


class TeamSummary(BaseModel):
    teamId: int
    teamName: str
    shortName: str
    teamIconUrl: str


class Probabilities(BaseModel):
    homeWin: float
    draw: float
    awayWin: float


class Score(BaseModel):
    home: int
    away: int


class ExpectedGoals(BaseModel):
    home: float
    away: float


class MatchHighlight(BaseModel):
    emoji: str
    text: str


class MatchPrediction(BaseModel):
    matchId: int
    matchDateTime: datetime | None
    homeTeam: TeamSummary
    awayTeam: TeamSummary
    probabilities: Probabilities
    predictedScore: Score
    expectedGoals: ExpectedGoals
    scoreMatrix: list[list[float]]
    confidenceLevel: str
    modelAgreement: bool
    dcProbabilities: Probabilities
    xgbProbabilities: Probabilities
    highlights: list[MatchHighlight] = []


class NextMatchdayResponse(BaseModel):
    matchday: int
    season: int
    generatedAt: datetime
    matches: list[MatchPrediction]


class StandingEntry(BaseModel):
    position: int
    teamId: int
    teamName: str
    teamIconUrl: str
    matches: int
    won: int
    draw: int
    lost: int
    goals: int
    opponentGoals: int
    goalDiff: int
    points: int


class StandingsResponse(BaseModel):
    season: int
    table: list[StandingEntry]


class TeamRating(BaseModel):
    team: str
    attack: float
    defense: float


class ModelInfoResponse(BaseModel):
    trainedAt: datetime | None
    seasonsUsed: list[int]
    trainingMatches: int
    dcGamma: float
    dcRho: float
    accuracy: float | None
    logLoss: float | None
    brierScore: float | None
