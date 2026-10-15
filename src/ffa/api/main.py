"""FastAPI app: projections, start/sit and lineups for any stored league's scoring.

The MCP server wraps these same endpoints, so the web app and AI agents get identical
answers. Players are addressed by key (nflverse gsis_id, or DST_<TEAM>); resolve names
with /players/search first.
"""

from __future__ import annotations

from collections.abc import Iterator

import duckdb
from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ffa import __version__
from ffa.config import settings
from ffa.db.models import LeagueRow, RosterEntryRow, TeamRow
from ffa.scoring.rules import ScoringRules, default_rules, from_sleeper
from ffa.service import startsit
from ffa.service.projections import upcoming_week
from ffa.warehouse import duck

app = FastAPI(
    title="FF Assistant API",
    version=__version__,
    description="Simulation-based fantasy football projections and start/sit decisions.",
)


# ---------------------------------------------------------------- dependencies


def get_con() -> Iterator[duckdb.DuckDBPyConnection]:
    con = duck.connect(settings.warehouse_dir)
    try:
        yield con
    finally:
        con.close()


def get_db() -> Iterator[Session | None]:
    try:
        from ffa.db.session import session

        db = session()
    except Exception:  # no app database configured: league endpoints are unavailable
        yield None
        return
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------- schemas


class WeekInfo(BaseModel):
    season: int
    week: int | None
    projections_available: bool


class SearchResult(BaseModel):
    key: str
    name: str | None
    team: str | None
    position: str
    match: float = Field(description="1.0 = exact name match")
    projected: bool = Field(description="False when the player is not playing this week (bye)")


class Player(BaseModel):
    key: str
    name: str
    team: str
    position: str
    opponent: str | None
    mean: float
    median: float
    p10: float
    p90: float
    play_probability: float
    injury_status: str | None
    expected_stats: dict[str, float]
    points_by_rule: dict[str, float] = Field(description="Expected points from each scoring rule")
    drivers: list[str] = Field(description="Plain-language reasons, each with its number")
    p_best: float | None = Field(None, description="Share of simulations this player scores most")


class Pairwise(BaseModel):
    a: str
    b: str
    p_a_beats_b: float


class StartSitRequest(BaseModel):
    players: list[str] = Field(min_length=2, max_length=6, description="Player keys")
    league: str | None = Field(
        None, description="League id (e.g. sleeper:123); its scoring is used"
    )
    week: int | None = None


class StartSitResponse(BaseModel):
    season: int
    week: int
    scoring: str
    recommendation: str
    close_call: bool
    notes: list[str]
    players: list[Player]
    pairwise: list[Pairwise]


class LineupSlot(BaseModel):
    slot: str
    key: str | None
    name: str | None
    mean: float
    median: float | None = None


class MyTeam(BaseModel):
    league: str
    team: str
    season: int
    week: int
    scoring: str
    lineup: list[LineupSlot]
    bench: list[Player]
    not_playing: list[str] = Field(
        description="Rostered players with no projection (bye, unmatched)"
    )


# ---------------------------------------------------------------- helpers


def _week(con, week: int | None) -> tuple[int, int]:
    season = settings.current_season
    w = week or upcoming_week(con, season)
    if w is None:
        raise HTTPException(404, f"No upcoming week in {season}")
    return season, w


def _league(db: Session | None, league: str | None) -> LeagueRow | None:
    uid = league or (
        f"sleeper:{settings.sleeper_league_id}" if settings.sleeper_league_id else None
    )
    if uid is None or db is None:
        return None
    row = db.get(LeagueRow, uid)
    if row is None and league is not None:
        raise HTTPException(404, f"Unknown league {league}; sync it first")
    return row


def _rules(row: LeagueRow | None) -> tuple[ScoringRules, str]:
    if row is None:
        return default_rules(), "default half-PPR"
    if row.platform == "sleeper":
        return from_sleeper(row.scoring), f"{row.name} ({row.platform})"
    raise HTTPException(501, f"Scoring for {row.platform} leagues is not supported yet")


# ---------------------------------------------------------------- endpoints


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/week", response_model=WeekInfo)
def week_info(con=Depends(get_con)) -> WeekInfo:
    season = settings.current_season
    w = upcoming_week(con, season)
    available = False
    if w is not None:
        try:
            available = bool(con.sql(
                f"select count(*) from projection_sims where season = {season} and week = {w}"
            ).fetchone()[0])  # fmt: skip
        except duckdb.Error:
            available = False
    return WeekInfo(season=season, week=w, projections_available=available)


@app.get("/players/search", response_model=list[SearchResult])
def players_search(
    q: str = Query(min_length=2), position: str | None = None, limit: int = 5,
    week: int | None = None, con=Depends(get_con),
) -> list[SearchResult]:  # fmt: skip
    season, w = _week(con, week)
    return [SearchResult(**r) for r in startsit.search(con, season, w, q, position, limit)]


def _views(con, keys: list[str], rules: ScoringRules, season: int, w: int):
    data = startsit.load_week(con, season, w, keys)
    missing = sorted(set(keys) - set(data.sims["key"].unique().to_list()))
    if missing:
        raise HTTPException(
            422,
            f"No projection in week {w} for {', '.join(missing)} "
            "(bye, not on an active roster, or unknown key). Use /players/search.",
        )
    return data


@app.get("/projections", response_model=list[Player])
def projections(
    keys: str = Query(description="Comma-separated player keys"), league: str | None = None,
    week: int | None = None, con=Depends(get_con), db=Depends(get_db),
) -> list[Player]:  # fmt: skip
    season, w = _week(con, week)
    rules, _ = _rules(_league(db, league))
    ks = [k.strip() for k in keys.split(",") if k.strip()]
    views = startsit.player_views(_views(con, ks, rules, season, w), rules)
    return [Player(**startsit.as_dict(views[k])) for k in ks]


@app.post("/start-sit", response_model=StartSitResponse)
def start_sit(req: StartSitRequest, con=Depends(get_con), db=Depends(get_db)) -> StartSitResponse:
    season, w = _week(con, req.week)
    rules, label = _rules(_league(db, req.league))
    if len(set(req.players)) != len(req.players):
        raise HTTPException(422, "Duplicate players")
    c = startsit.compare(_views(con, req.players, rules, season, w), rules)
    d = startsit.as_dict(c)
    return StartSitResponse(scoring=label, **d)


@app.get("/leagues")
def leagues(db=Depends(get_db)) -> list[dict]:
    if db is None:
        return []
    return [
        {
            "league": r.id,
            "name": r.name,
            "platform": r.platform,
            "season": r.season,
            "lineup_slots": r.lineup_slots,
        }  # fmt: skip
        for r in db.scalars(select(LeagueRow))
    ]


@app.get("/leagues/{league}/my-team", response_model=MyTeam)
def my_team(league: str, week: int | None = None, con=Depends(get_con),
            db=Depends(get_db)) -> MyTeam:  # fmt: skip
    if db is None:
        raise HTTPException(503, "App database not configured")
    row = _league(db, league)
    if row is None:
        raise HTTPException(404, f"Unknown league {league}")
    team = db.scalar(select(TeamRow).where(TeamRow.league_id == row.id, TeamRow.is_me))
    if team is None:
        raise HTTPException(404, "Your team isn't identified in this league; sync with --me")
    season, w = _week(con, week)
    rules, label = _rules(row)
    entries = db.scalars(select(RosterEntryRow).where(RosterEntryRow.team_id == team.id)).all()
    keys = [e.player_key for e in entries if e.player_key and e.slot_type != "ir"]
    data = startsit.load_week(con, season, w, keys)
    views = startsit.player_views(data, rules)
    lineup = startsit.best_lineup(views, row.lineup_slots)
    starters = {s["key"] for s in lineup}
    bench = sorted((v for k, v in views.items() if k not in starters), key=lambda v: -v.mean)
    names = {e.player_key: e.name or e.platform_player_id for e in entries}
    return MyTeam(
        league=row.id, team=team.name, season=season, week=w, scoring=label,
        lineup=[LineupSlot(**s) for s in lineup],
        bench=[Player(**startsit.as_dict(v)) for v in bench],
        not_playing=sorted(names.get(k) or k for k in set(keys) - set(views)),
    )  # fmt: skip
