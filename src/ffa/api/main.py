"""FastAPI app: projections, start/sit and lineups for any stored league's scoring.

The MCP server wraps these same endpoints, so the web app and AI agents get identical
answers. Players are addressed by key (nflverse gsis_id, or DST_<TEAM>); resolve names
with /players/search first.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime

import duckdb
from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ffa import __version__
from ffa.config import settings
from ffa.db.models import LeagueRow, RosterEntryRow, TeamRow
from ffa.scoring.rules import ScoringRules, default_rules, from_sleeper
from ffa.service import availability, startsit
from ffa.service.projections import latest_sleeper_status, upcoming_week
from ffa.warehouse import duck

app = FastAPI(
    title="FF Assistant API",
    version=__version__,
    description="Simulation-based fantasy football projections and start/sit decisions.",
)


# ---------------------------------------------------------------- dependencies


# The only warehouse tables the API reads (dims such as player_ids are always available).
API_DATASETS = (
    "schedules", "injuries", "team_games", "rosters",
    "projection_sims", "projection_inputs", "projection_teams",
)  # fmt: skip


def get_con() -> Iterator[duckdb.DuckDBPyConnection]:
    con = duck.connect(settings.warehouse_dir, datasets=API_DATASETS)
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
    projections_generated_at: datetime | None = Field(
        None, description="When this week's simulations were run (UTC)"
    )
    nflverse_injuries_fetched_at: datetime | None = Field(
        None,
        description="When we last downloaded nflverse's injury file (not how current "
        "the file is: nflverse publishes no update time and lags official reports)",
    )
    sleeper_status_fetched_at: datetime | None = Field(
        None, description="When Sleeper's player statuses were last fetched (once a day)"
    )
    injury_caveat: str = (
        "Game statuses come from nflverse's injury file and Sleeper's player DB (fetched once "
        "a day: ~5 PM CT, Sunday ~10:45 AM CT). Late scratches and inactives announced after "
        "the last fetch are not reflected; confirm game-day status before lineups lock."
    )
    sources: list[str] = Field(default_factory=list)


SOURCES = [
    "nflverse play-by-play, player stats, snap counts, rosters and injury reports "
    "(github.com/nflverse)",
    "Betting lines (spread, total) as published in nflverse schedules",
    "ffopportunity expected points (github.com/ffverse/ffopportunity), used in evaluation",
    "FF Assistant simulation model: 2,000 simulated games per week; 2025 holdout results "
    "in docs/evaluation_2025.md",
]


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
    injury_source: str | None = Field(None, description="Where the injury information came from")
    injury_updated_at: str | None = Field(None, description="When that source last updated it")
    injury_fetched_at: str | None = Field(None, description="When we fetched it")
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
    recommendation_name: str
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


class NotPlayingOut(BaseModel):
    key: str | None
    name: str
    reason: str = Field(description="bye, out, injured_reserve, pup, non_football_injury, "
                        "suspended, inactive, practice_squad, released, retired, "
                        "not_on_nfl_roster, unknown")  # fmt: skip
    detail: str
    source: str
    may_return: bool = Field(description="Could play later this season (bye: next week)")
    fantasy_slot: str | None = Field(None, description="'IR' if in your league's IR slot")


class MyTeam(BaseModel):
    league: str
    team: str
    season: int
    week: int
    scoring: str
    lineup: list[LineupSlot]
    bench: list[Player]
    not_playing: list[NotPlayingOut] = Field(
        description="Rostered players who won't play this week, each with a reason (bye, "
        "out, injured_reserve, pup, suspended, released, ...) and its source"
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
    info = WeekInfo(season=season, week=w, projections_available=False, sources=SOURCES)
    if w is None:
        return info

    # Only "no data yet" maps to null; any other error surfaces instead of reading as
    # "never fetched".
    try:
        info.projections_generated_at = con.sql(
            f"select max(_ingested_at) from {duck.week(con, 'projection_sims', season, w)}"
        ).fetchone()[0]
    except duck.MissingPartition:
        info.projections_generated_at = None
    info.projections_available = info.projections_generated_at is not None
    try:
        info.nflverse_injuries_fetched_at = con.sql(
            f"select max(_ingested_at) from {duck.week(con, 'injuries', season, w)}"
        ).fetchone()[0]
    except duck.MissingPartition:  # this week's report not published yet: last week's
        try:
            info.nflverse_injuries_fetched_at = con.sql(
                f"select max(_ingested_at) from {duck.week(con, 'injuries', season, w - 1)}"
            ).fetchone()[0]
        except duck.MissingPartition:
            info.nflverse_injuries_fetched_at = None
    info.sleeper_status_fetched_at = latest_sleeper_status(con, season, w)[1]
    return info


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
    name = next(p["name"] for p in d["players"] if p["key"] == d["recommendation"])
    return StartSitResponse(scoring=label, recommendation_name=name, **d)


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
    bench = sorted(
        (v for k, v in views.items()
         if k not in starters and v.play_probability >= startsit.OUT_THRESHOLD),
        key=lambda v: -v.mean,
    )  # fmt: skip
    names = {e.player_key: e.name or e.platform_player_id for e in entries}
    # Not playing: no projection (bye, reserve lists, ...), ruled out, or in the IR slot.
    no_proj = {k: names.get(k) or k for k in set(keys) - set(views)}
    out = availability.reasons(con, season, w, no_proj)
    for v in views.values():
        if v.play_probability < startsit.OUT_THRESHOLD:
            prefixes = ("Injury", "On injured", "Inactive")
            line = next((d for d in v.drivers if d.startswith(prefixes)), "Ruled out")
            out.append(availability.NotPlaying(
                v.key, v.name, "out", line, v.injury_source or "nflverse", True))  # fmt: skip
    ir_slot = {e.player_key: e.name or e.platform_player_id for e in entries
               if e.slot_type == "ir" and e.player_key}  # fmt: skip
    for r in availability.reasons(con, season, w, ir_slot):
        r.fantasy_slot = "IR"
        out.append(r)
    for e in entries:  # rostered but unmatched to an NFL player
        if not e.player_key:
            out.append(availability.NotPlaying(
                None, e.name or e.platform_player_id, "unknown",
                "Player not matched to the NFL data", "league sync", False,
                "IR" if e.slot_type == "ir" else None))  # fmt: skip
    return MyTeam(
        league=row.id, team=team.name, season=season, week=w, scoring=label,
        lineup=[LineupSlot(**s) for s in lineup],
        bench=[Player(**startsit.as_dict(v)) for v in bench],
        not_playing=[NotPlayingOut(**r.as_dict()) for r in sorted(out, key=lambda r: r.name)],
    )  # fmt: skip
