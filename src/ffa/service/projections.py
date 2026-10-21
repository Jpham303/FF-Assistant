"""Weekly projections: run the simulator for a week and store what the API serves.

Stored per (season, week), overwritten on each run:
    projection_sims/    simulated canonical stat lines (key, sim, stats) - league-neutral,
                        so any league's scoring is applied at request time
    projection_inputs/  the per-player inputs behind them (shares, efficiency, matchup
                        multipliers, availability) - the raw material for explanations
    projection_teams/   per-team game context (implied total, plays, pass rate)
and, as timestamped snapshots, a small per-player summary in projection_history/ so live
accuracy can be audited later.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl

from ffa.model.inputs import Models, build
from ffa.model.matchup import MatchupModel
from ffa.model.player import PlayerModel
from ffa.model.sim import simulate
from ffa.model.sim_params import SimParams
from ffa.model.team import TeamModel
from ffa.warehouse import duck, store

N_SIMS = 2000

# What the API reads from projection_inputs. A week whose inputs don't match fails the
# pipeline run here rather than the API at request time.
INPUT_SCHEMA: dict[str, type[pl.DataType] | tuple[type[pl.DataType], ...]] = {
    "key": pl.String, "name": pl.String, "team": pl.String, "position": pl.String,
    "opponent": pl.String, "roster_status": pl.String, "injury_status": pl.String,
    "practice": pl.String, "injury_source": pl.String,
    "injury_updated_at": pl.Datetime, "injury_fetched_at": pl.Datetime,
    "prev_out": pl.Boolean, "final_report": pl.Boolean,
    "p_play": pl.Float64, "car_share_s": pl.Float64, "tgt_share_s": pl.Float64,
    "rz_car_share_s": pl.Float64, "rz_tgt_share_s": pl.Float64,
}  # fmt: skip


class SchemaMismatch(ValueError):
    pass


def check_inputs(players: pl.DataFrame) -> None:
    """Raise SchemaMismatch listing every missing or mistyped column."""
    problems = []
    for col, want in INPUT_SCHEMA.items():
        if col not in players.columns:
            problems.append(f"{col}: missing")
            continue
        got = players.schema[col]
        if got != pl.Null and not isinstance(got, want if isinstance(want, tuple) else (want,)):
            problems.append(f"{col}: {got}, expected {want}")
    if problems:
        raise SchemaMismatch("projection_inputs format: " + "; ".join(problems))


def load_models(models_dir: str | Path) -> Models:
    m = Path(models_dir)
    return Models(
        TeamModel.load(m / "team_volume.json"),
        PlayerModel.load(m / "player.json"),
        MatchupModel.load(m / "matchup.json"),
        SimParams.load(m / "sim_params.json"),
    )


def upcoming_week(con, season: int) -> int | None:
    """First week of `season` with a game still unplayed."""
    row = con.sql(
        f"select min(week) from schedules where season = {int(season)} and home_score is null"
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else None


def latest_sleeper_status(con, season: int, week: int):
    """The most recent Sleeper status snapshot that applies to `week`: statuses are the
    current state, so a fetch filed under the previous week (e.g. Monday morning, before
    Monday night's game) still applies. Returns (frame, fetched_at) or (None, None)."""
    for w in (week, week - 1):
        if w < 1:
            continue
        try:
            t = duck.week(con, "sleeper_status", season, w)
        except duck.MissingPartition:
            continue
        df = con.sql(f"select * from {t}").pl()
        fetched = df["_ingested_at"].max() if "_ingested_at" in df.columns and df.height else None
        return df.drop("_ingested_at", strict=False), fetched
    return None, None


def _sleeper_status(con, season: int, week: int) -> pl.DataFrame | None:
    return latest_sleeper_status(con, season, week)[0]


@dataclass
class RunResult:
    season: int
    week: int
    players: int
    rows: int


def run_week(con, warehouse_dir: str | Path, models: Models, season: int, week: int,
             n: int = N_SIMS, now: datetime | None = None,
             record_history: bool = True) -> RunResult:  # fmt: skip
    """Simulate a week and store it. `record_history=False` rebuilds the stored week
    without adding a snapshot to projection_history (the record of what was projected
    at the time), e.g. when regenerating past weeks in the current format."""
    q = lambda t: con.sql(f"select * from {t} where season <= {int(season)}").pl()  # noqa: E731
    inp = build(
        season=season, week=week, models=models,
        team_games=q("team_games"), player_games=q("player_games"),
        rosters=con.sql(f"select * from rosters where season = {int(season)}").pl(),
        injuries=con.sql(f"select * from injuries where season = {int(season)}").pl(),
        statuses=_sleeper_status(con, season, week),
    )  # fmt: skip
    check_inputs(inp.players)
    lines = simulate(inp, models.sim, n=n, seed=season * 100 + week)
    tag = pl.lit(season).cast(pl.Int32).alias("season"), pl.lit(week).cast(pl.Int32).alias("week")
    f32 = [c for c, t in lines.schema.items() if t == pl.Float64]
    store.write(
        lines.with_columns(pl.col(f32).cast(pl.Float32), *tag), warehouse_dir, "projection_sims"
    )
    store.write(inp.players.with_columns(*tag), warehouse_dir, "projection_inputs")
    store.write(inp.teams.with_columns(*tag), warehouse_dir, "projection_teams")
    summary = (
        lines.group_by("key", "name", "team", "position")
        .agg(pl.col(c).mean() for c in f32 if c != "active")
        .with_columns(*tag)
    )
    if record_history:
        store.write(summary, warehouse_dir, "projection_history", volatile=True, ingested_at=now)
    return RunResult(season, week, inp.players.height, lines.height)
