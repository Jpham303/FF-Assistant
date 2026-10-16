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
from ffa.warehouse import store

N_SIMS = 2000


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


def _sleeper_status(con, season: int, week: int) -> pl.DataFrame | None:
    """Latest Sleeper status snapshot for the week, if one has been fetched."""
    try:
        return con.sql(
            f"select * exclude (_ingested_at) from sleeper_status "
            f"where season = {int(season)} and week = {int(week)}"
        ).pl()
    except Exception:  # not fetched yet: nflverse alone
        return None


@dataclass
class RunResult:
    season: int
    week: int
    players: int
    rows: int


def run_week(con, warehouse_dir: str | Path, models: Models, season: int, week: int,
             n: int = N_SIMS, now: datetime | None = None) -> RunResult:  # fmt: skip
    q = lambda t: con.sql(f"select * from {t} where season <= {int(season)}").pl()  # noqa: E731
    inp = build(
        season=season, week=week, models=models,
        team_games=q("team_games"), player_games=q("player_games"),
        rosters=con.sql(f"select * from rosters where season = {int(season)}").pl(),
        injuries=con.sql(f"select * from injuries where season = {int(season)}").pl(),
        statuses=_sleeper_status(con, season, week),
    )  # fmt: skip
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
    store.write(summary, warehouse_dir, "projection_history", volatile=True, ingested_at=now)
    return RunResult(season, week, inp.players.height, lines.height)
