"""Dagster definitions: nflverse ingestion assets and schedules.

No `from __future__ import annotations` here: Dagster inspects the real type of `context`.
"""

import polars as pl
from dagster import (
    AssetExecutionContext,
    AssetSelection,
    Definitions,
    MaterializeResult,
    RunRequest,
    ScheduleEvaluationContext,
    StaticPartitionsDefinition,
    asset,
    define_asset_job,
    schedule,
)

from ffa.config import settings
from ffa.db.session import session as db_session
from ffa.ingest import ids, nflverse
from ffa.leagues import sync
from ffa.leagues.sleeper import SleeperAdapter
from ffa.warehouse import duck, store

FIRST_SEASON = 2015
CURRENT = str(settings.current_season)
season_partitions = StaticPartitionsDefinition(
    [str(s) for s in range(FIRST_SEASON, settings.current_season + 1)]
)


def _make_asset(dataset: str):
    ds = nflverse.DATASETS[dataset]

    @asset(
        name=dataset,
        partitions_def=season_partitions,
        group_name="nflverse",
        description=f"nflverse {dataset}, one partition per season.",
    )
    def _ingest(context: AssetExecutionContext) -> MaterializeResult:
        season = int(context.partition_key)
        try:
            df = nflverse.fetch(dataset, season)
        except nflverse.NotPublishedError:
            context.log.warning(f"{dataset} {season}: not published yet")
            return MaterializeResult(metadata={"rows": 0, "partitions": 0})
        paths = store.write(df, settings.warehouse_dir, dataset, volatile=ds.volatile)
        weeks = sorted({int(p.parent.name.split("=")[1]) for p in paths})
        return MaterializeResult(
            metadata={"rows": df.height, "partitions": len(paths), "weeks": str(weeks)}
        )

    return _ingest


ingest_assets = [_make_asset(name) for name in nflverse.DATASETS]

nflverse_job = define_asset_job(
    "nflverse_ingest",
    selection=AssetSelection.groups("nflverse"),
)
injuries_job = define_asset_job(
    "injuries_refresh",
    selection=AssetSelection.assets("injuries", "rosters"),
)


@schedule(job=nflverse_job, cron_schedule="15 2 * * *", execution_timezone="America/Chicago")
def nightly_current_season(context: ScheduleEvaluationContext):
    """All datasets for the current season, after nflverse's overnight builds."""
    return RunRequest(partition_key=CURRENT, run_key=f"nightly-{context.scheduled_execution_time}")


@schedule(job=injuries_job, cron_schedule="5 * * * *", execution_timezone="America/Chicago")
def hourly_injuries(context: ScheduleEvaluationContext):
    """Injury reports and depth charts change through the week; snapshot them hourly."""
    return RunRequest(partition_key=CURRENT, run_key=f"hourly-{context.scheduled_execution_time}")


@asset(group_name="dims", deps=["rosters"])
def player_ids(context: AssetExecutionContext) -> MaterializeResult:
    """Crosswalk from nflverse gsis_id to Sleeper and Yahoo IDs, with source and conflicts."""
    rosters = (
        duck.connect(settings.warehouse_dir)
        .sql("select gsis_id, sleeper_id, yahoo_id, season, week from rosters")
        .pl()
    )
    try:
        sleeper = ids.load_sleeper_players()
    except Exception as exc:  # the other sources still produce a usable crosswalk
        context.log.warning(f"Sleeper player DB unavailable, continuing without it: {exc}")
        sleeper = None
    xw = ids.build_crosswalk(
        ids.load_nflverse_players(), rosters, ids.load_dynastyprocess(), sleeper
    )
    store.write_dim(xw, settings.warehouse_dir, "player_ids")
    if sleeper is not None:  # names for Sleeper IDs the crosswalk doesn't know yet
        store.write_dim(sleeper, settings.warehouse_dir, "sleeper_players")

    active = xw.filter(
        pl.col("position").is_in(ids.FANTASY_POSITIONS)
        & (pl.col("last_season") >= settings.current_season - 1)
    )
    return MaterializeResult(
        metadata={
            "players": xw.height,
            "active_fantasy_players": active.height,
            "active_with_sleeper_id": int(active["sleeper_id"].is_not_null().sum()),
            "active_with_yahoo_id": int(active["yahoo_id"].is_not_null().sum()),
            "sleeper_id_conflicts": int(xw["sleeper_id_conflict"].sum()),
            "yahoo_id_conflicts": int(xw["yahoo_id_conflict"].sum()),
            "sleeper_db_used": sleeper is not None,
        }
    )


dims_job = define_asset_job("dims_refresh", selection=AssetSelection.groups("dims"))


@schedule(job=dims_job, cron_schedule="45 2 * * *", execution_timezone="America/Chicago")
def nightly_dims(context: ScheduleEvaluationContext):
    """Rebuild the ID crosswalk after the nightly roster load."""
    return RunRequest(run_key=f"dims-{context.scheduled_execution_time}")


@asset(group_name="leagues", deps=["player_ids"])
def sleeper_league(context: AssetExecutionContext) -> MaterializeResult:
    """Your Sleeper league's settings, teams and rosters, resolved to canonical player keys
    and stored in Postgres. Skipped when SLEEPER_LEAGUE_ID is not set."""
    if not settings.sleeper_league_id:
        context.log.info("SLEEPER_LEAGUE_ID not set; skipping")
        return MaterializeResult(metadata={"skipped": True})

    resolver = ids.Resolver(
        duck.connect(settings.warehouse_dir).sql("select * from player_ids").pl()
    )
    adapter = SleeperAdapter(
        me=settings.sleeper_username or None,
        players=sync.load_sleeper_players(settings.warehouse_dir),
    )
    with db_session() as db:
        rep = sync.sync(adapter, settings.sleeper_league_id, resolver, db)
    for u in rep.unmatched:
        context.log.warning(f"Unmatched player on {u.team}: {u.platform_id} {u.name or ''}")
    return MaterializeResult(
        metadata={
            "league": rep.league,
            "teams": rep.teams,
            "players": rep.players,
            "match_rate": round(rep.match_rate, 4),
            "by_method": str(rep.by_method),
            "unmatched": str([u.platform_id for u in rep.unmatched]),
            "unsupported_slots": str(rep.unsupported_slots),
        }
    )


leagues_job = define_asset_job("leagues_refresh", selection=AssetSelection.groups("leagues"))


@schedule(job=leagues_job, cron_schedule="15 * * * *", execution_timezone="America/Chicago")
def hourly_leagues(context: ScheduleEvaluationContext):
    """Rosters change with waivers and trades; resync hourly."""
    return RunRequest(run_key=f"leagues-{context.scheduled_execution_time}")


defs = Definitions(
    assets=[*ingest_assets, player_ids, sleeper_league],
    jobs=[nflverse_job, injuries_job, dims_job, leagues_job],
    schedules=[nightly_current_season, hourly_injuries, nightly_dims, hourly_leagues],
)
