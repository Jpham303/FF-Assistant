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
from ffa.ingest import ids, nflverse
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


defs = Definitions(
    assets=[*ingest_assets, player_ids],
    jobs=[nflverse_job, injuries_job, dims_job],
    schedules=[nightly_current_season, hourly_injuries, nightly_dims],
)
