"""Dagster definitions: nflverse ingestion assets and schedules.

No `from __future__ import annotations` here: Dagster inspects the real type of `context`.
"""

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
from ffa.ingest import nflverse
from ffa.warehouse import store

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
    partitions_def=season_partitions,
)
injuries_job = define_asset_job(
    "injuries_refresh",
    selection=AssetSelection.assets("injuries", "rosters"),
    partitions_def=season_partitions,
)


@schedule(job=nflverse_job, cron_schedule="15 2 * * *", execution_timezone="America/Chicago")
def nightly_current_season(context: ScheduleEvaluationContext):
    """All datasets for the current season, after nflverse's overnight builds."""
    return RunRequest(partition_key=CURRENT, run_key=f"nightly-{context.scheduled_execution_time}")


@schedule(job=injuries_job, cron_schedule="5 * * * *", execution_timezone="America/Chicago")
def hourly_injuries(context: ScheduleEvaluationContext):
    """Injury reports and depth charts change through the week; snapshot them hourly."""
    return RunRequest(partition_key=CURRENT, run_key=f"hourly-{context.scheduled_execution_time}")


defs = Definitions(
    assets=ingest_assets,
    jobs=[nflverse_job, injuries_job],
    schedules=[nightly_current_season, hourly_injuries],
)
