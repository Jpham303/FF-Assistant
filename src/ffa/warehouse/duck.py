"""DuckDB views over the Parquet warehouse.

For each dataset two views are created:
    {dataset}          latest snapshot per (season, week) — what the app uses
    {dataset}_history  every snapshot — what the backtest uses with `as_of`
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb


def connect(root: str | Path, database: str = ":memory:") -> duckdb.DuckDBPyConnection:
    root = Path(root)
    con = duckdb.connect(database)
    for ds_dir in sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []:
        name = ds_dir.name
        glob = (ds_dir / "**" / "*.parquet").as_posix()
        con.execute(
            f"""
            CREATE OR REPLACE VIEW {name}_history AS
            SELECT * FROM read_parquet('{glob}', hive_partitioning = true,
                                       hive_types = {{'season': INTEGER, 'week': INTEGER}},
                                       union_by_name = true, filename = true)
            """
        )
        con.execute(
            f"""
            CREATE OR REPLACE VIEW {name} AS
            SELECT * EXCLUDE (filename) FROM {name}_history
            QUALIFY _ingested_at = max(_ingested_at) OVER (PARTITION BY season, week)
            """
        )
    return con


def as_of(
    con: duckdb.DuckDBPyConnection, dataset: str, cutoff: datetime
) -> duckdb.DuckDBPyRelation:
    """Rows of `dataset` as they stood at `cutoff`: the latest snapshot fetched before it."""
    return con.sql(
        f"""
        SELECT * EXCLUDE (filename) FROM {dataset}_history
        WHERE _ingested_at <= $cutoff
        QUALIFY _ingested_at = max(_ingested_at) OVER (PARTITION BY season, week)
        """,
        params={"cutoff": cutoff},
    )
