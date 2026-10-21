"""DuckDB views over the Parquet warehouse.

For each dataset two views are created:
    {dataset}          latest snapshot per (season, week) — what the app uses
    {dataset}_history  every snapshot — what the backtest uses with `as_of`
Each file in dims/ (e.g. player_ids) becomes a plain view of the same name.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import duckdb


def connect(
    root: str | Path, database: str = ":memory:", datasets: Iterable[str] | None = None
) -> duckdb.DuckDBPyConnection:
    """Views over the warehouse. `datasets` limits which dataset views are created: each one
    reads every file's metadata, which is slow on a cold cache or a Docker bind mount, so
    callers that need only a few tables (the API) should name them."""
    root = Path(root)
    con = duckdb.connect(database)
    _ROOTS[id(con)] = root
    dirs = sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []
    if datasets is not None:
        wanted = set(datasets)
        dirs = [d for d in dirs if d.name in wanted]
    for dim in sorted((root / "dims").glob("*.parquet")) if root.exists() else []:
        con.execute(
            f"CREATE OR REPLACE VIEW {dim.stem} AS SELECT * FROM read_parquet('{dim.as_posix()}')"
        )
    for ds_dir in dirs:
        # Only dataset folders become views; anything without Parquet files is skipped.
        if ds_dir.name == "dims" or next(ds_dir.rglob("*.parquet"), None) is None:
            continue
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


_ROOTS: dict[int, Path] = {}


class MissingPartition(LookupError):
    """No files for this dataset, season and week."""


def week(con: duckdb.DuckDBPyConnection, dataset: str, season: int, week: int) -> str:
    """A SQL table expression over ONE week's files of `dataset` (latest snapshot).

    Use this, not the all-weeks view, when a query needs a single week: it can't be broken
    by other weeks written in an older format. (DuckDB 1.5 mis-maps columns by position
    when files with different column sets are scanned together with `select *` and a
    filter, despite union_by_name.) Raises MissingPartition if the week has no files."""
    root = _ROOTS.get(id(con))
    if root is None:
        raise ValueError("connection was not opened with duck.connect")
    d = root / dataset / f"season={int(season)}" / f"week={int(week):02d}"
    if next(d.glob("*.parquet"), None) is None:
        raise MissingPartition(f"{dataset} {season} week {week}")
    glob = (d / "*.parquet").as_posix().replace("'", "''")
    return f"""(SELECT * EXCLUDE (filename) FROM read_parquet('{glob}', hive_partitioning = true,
                    hive_types = {{'season': INTEGER, 'week': INTEGER}}, union_by_name = true,
                    filename = true)
               QUALIFY _ingested_at = max(_ingested_at) OVER ())"""
