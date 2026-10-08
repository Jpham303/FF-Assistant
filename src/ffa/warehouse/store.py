"""Parquet warehouse, partitioned by dataset / season / week.

Layout:
    {root}/{dataset}/season=2026/week=05/data.parquet              stable datasets
    {root}/{dataset}/season=2026/week=05/snap_20261008T163000Z.parquet   volatile datasets

Every row carries `_ingested_at` (UTC), the moment it was fetched. Point-in-time queries
use it to reconstruct what was knowable before a given kickoff.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl


def _partition_dir(root: Path, dataset: str, season: int, week: int) -> Path:
    return root / dataset / f"season={season}" / f"week={week:02d}"


def write(
    df: pl.DataFrame,
    root: str | Path,
    dataset: str,
    *,
    volatile: bool = False,
    ingested_at: datetime | None = None,
) -> list[Path]:
    """Split `df` by (season, week) and write one file per partition. Returns written paths."""
    if df.is_empty():
        return []
    missing = {"season", "week"} - set(df.columns)
    if missing:
        raise ValueError(f"{dataset}: missing partition columns {missing}")

    ts = (ingested_at or datetime.now(UTC)).astimezone(UTC).replace(microsecond=0)
    df = df.with_columns(pl.lit(ts).alias("_ingested_at"))
    root = Path(root)
    written: list[Path] = []
    for (season, week), part in df.partition_by(["season", "week"], as_dict=True).items():
        if season is None or week is None:
            continue
        out = _partition_dir(root, dataset, int(season), int(week))
        out.mkdir(parents=True, exist_ok=True)
        name = f"snap_{ts:%Y%m%dT%H%M%SZ}.parquet" if volatile else "data.parquet"
        path = out / name
        # Partition values live in the directory names; drop them from the file to avoid
        # type clashes when DuckDB reads with hive partitioning.
        part.drop(["season", "week"]).write_parquet(path)
        written.append(path)
    return written
