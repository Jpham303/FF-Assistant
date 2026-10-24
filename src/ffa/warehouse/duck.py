"""DuckDB views over the Parquet warehouse.

For each dataset two views exist:
    {dataset}          latest snapshot per (season, week): what the app uses
    {dataset}_history  every snapshot: what the backtest uses with `as_of`
Each file in dims/ (e.g. player_ids) becomes a plain view of the same name.

Views are created lazily, the first time a query names them. Creating a view reads the
metadata of every file it covers, about 24 ms a file on a Docker bind mount from a
Windows drive, so opening all ~4,000 files up front took 90 seconds per connection
(Oct 24). The latest view of a snapshotted dataset (snap_*.parquet files, one per fetch)
covers only the newest file in each week's folder, found from file names alone.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import duckdb

_QUOTED = re.compile(r"'(?:[^']|'')*'")
_WORD = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_HIVE = "hive_partitioning = true, hive_types = {'season': INTEGER, 'week': INTEGER}"


def _latest_files(ds_dir: Path) -> list[str]:
    """Per (season, week) folder: data.parquet, or the newest snapshot (names sort by time)."""
    out = []
    for season in sorted(ds_dir.glob("season=*")):
        for week in sorted(season.glob("week=*")):
            snaps = sorted(week.glob("snap_*.parquet"))
            other = sorted(f for f in week.glob("*.parquet") if not f.name.startswith("snap_"))
            out += [f.as_posix() for f in other] + ([snaps[-1].as_posix()] if snaps else [])
    return out


class LazyConnection:
    """A DuckDB connection whose warehouse views are created on first use. Everything
    else is passed through to the underlying connection."""

    def __init__(self, con: duckdb.DuckDBPyConnection, root: Path,
                 datasets: set[str] | None) -> None:  # fmt: skip
        self._con = con
        self._root = root
        names = {p.name for p in root.iterdir() if p.is_dir() and p.name != "dims"} \
            if root.exists() else set()  # fmt: skip
        self._available = names if datasets is None else names & set(datasets)
        self._made: set[str] = set()

    # --- view creation
    def _ensure(self, query: str) -> None:
        words = set(_WORD.findall(_QUOTED.sub("''", query)))
        for w in words:
            if w in self._made:
                continue
            if w in self._available:
                self._make(w, history=False)
            elif w.endswith("_history") and w[: -len("_history")] in self._available:
                self._make(w[: -len("_history")], history=True)

    def _make(self, name: str, history: bool) -> None:
        ds_dir = self._root / name
        if history:
            glob = (ds_dir / "**" / "*.parquet").as_posix().replace("'", "''")
            if next(ds_dir.rglob("*.parquet"), None) is None:
                return
            self._con.execute(f"""CREATE OR REPLACE VIEW {name}_history AS
                SELECT * FROM read_parquet('{glob}', {_HIVE}, union_by_name = true,
                                           filename = true)""")  # fmt: skip
            self._made.add(f"{name}_history")
            return
        if next(ds_dir.glob("season=*/week=*/snap_*.parquet"), None) is None:
            # One file per week (rewritten in place): a glob picks up new weeks too.
            if next(ds_dir.rglob("*.parquet"), None) is None:
                return
            src = "'" + (ds_dir / "**" / "*.parquet").as_posix().replace("'", "''") + "'"
        else:
            files = _latest_files(ds_dir)
            if not files:
                return
            src = "[" + ", ".join("'" + f.replace("'", "''") + "'" for f in files) + "]"
        self._con.execute(f"""CREATE OR REPLACE VIEW {name} AS
            SELECT * EXCLUDE (filename)
            FROM read_parquet({src}, {_HIVE}, union_by_name = true, filename = true)
            QUALIFY _ingested_at = max(_ingested_at) OVER (PARTITION BY season, week)
            """)  # fmt: skip
        self._made.add(name)

    # --- query methods
    def sql(self, query: str, *args, **kwargs):
        self._ensure(query)
        return self._con.sql(query, *args, **kwargs)

    query = sql

    def execute(self, query: str, *args, **kwargs):
        self._ensure(query)
        return self._con.execute(query, *args, **kwargs)

    def close(self) -> None:
        _ROOTS.pop(id(self), None)
        self._con.close()

    def __getattr__(self, name: str):
        return getattr(self._con, name)

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def connect(
    root: str | Path, database: str = ":memory:", datasets: Iterable[str] | None = None
) -> LazyConnection:
    """Views over the warehouse, created when a query first names them. `datasets`
    limits which dataset views may be created (None = any)."""
    root = Path(root)
    con = duckdb.connect(database)
    for dim in sorted((root / "dims").glob("*.parquet")) if root.exists() else []:
        con.execute(
            f"CREATE OR REPLACE VIEW {dim.stem} AS SELECT * FROM read_parquet('{dim.as_posix()}')"
        )
    lazy = LazyConnection(con, root, None if datasets is None else set(datasets))
    _ROOTS[id(lazy)] = root
    _ROOTS[id(con)] = root
    return lazy


def as_of(con: LazyConnection, dataset: str, cutoff: datetime) -> duckdb.DuckDBPyRelation:
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


def week(con: LazyConnection, dataset: str, season: int, week: int) -> str:
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
