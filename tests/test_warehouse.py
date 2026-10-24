from datetime import UTC, datetime

import polars as pl

from ffa.warehouse import duck, store


def _injuries(status: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [2026, 2026],
            "week": [5, 6],
            "gsis_id": ["00-1", "00-1"],
            "report_status": [status, status],
        }
    )


def test_stable_dataset_overwrites_partition(tmp_path):
    df = pl.DataFrame(
        {"season": [2025, 2025, 2025], "week": [1, 1, 2], "player_id": ["a", "b", "a"]}
    )
    store.write(df, tmp_path, "player_stats")
    store.write(df, tmp_path, "player_stats")  # re-run must not duplicate rows
    paths = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.parquet"))
    assert paths == [
        "player_stats/season=2025/week=01/data.parquet",
        "player_stats/season=2025/week=02/data.parquet",
    ]
    con = duck.connect(tmp_path)
    assert con.sql("select count(*) from player_stats").fetchone()[0] == 3
    # Zero-padded folder names must still read back as integers.
    assert con.sql("select max(week) from player_stats").fetchone()[0] == 2


def test_volatile_dataset_keeps_snapshots_and_supports_as_of(tmp_path):
    t1 = datetime(2026, 10, 7, 12, tzinfo=UTC)
    t2 = datetime(2026, 10, 9, 12, tzinfo=UTC)
    store.write(_injuries("Questionable"), tmp_path, "injuries", volatile=True, ingested_at=t1)
    store.write(_injuries("Out"), tmp_path, "injuries", volatile=True, ingested_at=t2)

    con = duck.connect(tmp_path)
    latest = con.sql("select distinct report_status from injuries").fetchall()
    assert latest == [("Out",)]

    # What was known on Oct 8 — before the second report existed.
    before = duck.as_of(con, "injuries", datetime(2026, 10, 8, tzinfo=UTC))
    assert before.select("report_status").distinct().fetchall() == [("Questionable",)]
    assert con.sql("select count(*) from injuries_history").fetchone()[0] == 4


def test_rows_without_week_are_skipped(tmp_path):
    df = pl.DataFrame({"season": [2026, 2026], "week": [1, None], "x": [1, 2]})
    assert len(store.write(df, tmp_path, "rosters")) == 1


def test_folders_without_parquet_are_ignored(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "team_volume.json").write_text("{}")
    store.write(pl.DataFrame({"season": [2026], "week": [1], "x": [1]}), tmp_path, "player_stats")
    con = duck.connect(tmp_path)
    assert con.sql("select count(*) from player_stats").fetchone()[0] == 1


def test_connect_only_requested_datasets(tmp_path):
    import duckdb
    import polars as pl
    import pytest

    from ffa.warehouse import duck, store

    df = pl.DataFrame({"season": [2026], "week": [1], "x": [1]})
    store.write(df, tmp_path, "a")
    store.write(df, tmp_path, "b")
    con = duck.connect(tmp_path, datasets=["a"])
    assert con.sql("show tables").fetchall() == []  # nothing opened until used
    assert con.sql("select x from a").fetchall() == [(1,)]
    with pytest.raises(duckdb.CatalogException):
        con.sql("select * from b")


def test_lazy_views_latest_snapshot_history_and_quoted_paths(tmp_path):
    from datetime import UTC, datetime

    import polars as pl

    from ffa.warehouse import duck, store

    for hour, x in ((1, 10), (2, 20), (3, 30)):
        week7 = pl.DataFrame({"season": [2026], "week": [7], "x": [x]})
        store.write(week7, tmp_path, "snapds", volatile=True,
                    ingested_at=datetime(2026, 10, 24, hour, tzinfo=UTC))  # fmt: skip
    store.write(pl.DataFrame({"season": [2026], "week": [6], "x": [5]}), tmp_path, "snapds",
                volatile=True, ingested_at=datetime(2026, 10, 17, tzinfo=UTC))  # fmt: skip
    con = duck.connect(tmp_path)
    assert sorted(con.sql("select week, x from snapds").fetchall()) == [(6, 5), (7, 30)]
    assert con.sql("select count(*) from snapds_history").fetchall() == [(4,)]
    t = duck.week(con, "snapds", 2026, 7)  # a path in quotes names no view
    con.sql(f"select * from {t}").fetchall()
    cutoff = datetime(2026, 10, 24, 2, 30, tzinfo=UTC)
    assert duck.as_of(con, "snapds", cutoff).pl().sort("week")["x"].to_list() == [5, 20]
    store.write(pl.DataFrame({"season": [2026], "week": [8], "y": [1]}), tmp_path, "plain")
    con2 = duck.connect(tmp_path)
    con2.sql("select * from plain").fetchall()
    store.write(pl.DataFrame({"season": [2026], "week": [9], "y": [2]}), tmp_path, "plain")
    assert con2.sql("select count(*) from plain").fetchall() == [(2,)]  # new weeks picked up
