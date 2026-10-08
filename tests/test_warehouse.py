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
