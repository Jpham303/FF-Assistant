"""Engine vs nflverse's own fantasy points on every real player-week in the warehouse.

Skipped when the warehouse is empty (CI). Run locally after a backfill:
    uv run pytest tests/test_scoring_data.py
"""

from pathlib import Path

import polars as pl
import pytest

from ffa.config import settings
from ffa.scoring import nflverse_standard, player_stat_lines, score
from ffa.scoring.defense import load_pbp, normalize_teams, scoring_plays
from ffa.warehouse import duck

pytestmark = pytest.mark.skipif(
    not (Path(settings.warehouse_dir) / "pbp").exists(),
    reason="warehouse not loaded",
)


def test_scoring_events_reproduce_every_final_score():
    """Play-by-play scoring events must add up to every final score, all seasons."""
    con = duck.connect(settings.warehouse_dir)
    seasons = [s for (s,) in con.sql("select distinct season from pbp order by 1").fetchall()]
    checked = 0
    for season in seasons:  # one season at a time keeps memory flat
        pbp = load_pbp(con, season)
        sched = normalize_teams(
            con.sql(f"select * from schedules where season = {season}").pl()
        ).filter(pl.col("home_score").is_not_null())
        finals = pl.concat(
            [
                sched.select("game_id", pl.col(f"{s}_team").alias("scorer"),
                             pl.col(f"{s}_score").cast(pl.Float64).alias("final"))
                for s in ("home", "away")
            ]
        )  # fmt: skip
        ours = scoring_plays(pbp).group_by("game_id", "scorer").agg(pl.col("points").sum())
        j = finals.join(ours, on=["game_id", "scorer"], how="left").with_columns(
            pl.col("points").fill_null(0)
        )
        wrong = j.filter(pl.col("points") != pl.col("final"))
        assert wrong.height == 0, f"{season}: {wrong.head(3).to_dicts()}"
        checked += j.height
    assert checked > 1000


@pytest.mark.parametrize(("ppr", "column"), [(1.0, "fantasy_points_ppr"), (0.0, "fantasy_points")])
def test_engine_reproduces_nflverse(ppr, column):
    raw = duck.connect(settings.warehouse_dir).sql("select * from player_stats").pl()
    ours = score(player_stat_lines(raw), nflverse_standard(ppr))
    ref = raw.select(
        pl.col("player_id").alias("key"),
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        pl.col(column).alias("ref"),
        # nflverse counts scrimmage fumbles only; the platforms (and we) also count returns.
        (
            pl.col("fumbles_lost_total")
            - pl.col("sack_fumbles_lost")
            - pl.col("rushing_fumbles_lost")
            - pl.col("receiving_fumbles_lost")
        )
        .fill_null(0)
        .alias("return_fumbles"),
    )
    out = ours.join(ref, on=["key", "season", "week"])
    residual = (pl.col("fpts") - pl.col("ref") + 2 * pl.col("return_fumbles")).abs()
    assert out.height > 100_000
    assert out.filter(residual > 0.011).height == 0
