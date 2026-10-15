"""Engine vs nflverse's own fantasy points on every real player-week in the warehouse.

Skipped when the warehouse is empty (CI). Run locally after a backfill:
    uv run pytest tests/test_scoring_data.py
"""

from pathlib import Path

import polars as pl
import pytest

from ffa.config import settings
from ffa.scoring import nflverse_standard, player_stat_lines, score
from ffa.warehouse import duck

pytestmark = pytest.mark.skipif(
    not (Path(settings.warehouse_dir) / "player_stats").exists(),
    reason="warehouse not loaded",
)


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
