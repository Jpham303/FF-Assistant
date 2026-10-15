import polars as pl
import pytest

from ffa.model.asof import Decay, decayed, shrink


def games() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "team": ["A", "A", "A", "A", "B"],
            "season": [2025, 2025, 2025, 2026, 2025],
            "week": [1, 2, 3, 1, 1],
            "plays": [60.0, 70.0, None, 64.0, 55.0],  # A week 3 not played yet
            "targets": [10.0, 5.0, None, 8.0, 3.0],
            "team_targets": [40.0, 30.0, None, 32.0, 30.0],
        }
    )


def test_uses_only_earlier_games():
    d = decayed(games(), "team", {"pace": ("plays", None)}, Decay(half_life=1e9))
    a = d.filter(pl.col("team") == "A")["pace"].to_list()
    # Week 1: no history. Week 2: just week 1. Week 3: mean of 1-2 (equal weights with an
    # enormous half-life). 2026 week 1: still 1-2, since week 3 has no result.
    assert a[0] is None and a[1:] == pytest.approx([60.0, 65.0, 65.0])
    assert d.filter(pl.col("team") == "B")["pace"].to_list() == [None]


def test_recent_games_weigh_more_and_offseason_decays():
    d = decayed(games(), "team", {"pace": ("plays", None)}, Decay(half_life=1, offseason_gap=0))
    wk3 = d.filter((pl.col("team") == "A") & (pl.col("week") == 3)).row(0, named=True)
    # Weights: week 2 -> 0.5 (1 game ago), week 1 -> 0.25.
    assert wk3["pace"] == pytest.approx((70 * 0.5 + 60 * 0.25) / 0.75)
    assert wk3["pace_w"] == pytest.approx(0.75)
    gap = decayed(games(), "team", {"pace": ("plays", None)}, Decay(half_life=1, offseason_gap=8))
    nxt = gap.filter((pl.col("team") == "A") & (pl.col("season") == 2026)).row(0, named=True)
    assert nxt["pace_w"] < 0.75 * 0.5**8  # last season counts, but much less


def test_ratio_weights_by_denominator():
    d = decayed(games(), "team", {"tshare": ("targets", "team_targets")}, Decay(half_life=1e9))
    wk3 = d.filter((pl.col("team") == "A") & (pl.col("week") == 3)).row(0, named=True)
    assert wk3["tshare"] == pytest.approx(15 / 70)  # pooled, not the mean of 0.25 and 0.167
    assert wk3["tshare_w"] == pytest.approx(70)


def test_shrink():
    df = pl.DataFrame({"v": [0.30, None], "w": [30.0, None]})
    out = df.select(shrink("v", "w", 0.20, k=10).alias("s"))["s"].to_list()
    assert out == pytest.approx([(0.30 * 30 + 0.20 * 10) / 40, 0.20])
