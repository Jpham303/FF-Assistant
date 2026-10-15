import numpy as np
import polars as pl
import pytest

from ffa import evaluate


def test_prior_mean_uses_earlier_weeks_then_last_season():
    cur = pl.DataFrame(
        {"key": ["a", "a", "a"], "week": [1, 2, 3], "actual": [10.0, 20.0, 30.0]},
        schema_overrides={"week": pl.Int32},
    )
    prev = pl.DataFrame(
        {"key": ["a", "b"], "week": [1, 1], "actual": [4.0, 7.0]},
        schema_overrides={"week": pl.Int32},
    )
    out = evaluate._prior_mean(cur, "actual", prev)
    a = dict(out.filter(pl.col("key") == "a").select("week", "base_actual").iter_rows())
    assert a[1] == 4.0  # nothing earlier this season: last season's average
    assert a[2] == 10.0 and a[3] == 15.0 and a[4] == 20.0  # only strictly earlier weeks
    b = dict(out.filter(pl.col("key") == "b").select("week", "base_actual").iter_rows())
    assert b[5] == 7.0


def test_pair_probabilities_from_joint_draws():
    draws = pl.DataFrame(
        {
            "key": ["A"] * 4 + ["B"] * 4,
            "week": [1] * 8,
            "sim": [0, 1, 2, 3] * 2,
            "fpts": [10.0, 5.0, 8.0, 3.0, 9.0, 5.0, 2.0, 4.0],
        },
        schema_overrides={"week": pl.Int32},
    )
    pairs = pl.DataFrame(
        {"a_key": ["A"], "b_key": ["B"], "a_week": [1]}, schema_overrides={"a_week": pl.Int32}
    )
    p = evaluate.pair_probabilities(draws, pairs)["p_a"].item()
    assert p == pytest.approx((1 + 0.5 + 1 + 0) / 4)  # win, tie, win, loss


def test_accuracy_and_bootstrap():
    rng = np.random.default_rng(0)
    n = 400
    actual = rng.normal(10, 4, n)
    rows = pl.DataFrame(
        {
            "key": [f"p{i}" for i in range(n)],
            "week": np.repeat(np.arange(1, 21), 20).astype(np.int32),
            "position": ["WR"] * n,
            "played": [True] * n,
            "actual": actual,
            "proj": actual + rng.normal(0, 2, n),  # a good model
            "base_season_avg": actual + rng.normal(0, 4, n),  # a worse baseline
            "base_opportunity": actual + rng.normal(0, 3, n),
            "p10": actual - 5,
            "p90": actual + 5,
        }
    )
    acc = evaluate.accuracy(evaluate.Holdout(rows, pl.DataFrame())).row(0, named=True)
    assert acc["mae_model"] < acc["mae_opportunity"] < acc["mae_season_avg"]
    lo, hi = (float(x) for x in acc["edge_vs_season_avg_ci"].strip("[]").split(","))
    assert 0 < lo < acc["edge_vs_season_avg"] < hi
