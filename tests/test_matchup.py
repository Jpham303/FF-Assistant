import polars as pl
import pytest

from ffa.model import matchup
from ffa.model.asof import Decay


def player_games(weeks: int = 6) -> pl.DataFrame:
    """Two defenses facing the same kind of back: SOFT allows double his usual yards per
    carry, HARD allows exactly his usual."""
    rows = []
    for w in range(1, weeks + 1):
        for defense, ypc in (("SOFT", 8.0), ("HARD", 4.0)):
            rows.append(dict(
                game_id=f"{defense}{w}", season=2024, week=w, opponent=defense,
                position="RB", rush_att=20.0, rush_yd=20 * ypc, ypc_s=4.0,
                targets=4.0, rec=3.0, rec_yd=24.0, tgt_share_s=0.1, team_targets=40.0,
                catch_rate_s=0.75, ypr_s=8.0,
            ))  # fmt: skip
    return pl.DataFrame(rows)


def model(k: float) -> matchup.MatchupModel:
    return matchup.MatchupModel(
        decay={n: Decay(1e9, 0) for n in matchup.RATINGS}, k={n: k for n in matchup.RATINGS}
    )


def test_expected_and_defense_totals():
    dg = matchup.defense_games(matchup.expectations(player_games()))
    soft = dg.filter(pl.col("game_id") == "SOFT1").row(0, named=True)
    assert (soft["ypc_RB_act"], soft["ypc_RB_exp"]) == (160, 80)
    assert (soft["tgt_RB_act"], soft["tgt_RB_exp"]) == (4, 4)
    assert soft["catch_RB_exp"] == 3 and soft["ypr_RB_exp"] == 24


def test_ratings_from_earlier_games_only_and_shrunk():
    dg = matchup.defense_games(matchup.expectations(player_games()))
    r = model(k=80).ratings(dg)
    soft = r.filter(pl.col("defense") == "SOFT").sort("week")
    assert soft["ypc_RB"][0] == 1.0  # week 1: no history, neutral
    # Week 6: 5 earlier games at 2.0x with 400 expected yards of evidence, shrunk by k=80.
    assert soft["ypc_RB"][-1] == pytest.approx((2.0 * 400 + 1.0 * 80) / 480)
    hard = r.filter(pl.col("defense") == "HARD")
    assert hard["ypc_RB"].to_list() == pytest.approx([1.0] * 6)
    assert r["tgt_RB"].to_list() == pytest.approx([1.0] * 12)  # targets matched expectation


def test_switched_off_rating_is_neutral():
    dg = matchup.defense_games(matchup.expectations(player_games()))
    r = model(k=1e9).ratings(dg)
    assert r["ypc_RB"].to_list() == pytest.approx([1.0] * 12, abs=1e-5)
