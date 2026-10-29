"""Trade cards (Oct 29): the rules that turn the model's numbers into a card."""

from __future__ import annotations

import polars as pl
import pytest

from ffa.trade import cards as C
from ffa.trade import core
from tests.test_trade import _league


def test_confidence_from_your_chance_of_coming_out_ahead():
    assert C.confidence(0.95)["level"] == 5 and C.confidence(0.95)["label"] == "confident"
    assert C.confidence(0.80)["level"] == 4 and C.confidence(0.65)["level"] == 3
    assert C.confidence(0.55)["level"] == 2 and C.confidence(0.40)["level"] == 1
    assert C.confidence(None)["level"] == 1


def test_tiers_and_ordinals():
    assert (C.tier(6), C.tier(12), C.tier(13), C.tier(30), C.tier(None)) == (1, 1, 2, 3, None)
    assert [C.ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 32)] == [
        "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "32nd"]  # fmt: skip


def _jefferson(**kw) -> core.Player:
    p = core.Player("J", "Justin Jefferson", "WR", {}, nfl_team="MIN", per_game=14.0)
    p.season_rank, p.ros_rank = kw.get("season_rank", 6), kw.get("ros_rank", 18)
    p.injury, p.injury_context = kw.get("injury"), kw.get("ctx")
    return p


def test_watch_lines_rest_on_numbers():
    p = _jefferson(
        injury="Questionable",
        ctx="Official reports: practice: limited wk 7; "
        "This week: Questionable; Trend: improving; First listed week 3",
    )
    use = {"last_two": [7, 7], "per_game_norm": 8.3, "norm": "last season", "word": "targets"}
    lines = C.watch(p, use, (30, 32))
    assert lines[0].startswith("Questionable: Official reports: practice: limited wk 7")
    assert (
        "7 targets in each of his last two games, under the 8.3 a game he averaged last "
        "season" in lines[1]
    )
    assert "pay a WR2 price, not a WR1 one" in lines[2]  # WR6 so far, WR18 ahead
    assert "hardest WR schedules left (30th of 32)" in lines[3]
    # Nothing worth saying: no lines.
    calm = _jefferson(season_rank=5, ros_rank=7)
    assert C.watch(calm, {"last_two": [8, 9], "per_game_norm": 8.3, "norm": "last season",
                          "word": "targets"}, (5, 32)) == []  # fmt: skip


def test_summary_line():
    p = _jefferson(season_rank=6, ros_rank=6)
    assert C.summary(p, (1, 32), 8.4) == ("WR1 role (WR6 by projected points a game) on the "
                                          "easiest WR schedule left (of 32), and his last box "
                                          "score was 8.4.")  # fmt: skip
    assert "3rd-easiest" in C.summary(p, (3, 32), None)


def test_schedule_ranks_by_points_allowed_to_the_position():
    pg = pl.DataFrame({
        "season": [2026] * 4, "week": [1, 1, 2, 2], "position": ["WR"] * 4,
        "opponent": ["SOFT", "HARD", "SOFT", "HARD"],
        "rec": [8, 2, 9, 3], "rec_yd": [120, 20, 110, 30], "rec_td": [1, 0, 2, 0],
    })  # fmt: skip
    tg = pl.DataFrame({"season": [2026] * 4, "week": [3, 3, 4, 4], "game_type": ["REG"] * 4,
                       "team": ["A", "B", "A", "B"],
                       "opponent": ["SOFT", "HARD", "SOFT", "HARD"]})  # fmt: skip
    r = C.schedule_ranks(pg, tg, 2026, 3, 4)
    assert r[("A", "WR")] == (1, 2) and r[("B", "WR")] == (2, 2)


def test_usable_value_counts_starters_above_replacement_only():
    lg = _league()
    lg.replacement = {("RB", w): 7.0 for w in lg.weeks} | {("WR", w): 6.0 for w in lg.weeks}
    v = core.usable_value(lg, lg.teams["me"].roster)
    # Starters QB1 20 (no QB on waivers: 0 replacement), RB1 15 - 7, RB2 10 - 7,
    # WR1 12 - 6, flex WR2 6 - 6 = 0 -> 20 + 8 + 3 + 6 = 37; RB3 on the bench: nothing.
    assert v == pytest.approx(37.0)
    assert core.usable_value(lg, [*lg.teams["me"].roster, "RBFA"]) == pytest.approx(37.0)
