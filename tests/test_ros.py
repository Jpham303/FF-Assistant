"""Rest-of-season pieces: team ratings from lines, availability over the horizon, and
the as-of cut that keeps backtests from seeing the future."""

from __future__ import annotations

import polars as pl
import pytest

from ffa.model import horizon, ratings
from ffa.service import ros


def _games():
    """Season 2026, weeks 1-4: A scores more than B, C allows more than D; week 5 unposted."""
    rows = []
    sched = {1: [("A", "C"), ("B", "D")], 2: [("A", "D"), ("B", "C")],
             3: [("C", "A"), ("D", "B")], 4: [("D", "A"), ("C", "B")],
             5: [("A", "B"), ("C", "D")]}  # fmt: skip
    off = {"A": 4.0, "B": -4.0, "C": 0.0, "D": 0.0}
    dfn = {"A": 0.0, "B": 0.0, "C": 3.0, "D": -3.0}
    for wk, games in sched.items():
        for home, away in games:
            for team, opp, is_home in ((home, away, True), (away, home, False)):
                imp = 22 + 1.5 * is_home + off[team] + dfn[opp]
                rows.append({"season": 2026, "week": wk, "game_type": "REG", "team": team,
                             "opponent": opp, "is_home": is_home,
                             "implied": None if wk == 5 else imp})  # fmt: skip
    df = pl.DataFrame(rows)
    opp = df.select(
        "season",
        "week",
        pl.col("opponent").alias("team"),
        pl.col("team").alias("opponent"),
        pl.col("implied").alias("opp_implied"),
    )
    df = df.join(opp, on=["season", "week", "team", "opponent"])
    return df.with_columns(
        (pl.col("implied") - pl.col("opp_implied")).alias("spread_for"),
        (pl.col("implied") + pl.col("opp_implied")).alias("total_line"),
    )


def test_ratings_recover_offense_defense_and_home():
    r = ratings.fit(_games(), 2026, 4, ridge=0.01, decay=1.0)
    assert r.off["A"] - r.off["B"] == pytest.approx(8.0, abs=0.1)
    assert r.defense["C"] - r.defense["D"] == pytest.approx(6.0, abs=0.1)
    assert r.home == pytest.approx(1.5, abs=0.1)


def test_fill_lines_only_fills_missing_and_keeps_games_consistent():
    tg = _games()
    r = ratings.fit(tg, 2026, 4, ridge=0.01, decay=1.0)
    f = ratings.fill_lines(tg, r, 2026, 4)
    wk5 = f.filter(pl.col("week") == 5)
    assert wk5["lines_from_ratings"].all() and wk5["implied"].is_not_null().all()
    a = wk5.filter(pl.col("team") == "A").row(0, named=True)
    b = wk5.filter(pl.col("team") == "B").row(0, named=True)
    assert a["implied"] == pytest.approx(22 + 1.5 + 4, abs=0.2)  # A at home vs B
    assert a["spread_for"] == pytest.approx(-b["spread_for"]) and a["total_line"] == b["total_line"]
    assert not f.filter(pl.col("week") <= 4)["lines_from_ratings"].any()  # posted lines kept
    assert f.filter(pl.col("week") <= 4)["implied"].equals(
        tg.filter(pl.col("week") <= 4)["implied"]
    )


def test_horizon_categories_and_fallbacks():
    assert horizon.category("RES", None) == "reserve"
    assert horizon.category("ACT", "Out") == "out" and horizon.category("INA", None) == "inactive"
    assert horizon.category("ACT", None) == "healthy"
    t = {"RB|out|starter|2": 0.6, "any|out|role|9": 0.0, "any|out|any|8": 0.7,
         "any|healthy|starter|1": 0.9}  # fmt: skip
    assert horizon.p_play(t, "RB", "out", "starter", 2) == 0.6
    assert horizon.p_play(t, "WR", "out", "starter", 12) == 0.7  # k capped at 8, pooled role
    assert horizon.p_play(t, "QB", "questionable", "starter", 1) == 0.9  # unknown cat: healthy


def test_as_of_hides_results_rosters_and_future_lines():
    tg = _games().with_columns(pl.lit(30).alias("plays"))
    pg = pl.DataFrame({"key": ["p"] * 4, "season": [2026] * 4, "week": [1, 2, 3, 4]})
    ro = pl.DataFrame({"season": [2026] * 2, "week": [3, 5], "gsis_id": ["p", "p"]})
    inj = pl.DataFrame({"season": [2026] * 2, "week": [3, 5], "gsis_id": ["p", "p"]})
    f = ros.as_of(ros.Frames(tg, pg, ro, inj), 2026, 3, mask_future_lines=True)
    assert f.player_games["week"].max() == 2  # week 3 not played yet
    assert f.rosters["week"].max() == 3 and f.injuries["week"].max() == 3
    later = f.team_games.filter(pl.col("week") >= 3)
    assert later["plays"].is_null().all()  # no results from week 3 on
    assert f.team_games.filter(pl.col("week") == 3)["implied"].is_not_null().all()  # posted
    assert f.team_games.filter(pl.col("week") == 4)["implied"].is_null().all()  # not yet


def test_team_game_index_skips_byes():
    tg = pl.DataFrame({"season": [2026] * 4, "week": [5, 6, 8, 9], "team": ["A"] * 4,
                       "game_type": ["REG"] * 4})  # fmt: skip
    k = ros.team_game_index(tg, 2026, 5)
    assert dict(zip(k["week"].to_list(), k["k"].to_list(), strict=True)) == {6: 1, 8: 2, 9: 3}


def test_todays_category_uses_this_weeks_full_picture():
    st = pl.DataFrame({"key": ["a", "b", "c", "d", "e"], "team": ["X"] * 5,
                       "cat": ["healthy"] * 5, "role": ["starter"] * 5})  # fmt: skip
    players = pl.DataFrame({
        "key": ["a", "b", "c", "d"], "injury_status": [None, "Out", None, None],
        "p_play": [0.01, 0.0, 0.7, 1.0], "roster_status": ["ACT", "ACT", "ACT", "ACT"],
    })  # fmt: skip
    got = dict(ros._with_current(st, players).select("key", "cat").iter_rows())
    assert got == {"a": "out", "b": "out", "c": "questionable", "d": "healthy",
                   "e": "healthy"}  # e: on bye this week, unchanged  # fmt: skip


def test_roster_keeps_teams_on_bye_in_the_latest_week():
    """nflverse's weekly rosters leave out teams on bye: their players must still be on
    the roster (each team's own latest week), or rest of season loses them entirely."""
    from ffa.model.inputs import _roster

    def row(key, team, week, status="ACT"):
        return {"gsis_id": key, "full_name": key, "team": team, "position": "WR",
                "status": status, "season": 2026, "week": week}  # fmt: skip

    r = pl.DataFrame([
        row("A", "PHI", 5), row("A", "PHI", 6),
        row("W", "JAX", 5),                          # JAX on bye in week 6
        row("T", "JAX", 5), row("T", "KC", 6),       # traded JAX -> KC: newer listing wins
        row("S", "NYJ", 2),                          # stale team: dropped
    ])  # fmt: skip
    out = _roster(r, 2026, 6).sort("key")
    assert out["key"].to_list() == ["A", "T", "W"]
    assert dict(zip(out["key"], out["team"], strict=True)) == {"A": "PHI", "T": "KC", "W": "JAX"}
