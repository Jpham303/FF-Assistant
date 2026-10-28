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


LP, FP, DNP = (
    "Limited Participation in Practice",
    "Full Participation in Practice",
    "Did Not Participate In Practice",
)


def test_practicing_and_designated_to_return_categories():
    """Oct 28: Out-but-practicing players play their next game far more often; a reserve
    player with a game designation (designated to return) goes by the designation."""
    assert horizon.category("ACT", "Out", LP) == "out_practicing"
    assert horizon.category("ACT", "Out", DNP) == "out"
    assert horizon.category("ACT", "Doubtful", FP) == "doubtful_practicing"
    assert horizon.category("ACT", "Questionable", DNP) == "questionable"
    assert horizon.category("RES", "Questionable") == "questionable"
    assert horizon.category("RES", "Out", FP) == "out_practicing"
    assert horizon.category("RES", "Out", DNP) == "reserve"
    t = {"any|out|starter|1": 0.35, "any|out_practicing|starter|1": 0.55}
    assert horizon.p_play(t, "RB", "out_practicing", "starter", 1) == 0.55
    assert horizon.p_play(t, "RB", "doubtful_practicing", "starter", 1) == 1.0  # no cell: healthy
    assert horizon.p_play({"any|out|starter|2": 0.6}, "RB", "out_practicing", "starter", 2) == 0.6


def test_returning_player_takes_last_seasons_role():
    """Oct 28: Charbonnet (48% of snaps in 2025, no 2026 games yet) read as 'depth'."""

    def ro(key, season, week, status="ACT"):
        return {"gsis_id": key, "season": season, "week": week, "team": "SEA",
                "position": "RB", "status": status}  # fmt: skip

    rosters = pl.DataFrame(
        [
            ro("z", 2025, 1),
            ro("z", 2025, 2),
            ro("z", 2026, 1, "RES"),
            ro("z", 2026, 7, "ACT"),
            ro("k", 2026, 1),
            ro("k", 2026, 7),
        ]
    )
    pg = pl.DataFrame({
        "key": ["z", "z", "k"], "season": [2025, 2025, 2026], "week": [1, 2, 1],
        "snaps": [30, 30, 5], "snap_pct": [0.5, 0.5, 0.1], "played": [True, True, True],
    })  # fmt: skip
    inj = pl.DataFrame({"gsis_id": ["z"], "season": [2026], "week": [7],
                        "report_status": ["Out"], "practice_status": [FP]})  # fmt: skip
    st = horizon._standing(rosters, inj, pg).filter(pl.col("season") == 2026)
    z = st.filter((pl.col("key") == "z") & (pl.col("week") == 7)).row(0, named=True)
    assert (z["role"], z["returning"], z["cat"]) == ("role", True, "out_practicing")
    k = st.filter((pl.col("key") == "k") & (pl.col("week") == 7)).row(0, named=True)
    assert (k["role"], k["returning"]) == ("depth", False)  # has games: this season counts


def test_fresh_designation_beats_a_stale_reserve_listing():
    st = pl.DataFrame({"key": ["z", "i", "p"], "team": ["X"] * 3, "cat": ["reserve"] * 3,
                       "role": ["role"] * 3})  # fmt: skip
    players = pl.DataFrame({
        "key": ["z", "i", "p"], "injury_status": ["Questionable", "Out", "Out"],
        "practice": [None, None, FP], "p_play": [0.44, 0.0, 0.0],
        "roster_status": ["RES", "ACT", "ACT"], "designation": ["Questionable", "IR", "Out"],
    })  # fmt: skip
    got = dict(ros._with_current(st, players).select("key", "cat").iter_rows())
    assert got == {"z": "questionable", "i": "reserve", "p": "out_practicing"}


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


def test_weekly_projection_estimates_a_missing_line_and_flags_it():
    """Oct 27: DAL vs ARI had no line on Tuesday; with implied totals null, both teams
    were projected for ~0 touchdowns. Missing lines now come from team ratings."""
    from ffa.service.projections import with_lines

    rows = []
    posted_lines = [("AAA", "BBB", 27.0, 20.0), ("BBB", "CCC", 24.0, 21.0),
                    ("CCC", "AAA", 19.0, 26.0)]  # fmt: skip
    for wk, (h, a, imp_h, imp_a) in enumerate(posted_lines, start=1):
        for team, opp, home, imp, oimp in ((h, a, True, imp_h, imp_a), (a, h, False, imp_a, imp_h)):
            rows.append({"game_id": f"g{wk}", "season": 2026, "week": wk, "game_type": "REG",
                         "team": team, "opponent": opp, "is_home": home, "implied": imp,
                         "opp_implied": oimp, "spread_for": imp - oimp,
                         "total_line": imp + oimp})  # fmt: skip
    for team, opp, home in (("AAA", "BBB", True), ("BBB", "AAA", False)):  # week 4: no line yet
        rows.append({"game_id": "g4", "season": 2026, "week": 4, "game_type": "REG",
                     "team": team, "opponent": opp, "is_home": home, "implied": None,
                     "opp_implied": None, "spread_for": None, "total_line": None})  # fmt: skip
    tg = pl.DataFrame(rows)
    out = with_lines(tg, 2026, 4).filter(pl.col("week") == 4).sort("team")
    assert out["implied"].is_not_null().all() and out["lines_from_ratings"].all()
    a, b = out.row(0, named=True), out.row(1, named=True)
    assert a["implied"] > b["implied"]  # AAA scored more in the posted lines, and is home
    assert a["total_line"] == pytest.approx(b["total_line"])
    assert a["spread_for"] == pytest.approx(-b["spread_for"])
    posted = with_lines(tg, 2026, 1).filter(pl.col("week") == 1)
    assert not posted["lines_from_ratings"].any()  # posted lines are kept


def test_this_weeks_rest_of_season_row_comes_from_the_live_weekly_projection():
    """Oct 27: the snapshot was from Monday (before the week rolled over), so week 8 read
    Javonte Williams 16.6 and Maye's chance to play 0.94 while my_team said 8.8 and 1.0."""
    snap = pl.DataFrame({
        "league": ["lg"] * 4, "key": ["a", "a", "b", "b"], "name": ["A", "A", "B", "B"],
        "position": ["RB"] * 4, "team": ["DAL"] * 4, "target_week": [8, 9, 8, 9],
        "opponent": ["ARI", "IND", "ARI", "IND"], "play_probability": [0.94, 0.9, 0.5, 0.5],
        "expected_points": [16.6, 13.1, 5.0, 4.0], "proj_if_active": [16.6, 14.5, 10.0, 8.0],
        "sd": [7.0, 7.0, 5.0, 5.0], "playoff": [False] * 4, "season": [2026] * 4,
        "as_of_week": [7] * 4,
    })  # fmt: skip
    live = pl.DataFrame({
        "league": ["lg", "other"], "key": ["a", "a"], "expected_points": [8.8, 9.0],
        "proj_if_active": [8.8, 9.0], "p10_if_active": [3.5, 3.5], "p90_if_active": [14.6, 15.0],
        "play_probability": [1.0, 1.0], "sd": [4.0, 4.0], "name": ["A", "A"],
        "position": ["RB", "RB"], "team": ["DAL", "DAL"], "opponent": ["ARI", "ARI"],
        "lines_from_ratings": [True, True], "target_week": [8, 8],
    })  # fmt: skip
    out = ros.with_live_week(snap, live, 8)
    a8 = out.filter((pl.col("key") == "a") & (pl.col("target_week") == 8)).row(0, named=True)
    assert (a8["expected_points"], a8["play_probability"]) == (8.8, 1.0)
    assert a8["as_of_week"] == 7 and a8["league"] == "lg"
    # Later weeks keep the snapshot; b isn't in the live run this week (ruled out of the
    # projection, e.g. traded or released): his week-8 row goes, week 9 stays.
    assert out.filter(pl.col("target_week") == 9)["expected_points"].to_list() == [13.1, 4.0]
    assert out.filter(pl.col("key") == "b")["target_week"].to_list() == [9]
    assert ros.with_live_week(snap, pl.DataFrame(), 8).equals(snap)  # no live run: as is
