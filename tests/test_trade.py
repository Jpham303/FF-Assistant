"""Trade evaluation on hand-built rosters: every number here can be checked by hand."""

from __future__ import annotations

import pytest

from ffa.trade import core
from ffa.trade.core import League, Player, Team

WEEKS = [8, 9, 10]


def _p(key, pos, pts, rank=None, ros_rank=None, weeks=WEEKS):
    return Player(key, key, pos, {w: (pts, 4.0) for w in weeks}, season_points=pts * 7,
                  season_rank=rank, ros_rank=ros_rank)  # fmt: skip


def _league():
    players = [
        _p("QB1", "QB", 20), _p("RB1", "RB", 15), _p("RB2", "RB", 10, rank=24, ros_rank=24),
        _p("RB3", "RB", 5, rank=40, ros_rank=40), _p("WR1", "WR", 12), _p("WR2", "WR", 6),
        _p("QB2", "QB", 18), _p("RB4", "RB", 8, rank=30, ros_rank=30), _p("WR3", "WR", 14),
        _p("WR4", "WR", 11), _p("WR5", "WR", 9), _p("WR6", "WR", 3),
        _p("RBFA", "RB", 7), _p("WRFA", "WR", 6),
        _p("RBBYE", "RB", 9, weeks=[8, 10]),  # free agent on bye in week 9
    ]  # fmt: skip
    return League(
        slots=["QB", "RB", "RB", "WR", "FLEX"], weeks=WEEKS, playoff_weeks={10},
        players={p.key: p for p in players}, free_agents=["RBFA", "WRFA", "RBBYE"],
        teams={
            "me": Team("me", "Mine", ["QB1", "RB1", "RB2", "RB3", "WR1", "WR2"], 6),
            "them": Team("them", "Theirs", ["QB2", "RB4", "WR3", "WR4", "WR5", "WR6"], 6),
        },
    )  # fmt: skip


def test_lineup_fills_restrictive_slots_first():
    lg = _league()
    total, var, starters = core.lineup(lg, lg.teams["me"].roster, 8)
    assert total == 20 + 15 + 10 + 12 + 6  # flex: WR2 (6) over RB3 (5)
    assert starters == ["QB1", "RB1", "RB2", "WR1", "WR2"] and var == 5 * 16


def test_one_for_one_both_sides_by_hand():
    lg = _league()
    r = core.evaluate(lg, "me", "them", give=["RB3"], get=["WR4"])
    # You: flex goes from WR2 (6) to WR4 (11): +5 a week. Them: RB3 fills their empty
    # second RB slot (+5), WR5 (9) replaces WR4 (11) at flex (-2): +3 a week.
    assert r["you"]["per_week"] == 5.0 and r["you"]["ros_change"] == 15.0
    assert r["them"]["per_week"] == 3.0
    assert r["you"]["playoff_weeks_change"] == 5.0
    assert not r["you"]["drops"] and not r["you"]["adds"]


def test_two_for_one_names_the_free_agent_and_drops_on_the_other_side():
    lg = _league()
    r = core.evaluate(lg, "me", "them", give=["RB2", "WR2"], get=["WR3"])
    add = r["you"]["adds"][0]
    # Best free agent for your open spot: RBBYE (9 in weeks 8 and 10, bye in 9) adds
    # 4 + 0 + 4 over RB3 (5) = 8; RBFA (7) adds 2 + 2 + 2 = 6.
    assert add["key"] == "RBBYE" and add["points"] == 8.0
    assert [d["key"] for d in r["them"]["drops"]] == ["WR6"]  # 7 for 6 spots: least costly
    # Lineup by week: before 63; after, week 8 and 10: 20+15+9+14+12 = 70, week 9 (RBBYE
    # on bye): 20+15+5+14+12 = 66.
    assert r["you"]["by_week"] == {8: 7.0, 9: 3.0, 10: 7.0}


def test_playoff_toggle_doubles_playoff_weeks():
    lg = _league()
    off = core.evaluate(lg, "me", "them", ["RB3"], ["WR4"], playoff_weighting=False)
    on = core.evaluate(lg, "me", "them", ["RB3"], ["WR4"], playoff_weighting=True)
    assert on["you"]["ros_change"] == off["you"]["ros_change"] + 5.0  # week 10 counted twice


def test_same_position_uses_rank_and_flags_stale_ranks():
    lg = _league()
    lg.players["RB4"].ros_rank = 45  # ranked RB30 so far, projects RB45
    r = core.evaluate(lg, "me", "them", give=["RB2"], get=["RB4"])
    looks = r["how_it_looks"]
    assert looks["same_position"] and looks["favors"] == "them"  # RB30 for your RB24
    assert any("RB30 so far but projects RB45" in f for f in looks["flags"])
    assert r["verdict"] == "decline"


def test_cross_position_uses_projected_points_and_disagreement_is_a_close_call():
    lg = _league()
    r = core.evaluate(lg, "me", "them", give=["RB1"], get=["WR4"])
    assert not r["how_it_looks"]["same_position"]
    assert r["how_it_looks"]["favors"] == "them"  # 45 vs 33 projected
    assert r["verdict"] in ("decline", "close call") and 0 <= r["p_ahead"] <= 1


def test_players_must_be_on_the_right_rosters():
    with pytest.raises(ValueError):
        core.evaluate(_league(), "me", "them", give=["WR3"], get=["WR4"])


def test_playoff_odds_follow_strength_and_standings():
    from ffa.trade import playoffs

    records = {"a": (5, 700.0), "b": (5, 690.0), "c": (2, 600.0), "d": (2, 590.0)}
    weekly = {t: {w: (mu, 100.0) for w in (8, 9, 10)}
              for t, mu in {"a": 110, "b": 90, "c": 110, "d": 90}.items()}  # fmt: skip
    sched = {8: [("a", "b"), ("c", "d")], 9: [("a", "c"), ("b", "d")], 10: [("a", "d"), ("b", "c")]}
    o = playoffs.odds(records, weekly, sched, playoff_teams=2, n=4000)
    assert o["a"] > 0.95 and o["d"] < 0.1  # 3 games left can't close a 3-win gap often
    assert sum(o.values()) == pytest.approx(2.0, abs=0.01)  # two spots, every season
    stronger = dict(weekly, d={w: (130, 100.0) for w in (8, 9, 10)})
    assert playoffs.odds(records, stronger, sched, 2, n=4000)["d"] > o["d"]
