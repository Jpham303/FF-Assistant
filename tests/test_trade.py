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


def test_same_position_rank_is_context_and_the_lineup_decides():
    lg = _league()
    lg.players["RB4"].ros_rank = 45  # ranked RB30 so far, projects RB45
    r = core.evaluate(lg, "me", "them", give=["RB2"], get=["RB4"])
    looks = r["how_it_looks"]
    assert looks["same_position"] and looks["favors"] == "them"  # RB30 for your RB24
    assert any("RB30 so far but projects RB45" in f for f in looks["flags"])
    assert r["verdict"] == "decline"  # RB4 (8) for RB2 (10): your lineup -2 a week
    # The case that changed (Oct 27, option A): a better rank so far (RB5 for your RB24)
    # used to make this a "close call"; your lineup still loses 2 a week, so: decline.
    lg.players["RB4"].season_rank = 5
    r = core.evaluate(lg, "me", "them", give=["RB2"], get=["RB4"])
    assert r["how_it_looks"]["favors"] == "you" and r["verdict"] == "decline"
    assert "favoring you" in r["verdict_reason"]


def test_cross_position_looks_use_projected_points():
    lg = _league()
    r = core.evaluate(lg, "me", "them", give=["RB1"], get=["WR4"])
    assert not r["how_it_looks"]["same_position"]
    assert r["how_it_looks"]["favors"] == "them"  # 45 vs 33 projected
    # RB1 (15) for WR4 (11): RB2 and RB3 start at RB, WR4 at flex: -15 + 5 + 11 - 6 = -5
    assert r["verdict"] == "decline" and r["you"]["per_week"] == -5.0
    assert 0 <= r["p_ahead"] <= 1


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


def test_bracket_order_and_round_weeks():
    from ffa.trade import playoffs

    assert playoffs.bracket_order(4) == [1, 4, 2, 3]
    assert playoffs.bracket_order(8) == [1, 8, 4, 5, 2, 7, 3, 6]  # 6 teams: 1, 2 get byes
    assert playoffs.round_weeks([15, 16, 17], 3, 0) == [[15], [16], [17]]
    assert playoffs.round_weeks([15, 16], 1, 1) == [[15, 16]]  # two-week final
    assert playoffs.round_weeks([15, 16, 17, 18], 2, 2) == [[15, 16], [17, 18]]
    assert playoffs.round_weeks([15, 16], 3, 0) is None  # not every round projected


def _six(mu: dict[str, float], wins: dict[str, float]) -> tuple:
    teams = sorted(mu)
    records = {t: (wins[t], 600.0) for t in teams}
    weekly = {t: {w: (mu[t], 400.0) for w in (12, 13, 14, 15, 16, 17)} for t in teams}
    pairs = [(teams[i], teams[i + 1]) for i in range(0, len(teams), 2)]
    return records, weekly, {12: pairs, 13: pairs, 14: pairs}


def test_title_odds_from_the_bracket():
    """Oct 28: title odds alongside playoff odds for every trade."""
    from ffa.trade import playoffs

    teams = [chr(97 + i) for i in range(8)]
    # Equal strength; records set the seeds (3 games can't close a 4-win gap).
    wins = {t: 40 - 4 * i for i, t in enumerate(teams)}
    rec, wk, sch = _six({t: 100.0 for t in teams}, wins)
    o = playoffs.season(rec, wk, sch, 6, [15, 16, 17], 0, n=6000)
    assert sum(v["title"] for v in o.values()) == pytest.approx(1.0, abs=0.005)
    assert o["g"]["title"] == 0 and o["h"]["title"] == 0  # not in the playoffs, no title
    # Byes: the top two seeds play two rounds, the others three.
    assert o["a"]["title"] == pytest.approx(0.25, abs=0.03)
    assert o["c"]["title"] == pytest.approx(0.125, abs=0.03)
    # A team far stronger than the rest wins nearly every time it's in.
    strong = {t: 100.0 for t in teams} | {"d": 200.0}
    rec2, wk2, sch2 = _six(strong, wins)
    o2 = playoffs.season(rec2, wk2, sch2, 6, [15, 16, 17], 0, n=4000)
    assert o2["d"]["title"] > 0.95
    # Two-week rounds need more weeks than are projected: no title odds, playoffs still.
    o3 = playoffs.season(rec, wk, sch, 6, [15, 16, 17], 2, n=1000)
    assert o3["a"]["title"] is None and o3["a"]["playoffs"] > 0.9


# ---------------------------------------------------------------- target search

from ffa.trade import search  # noqa: E402


def test_needs_are_points_a_week_against_the_middle_team():
    lg = _league()
    n = search.needs(lg)
    # RB starters: you RB1+RB2 (+RB3 never: WR2 takes flex) = 25 a week; them RB4 = 8.
    # Two teams: the "middle" is the higher one (index 1 of 2), so you're at 0, they trail 17.
    assert n["me"]["RB"] == 0.0 and n["them"]["RB"] == 17.0
    assert n["me"]["WR"] > 0  # their WRs are better


def test_search_finds_the_hand_checked_swap_and_scores_like_evaluate():
    lg = _league()
    found = search.search(lg, "me")
    assert found, "the RB3-for-WR4 swap (+5 / +3 a week) should be found"
    n = len(WEEKS)
    for p in found:
        assert p.mine >= search.MY_MIN_GAIN * n and p.theirs >= -search.THEIR_MAX_LOSS * n
        r = core.evaluate(lg, "me", p.partner, list(p.give), list(p.get))
        assert r["you"]["ros_change"] == round(p.mine, 1)
        assert r["them"]["ros_change"] == round(p.theirs, 1)
    assert found[0].mine >= 15.0  # at least as good as RB3 for WR4


def test_search_respects_a_named_offer_and_a_position():
    lg = _league()
    for p in search.search(lg, "me", give=["RB3"]):
        assert p.give == ("RB3",)
    for p in search.search(lg, "me", position="WR"):
        assert all(lg.players[k].position == "WR" for k in p.get)
    assert search.search(lg, "me", position="TE") == []  # nobody has one
    with pytest.raises(ValueError, match="Not on your roster"):
        search.search(lg, "me", give=["WR3"])


def test_search_never_returns_a_trade_that_guts_them():
    lg = _league()
    # Their QB2 would start for nobody on your side (QB1 is better), and taking WR3+WR4
    # for scraps costs them far more than THEIR_MAX_LOSS a week: never suggested.
    for p in search.search(lg, "me"):
        assert not {"WR3", "WR4"} <= set(p.get) or p.theirs >= -search.THEIR_MAX_LOSS * len(WEEKS)


def test_waiver_baseline_makes_only_the_moves_that_help_and_protects_named_players():
    lg = _league()
    out, moves = core.with_waivers(lg)
    got = [(m.team, m.add.key, round(m.add.points, 1), m.drop.key if m.drop else None)
           for m in moves]  # fmt: skip
    # You: RBBYE (9 in weeks 8 and 10, bye in 9) takes the flex from WR2 (6) in those two
    # weeks: +3 +3 = 6, dropping RB3 (5, never starts; 15 rest of season < RBBYE's 18).
    # Then RBFA (7) beats WR2 (6) at flex in week 9 only: +1 over 3 weeks = 0.33 a week,
    # just over WAIVER_MIN (0.3); RBFA's 21 > WR2's 18. Your drops don't return to the
    # pool (Oct 27: one team's model drop became another's add), so they make no move.
    assert got == [("me", "RBBYE", 6.0, "RB3"), ("me", "RBFA", 1.0, "WR2")]
    assert set(out.free_agents) == {"WRFA"}
    assert lg.teams["me"].roster == ["QB1", "RB1", "RB2", "RB3", "WR1", "WR2"]  # original kept
    _, moves2 = core.with_waivers(lg, protect=frozenset({"RB3", "WR2"}))
    assert all(m.drop is None or m.drop.key not in ("RB3", "WR2") for m in moves2)


def test_waiver_baseline_never_drops_a_better_player_for_a_lesser_free_agent():
    lg = _league()
    lg.players["QB3"] = _p("QB3", "QB", 18)  # a backup: never starts, 54 rest of season
    lg.players["WRFA2"] = _p("WRFA2", "WR", 7)  # 21: would start at flex over WR2 (6)
    lg.free_agents = ["WRFA2"]
    me = lg.teams["me"]
    me.roster.append("QB3")
    me.size_limit = 7
    _, moves = core.with_waivers(lg)
    # QB3 and RB3 both cost the lineup nothing; only RB3 (15) is outscored by WRFA2 (21).
    assert [(m.add.key, m.drop.key) for m in moves if m.team == "me"] == [("WRFA2", "RB3")]


def test_ir_players_are_tradeable_and_take_no_roster_spot():
    lg = _league()
    me, them = lg.teams["me"], lg.teams["them"]
    me.ir, me.ir_slots = {"RB3"}, 1  # RB3 sits in your IR slot: 5 players count, 1 open
    # Them, no IR slot: RB3 lands on their bench. They gave WR6 and got RB3: still 6
    # counted, no drop, no free agent.
    r = core.evaluate(lg, "me", "them", give=["RB3"], get=["WR6"])
    assert not r["them"]["drops"] and not r["them"]["adds"]
    # You: RB3 leaves IR, WR6 takes the spot that was already open: no drop, and no
    # free agent either (trading an IR player frees no roster spot).
    assert not r["you"]["drops"] and not r["you"]["adds"]
    # Them with an open IR slot: RB3 goes there, so WR6 leaving frees a bench spot and
    # they add the best free agent.
    them.ir_slots = 1
    r = core.evaluate(lg, "me", "them", give=["RB3"], get=["WR6"])
    assert [a["key"] for a in r["them"]["adds"]] == ["RBBYE"]


def test_an_ir_player_returning_counts_in_the_weeks_he_plays():
    lg = _league()
    lg.players["RBIR"] = _p("RBIR", "RB", 20, weeks=[10])  # back in week 10 only
    me = lg.teams["me"]
    me.roster.append("RBIR")
    me.ir, me.ir_slots = {"RBIR"}, 1
    assert me.counted(me.roster) == 6  # still within the limit
    # Week 10: RBIR (20) starts over RB2 (10), who takes the flex from WR2 (6): +14.
    assert core.lineup(lg, me.roster, 10)[0] - core.lineup(lg, me.roster, 9)[0] == 14


def test_across_positions_the_lineup_change_decides_the_verdict():
    """Oct 27: Aaron Jones for Kirk Cousins gained 2.45 a week but read 'close call'
    because Cousins projects more raw points."""
    looks = {"same_position": False, "favors": "them",
             "basis": "rest-of-season projected points: you get 107.6, give 114.4"}  # fmt: skip
    verdict, why = core._verdict(looks, 2.45)
    assert verdict == "accept" and why.startswith("your lineup +2.5 a week")
    assert "how it looks to them" in why
    assert core._verdict(looks | {"favors": "you"}, -1.0)[0] == "decline"
    assert core._verdict(looks, 0.2)[0] == "close call"


def test_a_backup_is_worth_the_chance_a_starter_misses():
    """Oct 27: bench depth was worth 0, so GeumSon 'cut' Lamar Jackson to take a third
    defense."""
    lg = _league()
    me = lg.teams["me"]
    lg.players["QBB"] = _p("QBB", "QB", 18)  # backup QB
    me.roster.append("QBB")
    base = core.lineup(lg, me.roster, 8)[0]  # everyone certain to play: backup adds 0
    assert base == core.lineup(lg, [k for k in me.roster if k != "QBB"], 8)[0]
    lg.players["QB1"].play = {8: 0.8}  # starter 80% to play (20 expected, as before)
    lg.changed()
    # Week 8: QB1 misses 20% of the time; QBB (18) then plays: +0.2 x 18 = 3.6.
    assert round(core.lineup(lg, me.roster, 8)[0] - base, 6) == 3.6
    # Two risky starters, one backup: the riskier gets him (RB1 30% to miss vs RB2 10%).
    lg.players["QB1"].play = {}
    lg.players["RB1"].play, lg.players["RB2"].play = {8: 0.7}, {8: 0.9}
    lg.changed()
    # RB3 (5) is the bench RB; WR2 (6) is the flex starter: RB3 backs up RB1: 0.3 x 5.
    # RB2's miss has no backup left: RBs only (QBB can't play RB).
    assert round(core.lineup(lg, me.roster, 8)[0] - base, 6) == 1.5


def test_ties_in_drop_cost_cut_the_player_with_fewer_points():
    lg = _league()
    them = lg.teams["them"]
    lg.players["DEFA"] = _p("DEFA", "DEF", 4)
    lg.players["QBB"] = _p("QBB", "QB", 15)  # their backup QB: never starts, all play
    them.roster.append("QBB")
    them.size_limit = 7
    # They receive a defense they can't start (no DEF slot). The defense, their backup QB
    # and WR6 all cost the lineup 0: the cut is the one with the fewest points, WR6 (9),
    # never the backup QB (45).
    after, drops, _ = core.balance(lg, them, [*them.roster, "DEFA"], set(), set())
    assert [d.key for d in drops] == ["WR6"]


def test_kickers_and_defenses_are_worth_only_what_they_beat_the_waiver_wire_by():
    """Oct 27: managers stream K and D/ST, so trading for one is worth what he beats the
    week's best free agent by, not his points."""
    lg = _league()
    lg.slots = [*lg.slots, "DEF"]
    for k, pts in (("DEF_MINE", 5), ("DEF_THEIRS", 9), ("DEF_FA", 7), ("DEF_FA2", 6)):
        lg.players[k] = _p(k, "DEF", pts)
    lg.teams["me"].roster.append("DEF_MINE")
    lg.teams["them"].roster.append("DEF_THEIRS")
    lg.free_agents += ["DEF_FA", "DEF_FA2"]
    core.with_streams(lg)
    assert lg.stream[("DEF", 8)] == 7  # the best free agent each week
    me = lg.teams["me"].roster
    # Your DEF (5) is below the stream (7): the slot counts 7, as if you streamed into
    # his roster spot. With no defense rostered there's no spot to stream into: empty.
    no_def = [k for k in me if k != "DEF_MINE"]
    assert core.lineup(lg, me, 8)[0] == core.lineup(lg, no_def, 8)[0] + 7
    # Getting their DEF (9) is worth 9 - 7 = 2 a week to you, not 9 - 5 = 4.
    assert core.lineup(lg, [*me, "DEF_THEIRS"], 8)[0] - core.lineup(lg, me, 8)[0] == 2


def test_managers_credit_kickers_and_defenses_with_little_trade_value():
    from ffa.trade import packages as P

    d = Player("d", "d", "DEF", {}, per_game=9.0)
    d.ppg, d.last3 = 9.0, 9.0
    r = Player("r", "r", "RB", {}, per_game=9.0)
    r.ppg, r.last3 = 9.0, 9.0
    assert round(P.perceived(d), 6) == 0.9 and P.perceived(r) == 9.0


def test_no_package_is_only_kickers_or_defenses_on_a_side():
    lg = _league()
    lg.slots = [*lg.slots, "K"]
    for k, pts in (("K_MINE", 5), ("K_THEIRS", 9)):
        lg.players[k] = _p(k, "K", pts)
    lg.teams["me"].roster.append("K_MINE")
    lg.teams["them"].roster.append("K_THEIRS")
    for p in search.team_pool(lg, "me", "them", max_loss=5.0):
        assert any(lg.players[k].position != "K" for k in p.give)
        assert any(lg.players[k].position != "K" for k in p.get)


def test_your_only_kicker_is_not_cut_for_free_when_streaming_needs_his_spot():
    """Oct 27: Cam Little was dropped at -0.0 in several deals: the stream filled the K
    slot without needing a roster spot."""
    lg = _league()
    lg.slots = [*lg.slots, "K"]
    lg.players["K1"] = _p("K1", "K", 8)
    lg.players["K_FA"] = _p("K_FA", "K", 9)  # the stream beats him every week
    lg.free_agents.append("K_FA")
    me = lg.teams["me"]
    me.roster.append("K1")
    me.size_limit = 7
    core.with_streams(lg)
    after, drops, _ = core.balance(lg, me, [*me.roster, "WR6"], set(), set())
    # Over the limit by one: WR6 (3) or RB3 (5), both never start, go before the kicker,
    # whose spot is the one you stream into (cutting him costs the whole K slot).
    assert drops and all(d.key != "K1" for d in drops)


def test_a_backups_value_shows_in_the_reasons():
    lg = _league()
    lg.players["WR1"].play = {w: 0.6 for w in WEEKS}  # your WR1 is 40% to miss
    # Adding WRFA (6) as a bench WR: he covers WR1's absences, 0.4 x 6 = 2.4 a week,
    # 7.2 over 3 weeks; shown as a reason (it used to be invisible).
    side = core._side(lg, lg.teams["me"], lg.teams["me"].roster,
                      [*lg.teams["me"].roster, "WRFA"], set(), set(), False)  # fmt: skip
    assert any(x.startswith("WRFA (WR) as a backup") and x.endswith("+7.2 points")
               for x in side.reasons)  # fmt: skip


def test_a_manager_keeps_an_injured_returning_backup_over_a_bench_body():
    """Oct 27: GeumSon 'cut' Lamar Jackson (0.6 to their lineup behind Burrow, 93.7
    points from his return) before anyone else."""
    lg = _league()
    them = lg.teams["them"]
    lg.players["QBHURT"] = Player("QBHURT", "QBHURT", "QB",
                                  {8: (0.0, 0.0), 9: (18.0, 5.0), 10: (18.0, 5.0)})  # fmt: skip
    them.roster.append("QBHURT")  # behind QB2 (18): costs their lineup 0
    them.size_limit = 7
    # One more player arrives: the cut is WR6 (3 a week, never starts: cost 0 + 0.25 x
    # 9 = 2.25), not QBHURT (cost 0 + 0.25 x 36 = 9).
    _, drops, _ = core.balance(lg, them, [*them.roster, "RB3"], set(), set())
    assert [d.key for d in drops] == ["WR6"]


def test_your_trade_value_lost_is_reported():
    lg = _league()
    # QB1 (20 a week) for WR4 (11): you give up 9 a week of value.
    r = core.evaluate(lg, "me", "them", give=["QB1"], get=["WR4"])
    v = r["your_trade_value"]
    assert (v["give_per_week"], v["get_per_week"], v["lost_per_week"]) == (20.0, 11.0, 9.0)
    assert "flag" in v
    # Depth counts less: RB2 (10) + RB3 (5) read as 10 + 0.3 x 5 = 11.5.
    assert core.trade_value(lg, ["RB2", "RB3"]) == 11.5


def test_trade_value_is_over_replacement():
    """Oct 27: raw points made a backup QB with Mayfield on waivers look like 5 a week
    of value."""
    lg = _league()
    lg.players["QBFA"] = _p("QBFA", "QB", 17)  # a free agent QB
    lg.free_agents.append("QBFA")
    core.with_streams(lg)  # sets each position's replacement level
    assert core.over_replacement(lg, "QB1") == 3.0  # 20 - 17 a week
    assert core.over_replacement(lg, "QB2") == 1.0  # 18 - 17
    # RB replacement: RBBYE (9) in weeks 8 and 10, RBFA (7) in week 9 (RBBYE's bye).
    assert core.over_replacement(lg, "RB1") == ((15 - 9) + (15 - 7) + (15 - 9)) / 3
    # A player who never beats the waiver wire is worth 0, never less.
    lg.players["QBBAD"] = _p("QBBAD", "QB", 12)
    assert core.over_replacement(lg, "QBBAD") == 0.0
    r = core.evaluate(lg, "me", "them", give=["QB1"], get=["WR4"])
    assert "value over replacement" in r["your_trade_value"]["basis"]
