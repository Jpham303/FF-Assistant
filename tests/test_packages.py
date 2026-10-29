"""Package generator: grades, what managers see, and offers on hand-built rosters."""

from __future__ import annotations

from ffa.trade import packages as P
from ffa.trade.core import Player
from tests.test_trade import _league


def _seen(key, ppg, last3=None, rnd=None, injury=None, per_game=10.0):
    p = Player(key, key, "RB", {}, per_game=per_game)
    p.ppg, p.last3, p.draft_round, p.injury = ppg, last3, rnd, injury
    return p


def test_what_managers_see():
    # Half what he's done (0.6 x season + 0.4 x last 3 = 12), half his projection (10).
    assert P.perceived(_seen("a", 10.0, 15.0)) == 0.5 * (0.6 * 10 + 0.4 * 15) + 0.5 * 10
    assert P.perceived(_seen("a", 10.0)) == 10.0  # no last-3: season rate
    assert P.perceived(_seen("a", None, per_game=8.0)) == 8.0  # no games: the projection
    # Oct 29: Michael Wilson, 9.1 so far, 6.4 lately, projects 12.1: ~10, not 8.
    assert round(P.perceived(_seen("w", 9.1, 6.4, per_game=12.1)), 2) == 10.06
    assert P.perceived(_seen("a", 10.0, 15.0, per_game=None)) == 12.0  # no projection
    assert P.perceived(_seen("a", 10.0, 10.0, injury="Out")) == 3.0
    assert round(P.perceived(_seen("a", 10.0, 10.0, rnd=2)), 6) == 11.0  # early pick +10%
    assert round(P.perceived(_seen("a", 10.0, 10.0, rnd=5)), 6) == 10.5
    assert P.perceived(_seen("a", 10.0, 10.0, rnd=9)) == 10.0


def test_depth_is_worth_less_than_one_better_player():
    lg = _league()
    for k, v in (("RB1", 15.0), ("RB2", 10.0), ("RB3", 5.0)):
        lg.players[k].ppg, lg.players[k].last3 = v, v
    # RB2 + RB3 read as 10 + 0.3 x 5 = 11.5, not 15: RB1 for both looks worse for the
    # side getting the two (it gets 11.5 for 15).
    assert P.side_value(lg, ("RB2", "RB3")) == 11.5
    assert round(P.looks_for_them(lg, ("RB2", "RB3"), ("RB1",)), 4) == round((11.5 - 15) / 15, 4)


def test_grades_come_from_the_acceptance_score():
    g = P.grade
    assert g(1.0, 1.2) == ("Likely", None)
    assert g(1.0, 0.9) == ("Fair", "0.10 short of Likely")  # near the line: said
    assert g(1.0, 0.3) == ("Fair", None)
    assert g(1.0, -0.5) == ("Stretch", None)
    assert g(1.0, -2.0) == ("Long shot", None)
    assert g(0.7, -2.0)[0] == P.NEAR_MISS  # a long shot needs you +1.0: near miss
    assert g(2.0, -3.1) == (P.NEAR_MISS, "0.10 short of Long shot")
    assert g(2.0, -4.0) == (None, None)
    assert g(0.45, 1.5)[0] == P.NEAR_MISS  # 0.05 under your 0.5 minimum
    assert g(0.2, 1.5) == (None, None)  # well under: noise


def test_how_it_looks_and_cuts_move_the_score():
    """Oct 27: Pierce (IR) for Stevenson graded Fair though it read 0.88 worse for them;
    a 4-for-1 graded Likely though they'd cut Lamar Jackson and DJ Moore."""
    lg = _league()
    for k, v in (("RB1", 15.0), ("RB2", 10.0), ("RB3", 5.0), ("WR4", 11.0), ("QB2", 18.0)):
        lg.players[k].ppg, lg.players[k].last3 = v, v
    from ffa.trade.search import Package

    # They give WR4 (11) for RB3 (5): looks -6 a game -> -1.8 a week of score. Oct 29:
    # and they give up 6 a week of value over replacement (no free agents in this
    # league: replacement 0) -> -3.0. Score 0.5 - 1.8 - 3.0 = -4.3.
    pk = Package("them", ("RB3",), ("WR4",), 0.0, 0.0)
    sc, parts = P.acceptance(lg, pk, theirs=0.5)
    assert round(parts["how_it_looks"], 2) == -1.8 and parts["their_value"] == -3.0
    assert parts["their_value_change_per_week"] == -6.0 and round(sc, 2) == -4.3
    # The same, but they have to cut QB2 (18 a game): another -5.4.
    pk.cuts = ("QB2",)
    sc2, parts2 = P.acceptance(lg, pk, theirs=0.5)
    assert round(parts2["cuts"], 2) == -5.4 and round(sc2, 2) == -9.7
    assert parts2["cut_players"] == ["QB2"]
    # A cut that's only the baseline's pickup costs nothing: they never had him.
    assert round(P.acceptance(lg, pk, 0.5, frozenset({"QB2"}))[0], 2) == -4.3


def test_generate_on_hand_built_rosters():
    lg = _league()
    for p in lg.players.values():
        vals = [v[0] for v in p.weekly.values()]
        p.per_game = p.ppg = p.last3 = sum(vals) / len(vals)
    out = P.generate(lg, "me")
    offers = out["them"]
    assert offers, "RB3 for WR3 (+8 a week for you, even for them) at least"
    grades = [o.grade for o in offers]
    # Oct 29: up to 3 Likely and 3 others (Fair/Stretch/Long shot together) a team.
    assert grades.count("Likely") <= P.PER_GROUP
    assert sum(g in ("Fair", "Stretch", "Long shot") for g in grades) <= P.PER_GROUP
    for o in offers:
        assert o.mine >= P.OFFER_MIN_GAIN - P.NEAR_GAIN
        assert P.grade(o.mine, o.score)[0] == o.grade


def test_a_cut_costs_no_more_than_he_will_give_them():
    """Oct 27: Lamar Jackson (0.6 a week ahead) was charged as a ~6-a-game cut."""
    from ffa.trade.search import Package

    lg = _league()
    lg.players["QB2"].ppg = lg.players["QB2"].last3 = 21.0  # looks like a star
    lg.players["QB2"].weekly = {8: (0.6, 1.0), 9: (0.6, 1.0), 10: (0.6, 1.0)}  # hurt
    assert P.cut_value(lg, lg.players["QB2"]) == 0.6
    pk = Package("them", ("RB3",), ("WR4",), 0.0, 0.0, cuts=("QB2",))
    assert round(P.acceptance(lg, pk, 0.0)[1]["cuts"], 2) == -0.18  # 0.3 x 0.6


def test_giving_away_trade_value_needs_a_bigger_gain():
    """Oct 27: Maye + Charbonnet for Hubbard (~11 a week of value given up) graded
    Stretch on a +1.29 lineup gain."""
    assert P.min_gain(0.0) == P.OFFER_MIN_GAIN  # no value lost: the usual minimum
    assert P.min_gain(11.0) == 2.75  # 0.25 x 11
    assert P.grade(1.29, -0.5, P.min_gain(11.0)) == (None, None)  # far short: never
    g, note = P.grade(2.70, -0.5, P.min_gain(11.0))  # 0.05 short: near miss, says why
    assert g == P.NEAR_MISS and "trade value you give up" in note
    assert P.grade(3.0, -0.5, P.min_gain(11.0))[0] == "Stretch"


def test_generate_narrows_to_named_players_and_a_position():
    """Issue #2: the target search is the generator narrowed to the question."""
    lg = _league()
    for p in lg.players.values():
        vals = [v[0] for v in p.weekly.values()]
        p.per_game = p.ppg = p.last3 = sum(vals) / len(vals)
    wide = [o for offs in P.generate(lg, "me").values() for o in offs]
    gives = {k for o in wide for k in o.package.give}
    give = (sorted(gives)[0],)
    narrowed = [o for offs in P.generate(lg, "me", give=give).values() for o in offs]
    assert all(o.package.give == give for o in narrowed)
    pos = lg.players[wide[0].package.get[0]].position
    by_pos = [o for offs in P.generate(lg, "me", position=pos).values() for o in offs]
    assert by_pos and all(lg.players[k].position == pos for o in by_pos for k in o.package.get)


def test_generated_trades_send_at_most_3_and_get_at_most_2():
    """Oct 29: generated trades are up to 3 players sent, 2 received."""
    from ffa.trade.search import MAX_GET, MAX_GIVE

    assert (MAX_GIVE, MAX_GET) == (3, 2)
    lg = _league()
    for p in lg.players.values():
        vals = [v[0] for v in p.weekly.values()]
        p.per_game = p.ppg = p.last3 = sum(vals) / len(vals)
    for offs in P.generate(lg, "me").values():
        for o in offs:
            assert len(o.package.give) <= 3 and len(o.package.get) <= 2


def test_a_named_trade_can_be_any_size():
    """Oct 29: the 3-for-2 limit is for generated trades only; evaluate_trade takes any."""
    from ffa.trade import core

    lg = _league()
    r = core.evaluate(lg, "me", "them", give=["RB2", "RB3", "WR1", "WR2"],
                      get=["WR3", "WR4", "WR5"])  # fmt: skip
    assert r["verdict"] and "per_week" in r["you"]


def test_best_trades_first_likely_then_fair_by_title_odds():
    """Oct 29: Likely first, then Fair, each by your title-odds gain (your gain a week
    breaking ties); anything else after."""
    items = [
        {"grade": "Fair", "title_odds": {"you": {"change": 0.04}}, "you_per_week": 1.0},
        {"grade": "Likely", "title_odds": {"you": {"change": 0.01}}, "you_per_week": 0.6},
        {"grade": "Stretch", "title_odds": {"you": {"change": 0.09}}, "you_per_week": 3.0},
        {"grade": "Likely", "title_odds": {"you": {"change": 0.02}}, "you_per_week": 0.5},
        {"grade": "Fair", "title_odds": None, "you_per_week": 1.5},
        {"grade": "Fair", "title_odds": {"you": {"change": 0.04}}, "you_per_week": 1.2},
    ]
    got = [(x["grade"], x["you_per_week"]) for x in sorted(items, key=P.item_rank)]
    assert got == [("Likely", 0.5), ("Likely", 0.6), ("Fair", 1.2), ("Fair", 1.0),
                   ("Fair", 1.5), ("Stretch", 3.0)]  # fmt: skip


def test_only_likely_and_fair_are_generated_other_grades_label_named_trades():
    lg = _league()
    for p in lg.players.values():
        vals = [v[0] for v in p.weekly.values()]
        p.per_game = p.ppg = p.last3 = sum(vals) / len(vals)
    for offs in P.generate(lg, "me").values():
        assert all(o.grade in P.GENERATED for o in offs)
    assert [P.label(x) for x in (1.2, 0.0, -0.5, -2.9, -3.1)] == [
        "Likely", "Fair", "Stretch", "Long shot", "Very unlikely"]  # fmt: skip


def test_their_value_loss_is_not_discounted_by_stakes():
    """Oct 29: an all-but-eliminated team (stakes 0.13) still won't give away value:
    Antonio Williams + Jameson Williams for Derrick Henry + Michael Wilson graded Stretch."""
    from ffa.trade.search import Package

    lg = _league()
    for k, v in (("RB3", 5.0), ("WR4", 11.0)):
        lg.players[k].ppg, lg.players[k].last3 = v, v
    pk = Package("them", ("RB3",), ("WR4",), 0.0, 0.0)
    seller, parts = P.acceptance(lg, pk, theirs=-3.0, stakes=0.0)
    assert parts["their_lineup_weighted"] == 0.0 and parts["their_value"] == -3.0
    assert round(seller, 2) == -4.8 and P.grade(2.0, seller)[0] is None  # not an offer
