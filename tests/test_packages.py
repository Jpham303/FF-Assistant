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
    assert P.perceived(_seen("a", 10.0, 15.0)) == 0.6 * 10 + 0.4 * 15  # 12.0
    assert P.perceived(_seen("a", 10.0)) == 10.0  # no last-3: season rate
    assert P.perceived(_seen("a", None, per_game=8.0)) == 8.0  # no games: the projection
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

    # They give WR4 (11) for RB3 (5): looks -6 a game -> -1.8 a week of score.
    pk = Package("them", ("RB3",), ("WR4",), 0.0, 0.0)
    sc, parts = P.acceptance(lg, pk, theirs=0.5)
    assert round(parts["how_it_looks"], 2) == -1.8 and round(sc, 2) == -1.3
    # The same, but they have to cut QB2 (18 a game): another -5.4.
    pk.cuts = ("QB2",)
    sc2, parts2 = P.acceptance(lg, pk, theirs=0.5)
    assert round(parts2["cuts"], 2) == -5.4 and round(sc2, 2) == -6.7
    assert parts2["cut_players"] == ["QB2"]
    # A cut that's only the baseline's pickup costs nothing: they never had him.
    assert round(P.acceptance(lg, pk, 0.5, frozenset({"QB2"}))[0], 2) == -1.3


def test_generate_on_hand_built_rosters():
    lg = _league()
    for p in lg.players.values():
        vals = [v[0] for v in p.weekly.values()]
        p.per_game = p.ppg = p.last3 = sum(vals) / len(vals)
    out = P.generate(lg, "me")
    offers = out["them"]
    assert offers, "RB3 for WR3 (+8 a week for you, even for them) at least"
    grades = [o.grade for o in offers]
    order = [*P.GRADES, P.NEAR_MISS]
    assert grades == [g for g in order if g in grades]  # Likely first, one per grade
    for o in offers:
        assert o.mine >= P.OFFER_MIN_GAIN - P.NEAR_GAIN
        assert P.grade(o.mine, o.score)[0] == o.grade
