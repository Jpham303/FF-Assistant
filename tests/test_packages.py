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


def test_grades():
    g = P.grade
    assert g(0.05, 1.0, 0.0) is None  # you don't gain: never suggested
    assert g(1.0, 0.5, 0.0) == "Likely"  # they gain, looks even
    assert g(1.0, 0.5, -0.3) == "Fair"  # they gain, but it looks worse to them
    assert g(1.0, 0.0, 0.0) == "Fair"  # break even
    assert g(0.8, -0.6, 0.05) == "Stretch"  # small loss, looks fine to them
    assert g(0.3, -0.6, 0.05) is None  # ... but you must gain at least 0.5
    assert g(2.0, -2.5, 0.2) == "Long shot"  # bigger loss, a name sells it
    assert g(2.0, -2.5, -0.3) is None  # bigger loss and it looks it: not suggested
    assert g(2.0, -3.5, 0.5) is None  # past LONG_SHOT_MAX_LOSS


def test_generate_on_hand_built_rosters():
    lg = _league()
    for p in lg.players.values():
        vals = [v[0] for v in p.weekly.values()]
        p.per_game = p.ppg = p.last3 = sum(vals) / len(vals)
    out = P.generate(lg, "me")
    offers = out["them"]
    assert offers, "RB3 for WR3 (+8 a week for you, even for them) at least"
    grades = [o.grade for o in offers]
    assert grades == [g for g in P.GRADES if g in grades]  # Likely first, one per grade
    for o in offers:
        assert o.mine >= P.MY_MIN_GAIN
        assert P.grade(o.mine, o.theirs, o.looks) == o.grade
