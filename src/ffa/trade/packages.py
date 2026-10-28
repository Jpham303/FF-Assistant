"""Trade package generator: offers for every opponent, graded by how likely they are to
be accepted (pure: no database, no network).

For each opponent the target search's pool of packages (ffa.trade.search.team_pool,
widened so long shots are found) is graded, and the best package in each grade where
you gain is kept: some opponents get four, some none. A package that misses a grade or
a minimum by a hair is kept as that team's near miss instead of vanishing.

Acceptance score, in points a week (the grade's inputs, weighed as the spec lists):

    their lineup change
    + LOOKS_WEIGHT x (what they get - what they give, perceived points a game)
    - CUT_WEIGHT x (perceived points a game of anyone they'd have to cut)

Grades by score (a scored grade, not a probability: no data exists on offers turned
down): Likely >= 1.0, Fair >= 0, Stretch >= -1.0, Long shot >= -3.0. Every offer gains
you at least OFFER_MIN_GAIN a week (LONG_SHOT_MY_GAIN for long shots).

"What managers see" (perceived points a game): points per game so far in the league's
scoring and over the last 3 games, discounted for injury status, kickers and defenses
at a tenth (streamed), with a small premium for an early pick in the league's draft
(name value). A side's players are weighted best first (STUD_WEIGHTS: depth is worth
less than one better player), so a 2-for-1 consolidation reads as managers read it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ffa.trade.core import CLOSE, STREAMABLE, STREAMABLE_SEEN, League, Player, trade_value
from ffa.trade.search import Package, _distinct, team_pool

GRADES = ("Likely", "Fair", "Stretch", "Long shot")
NEAR_MISS = "Near miss"
THRESHOLDS = (("Likely", 1.0), ("Fair", 0.0), ("Stretch", -1.0), ("Long shot", -3.0))
LOOKS_WEIGHT = 0.3  # a perceived point a game they come out ahead = 0.3 a week of score
CUT_WEIGHT = 0.3  # cutting a real player hurts: Lamar Jackson (~18 a game) costs ~5.4
# An offer must gain you what a trade check calls a gain (core.CLOSE, 0.5 a week): under
# it the 'gain' is within the model's noise, e.g. one week of flex shuffling (Oct 27).
OFFER_MIN_GAIN = CLOSE
LONG_SHOT_MY_GAIN = 1.0
VALUE_RATIO = 0.25  # each point a week of trade value you give up needs 0.25 of gain
PER_GRADE = 2  # offers per grade per team, each a different core deal (Oct 27: one slot
# per grade hid Jameson + Barkley for Hubbard behind Barkley for Jones)
NEAR_SCORE = 0.25  # within this of the next grade (or of Long shot): said, or kept
NEAR_GAIN = 0.1  # within this under your minimum gain: kept as a near miss
LOOKS_EVEN = 0.10  # for 'reads as': perceived values within 10% read as even
SEASON_WEIGHT, LAST3_WEIGHT = 0.6, 0.4
INJURY_FACTOR = {
    "Out": 0.3,
    "Doubtful": 0.6,
    "Questionable": 0.9,
    "IR": 0.3,
    "PUP": 0.3,
    "Sus": 0.3,
    "NA": 0.3,
}  # reserve lists read as Out
DRAFT_PREMIUM = ((3, 0.10), (6, 0.05))  # (through round, premium)
STUD_WEIGHTS = (1.0, 0.3, 0.15, 0.1)  # a starter isn't sold for three backups
SELL_HIGH = 1.2  # so far vs projected, per game: a 20% gap is worth saying


def perceived(p: Player) -> float:
    """Points a game a manager would credit him with, from what he can see."""
    if p.ppg is None:  # no games yet (rookie, injured all year): the projection is all
        base = p.per_game or 0.0  # anyone has to go on
    else:
        base = SEASON_WEIGHT * p.ppg + LAST3_WEIGHT * (p.last3 if p.last3 is not None
                                                        else p.ppg)  # fmt: skip
    base *= INJURY_FACTOR.get(p.injury or "", 1.0)
    if p.position in STREAMABLE:  # kickers and defenses are streamed: little trade value
        base *= STREAMABLE_SEEN
    for rounds, bump in DRAFT_PREMIUM:
        if p.draft_round is not None and p.draft_round <= rounds:
            return base * (1 + bump)
    return base


def cut_value(lg: League, p: Player) -> float:
    """What cutting him costs a manager, a game: what he looks like, but never more than
    he's projected to give them a week from here. A player who isn't going to play
    isn't a painful cut, whatever his name (Oct 27: Lamar Jackson, 0.6 a week ahead,
    charged as ~6 a game)."""
    ahead = p.ros(lg.weeks) / max(len(lg.weeks), 1)
    return min(perceived(p), ahead)


def side_value(lg: League, keys: tuple[str, ...] | list[str]) -> float:
    vals = sorted((perceived(lg.players[k]) for k in keys if k in lg.players), reverse=True)
    return sum(v * w for v, w in zip(vals, STUD_WEIGHTS, strict=False))


def looks_for_them(lg: League, give: tuple[str, ...], get: tuple[str, ...]) -> float:
    """How the trade looks to the other manager: (what he gets - what he gives) / what
    he gives, on perceived value. >= -LOOKS_EVEN reads as even or better for him."""
    gets, gives = side_value(lg, give), side_value(lg, get)  # his view: he gets your give
    return (gets - gives) / max(gives, 1.0)


def acceptance(lg: League, p: Package, theirs: float,
               untradeable: frozenset[str] = frozenset()) -> tuple[float, dict]:  # fmt: skip
    """Their acceptance score (points a week) and its parts. theirs: their lineup change a
    week. Players they'd 'cut' who are only the waiver baseline's pickups cost nothing:
    they never had them."""
    looks_pts = side_value(lg, p.give) - side_value(lg, p.get)
    cut = [k for k in p.cuts if k not in untradeable and k in lg.players]
    cut_pts = sum(cut_value(lg, lg.players[k]) for k in cut)
    score = theirs + LOOKS_WEIGHT * looks_pts - CUT_WEIGHT * cut_pts
    return score, {"their_lineup": round(theirs, 2),
                   "how_it_looks": round(LOOKS_WEIGHT * looks_pts, 2),
                   "cuts": round(-CUT_WEIGHT * cut_pts, 2),
                   "cut_players": [lg.players[k].name for k in cut]}  # fmt: skip


def min_gain(value_lost: float) -> float:
    """What an offer must gain you a week: OFFER_MIN_GAIN, or VALUE_RATIO x the trade
    value you give away, whichever is more (Oct 27: Maye + Charbonnet for Hubbard gave
    up ~11 a week of value for +1.29 and graded Stretch)."""
    return max(OFFER_MIN_GAIN, VALUE_RATIO * value_lost)


def grade(mine: float, score: float, need: float = OFFER_MIN_GAIN,
          ) -> tuple[str | None, str | None]:  # fmt: skip
    """(grade, note): the grade, or NEAR_MISS when a minimum is missed by a hair, or
    None; the note says how far from the next grade up (or the minimum) it is.
    need: your minimum gain a week (min_gain)."""
    if mine < need:
        if mine >= need - NEAR_GAIN and score >= THRESHOLDS[-1][1]:
            why = ("the minimum" if need == OFFER_MIN_GAIN
                   else "what the trade value you give up needs")  # fmt: skip
            return NEAR_MISS, (f"your gain {mine:.2f} a week, {need - mine:.2f} under {why} "
                               f"({need:.2f})")  # fmt: skip
        return None, None
    for k, (g, cut) in enumerate(THRESHOLDS):
        if score < cut:
            continue
        if g == "Long shot" and mine < LONG_SHOT_MY_GAIN:
            return NEAR_MISS, (f"a long shot only if you gained {LONG_SHOT_MY_GAIN} a week "
                               f"(you gain {mine:.2f})")  # fmt: skip
        if k > 0 and THRESHOLDS[k - 1][1] - score <= NEAR_SCORE:
            up, line = THRESHOLDS[k - 1]
            return g, f"{line - score:.2f} short of {up}"
        return g, None
    if score >= THRESHOLDS[-1][1] - NEAR_SCORE and mine >= LONG_SHOT_MY_GAIN:
        return NEAR_MISS, f"{THRESHOLDS[-1][1] - score:.2f} short of Long shot"
    return None, None


@dataclass
class Offer:
    grade: str
    package: Package
    mine: float  # a week
    theirs: float
    looks: float  # looks_for_them, for 'reads as'
    score: float = 0.0  # acceptance score, points a week
    parts: dict = field(default_factory=dict)
    note: str | None = None  # near a grade line: how far
    reasons_you: list[str] = field(default_factory=list)
    reasons_them: list[str] = field(default_factory=list)


def generate(lg: League, me: str, *, playoff_weighting: bool = False,
             partners: list[str] | None = None,
             untradeable: frozenset[str] = frozenset(), give: tuple[str, ...] = (),
             position: str | None = None) -> dict[str, list[Offer]]:  # fmt: skip
    """team -> its best offer in each grade (Likely first), you gaining in every one,
    then its best near miss if one exists. `give` / `position` narrow the search the way
    the target search asks (your named players only; targets at one position)."""
    n = max(len(lg.weeks), 1)
    out: dict[str, list[Offer]] = {}
    for tid in lg.teams:
        if tid == me or (partners and tid not in partners):
            continue
        cands: dict[str, list[Offer]] = {}
        for p in team_pool(lg, me, tid, playoff_weighting=playoff_weighting,
                           max_loss=-THRESHOLDS[-1][1] + NEAR_SCORE,
                           untradeable=untradeable, give=tuple(give),
                           position=position):  # fmt: skip
            mine, theirs = p.mine / n, p.theirs / n
            sc, parts = acceptance(lg, p, theirs, untradeable)
            lost = trade_value(lg, p.give) - trade_value(lg, p.get)
            g, note = grade(mine, sc, min_gain(lost))
            if g is None:
                continue
            cands.setdefault(g, []).append(Offer(
                g, p, mine, theirs, looks_for_them(lg, p.give, p.get), sc,
                parts | {"your_value_lost_per_week": round(lost, 2),
                         "your_gain_needed": round(min_gain(lost), 2)}, note))  # fmt: skip
        out[tid] = []
        for g in (*GRADES, NEAR_MISS):
            ranked = sorted(cands.get(g, []), key=lambda o: (-o.mine, -o.score))
            keep = _distinct(lg, [o.package for o in ranked])
            picked = [o for o in ranked if o.package in keep]
            out[tid] += picked[: 1 if g == NEAR_MISS else PER_GRADE]
    return out


def reasons(lg: League, offer: Offer, evaluation: dict, team_needs: dict[str, float],
            ) -> tuple[list[str], list[str]]:  # fmt: skip
    """Two or three reasons a side, each with the number it rests on."""
    p = offer.package
    you = list(evaluation["you"]["reasons"][:2])
    for k in p.give:
        pl_ = lg.players[k]
        if pl_.ppg and pl_.per_game and pl_.ppg >= SELL_HIGH * pl_.per_game:
            you.append(f"Sell high: {pl_.name} averages {pl_.ppg:.1f} so far but projects "
                       f"{pl_.per_game:.1f} a game the rest of the way")  # fmt: skip
            break
    for k in p.get:
        pl_ = lg.players[k]
        if pl_.ppg is not None and pl_.per_game and pl_.per_game >= SELL_HIGH * max(pl_.ppg, 0.1):
            you.append(f"Buy low: {pl_.name} averages {pl_.ppg:.1f} so far but projects "
                       f"{pl_.per_game:.1f} a game the rest of the way")  # fmt: skip
            break
    them = list(evaluation["them"]["reasons"][:2])
    filled = [(team_needs.get(lg.players[k].position, 0.0), lg.players[k].position)
              for k in p.give]  # fmt: skip
    gap, pos = max(filled, default=(0.0, ""))
    if gap > 0:
        them.append(f"Fills their {pos} need: their {pos} starters are {gap:.1f} a week "
                    "below the league's middle team")  # fmt: skip
    if len(p.get) > len(p.give):
        them.append(f"Consolidation: {len(p.get)}-for-{len(p.give)} opens a roster spot "
                    "for them")  # fmt: skip
    return you[:3], them[:3]
