"""Trade package generator: offers for every opponent, graded by how likely they are to
be accepted (pure: no database, no network).

For each opponent the target search's pool of packages (ffa.trade.search.team_pool,
widened so long shots are found) is graded, and the best package in each grade where
you gain is kept: some opponents get four, some none.

Grades (a scored grade, not a probability: no data exists on offers turned down):

- Likely     their lineup gains at least LIKELY_THEIR_GAIN a week, and it looks even or
             better for them on what managers see
- Fair       their lineup gains (or breaks even): both sides gain
- Stretch    they lose up to STRETCH_MAX_LOSS a week, but it looks even or better for
             them; you gain at least STRETCH_MY_GAIN (buying low / selling high)
- Long shot  they lose more (up to LONG_SHOT_MAX_LOSS) and only what they see sells it
             (it looks even or better for them); you gain at least LONG_SHOT_MY_GAIN
- none       anything else: a trade that's worse for them and looks it isn't suggested

"What managers see" (perceived points a game): points per game so far in the league's
scoring and over the last 3 games, discounted for this week's injury status, with a
small premium for an early pick in the league's draft (name value). A side's players are
weighted best first (STUD_WEIGHTS: depth is worth less than one better player), so a
2-for-1 consolidation reads as managers read it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ffa.trade.core import League, Player
from ffa.trade.search import MY_MIN_GAIN, Package, team_pool

GRADES = ("Likely", "Fair", "Stretch", "Long shot")
LIKELY_THEIR_GAIN = 0.3  # points a week
STRETCH_MAX_LOSS = 1.0
STRETCH_MY_GAIN = 0.5
LONG_SHOT_MAX_LOSS = 3.0  # beyond this a package isn't suggested at all
LONG_SHOT_MY_GAIN = 1.0
LOOKS_EVEN = 0.10  # perceived values within 10% read as even
SEASON_WEIGHT, LAST3_WEIGHT = 0.6, 0.4
INJURY_FACTOR = {"Out": 0.3, "Doubtful": 0.6, "Questionable": 0.9}
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
    for rounds, bump in DRAFT_PREMIUM:
        if p.draft_round is not None and p.draft_round <= rounds:
            return base * (1 + bump)
    return base


def side_value(lg: League, keys: tuple[str, ...] | list[str]) -> float:
    vals = sorted((perceived(lg.players[k]) for k in keys if k in lg.players), reverse=True)
    return sum(v * w for v, w in zip(vals, STUD_WEIGHTS, strict=False))


def looks_for_them(lg: League, give: tuple[str, ...], get: tuple[str, ...]) -> float:
    """How the trade looks to the other manager: (what he gets - what he gives) / what
    he gives, on perceived value. >= -LOOKS_EVEN reads as even or better for him."""
    gets, gives = side_value(lg, give), side_value(lg, get)  # his view: he gets your give
    return (gets - gives) / max(gives, 1.0)


def grade(mine: float, theirs: float, looks: float) -> str | None:
    """mine / theirs: lineup change a week; looks: looks_for_them."""
    if mine < MY_MIN_GAIN:
        return None
    even = looks >= -LOOKS_EVEN
    if theirs >= LIKELY_THEIR_GAIN and even:
        return "Likely"
    if theirs >= 0:
        return "Fair"
    if theirs >= -STRETCH_MAX_LOSS and even and mine >= STRETCH_MY_GAIN:
        return "Stretch"
    if theirs >= -LONG_SHOT_MAX_LOSS and even and mine >= LONG_SHOT_MY_GAIN:
        return "Long shot"  # only what he sees (a name, a hot streak) sells it
    return None


@dataclass
class Offer:
    grade: str
    package: Package
    mine: float  # a week
    theirs: float
    looks: float
    reasons_you: list[str] = field(default_factory=list)
    reasons_them: list[str] = field(default_factory=list)


def generate(lg: League, me: str, *, playoff_weighting: bool = False,
             partners: list[str] | None = None) -> dict[str, list[Offer]]:  # fmt: skip
    """team -> its best offer in each grade (Likely first), you gaining in every one."""
    n = max(len(lg.weeks), 1)
    out: dict[str, list[Offer]] = {}
    for tid in lg.teams:
        if tid == me or (partners and tid not in partners):
            continue
        best: dict[str, Offer] = {}
        for p in team_pool(lg, me, tid, playoff_weighting=playoff_weighting,
                           max_loss=LONG_SHOT_MAX_LOSS):  # fmt: skip
            mine, theirs = p.mine / n, p.theirs / n
            lk = looks_for_them(lg, p.give, p.get)
            g = grade(mine, theirs, lk)
            if g is None:
                continue
            cur = best.get(g)
            if cur is None or (mine, theirs) > (cur.mine, cur.theirs):
                best[g] = Offer(g, p, mine, theirs, lk)
        out[tid] = [best[g] for g in GRADES if g in best]
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
