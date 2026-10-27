"""Target search: which trades raise your lineup without hurting theirs much (pure).

"Who should I target if I trade X?" or "I need a RB, what's realistic?":

1. Needs. Each team's points at each position from its weekly best lineup, against the
   league's middle team at that position. Below the middle is a need.
2. Candidates. Players you could get: on another roster and starting for you in at least
   one remaining week, ranked by what they add to your lineup. Players you could give:
   starting for that team in at least one week, ranked by what they add there. Your
   named players are the only ones you give when you name any.
3. Packages. Every 1-for-1 between the two candidate lists, then the promising ones
   grow by one player at a time (a sweetener from you, or a second player from them for
   consolidation) up to MAX_SIDE a side, while it makes the deal better for both.
4. Score both sides with the trade-evaluation rule (drops, free-agent adds), keep the
   ones where you gain and they lose at most THEIR_MAX_LOSS a week, best first: one per
   core deal (the best player each way), and one per team in a league-wide search so
   the list isn't one manager's roster.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ffa.trade.core import League, after_trade, balance, lineup, value

MAX_SIDE = 4
CANDIDATES = 6  # per side, per opponent, for the 1-for-1 seed
BEAM = 4  # packages per opponent kept at each growth step
MY_MIN_GAIN = 0.1  # points a week: at least a slight edge
THEIR_MAX_LOSS = 1.0  # points a week: they can lose this much and still say yes
PER_OPPONENT = 1  # packages per team in a league-wide search (all of them for a named team)
TOP = 5


@dataclass
class Package:
    partner: str
    give: tuple[str, ...]
    get: tuple[str, ...]
    mine: float  # your lineup change, rest of season (weighted if the toggle is on)
    theirs: float
    tags: list[str] = field(default_factory=list)

    def key(self) -> tuple:
        return (self.partner, tuple(sorted(self.give)), tuple(sorted(self.get)))


def position_points(lg: League, roster: list[str]) -> dict[str, float]:
    """Rest-of-season lineup points from each position's starters."""
    out: dict[str, float] = {}
    for w in lg.weeks:
        for k in lineup(lg, roster, w)[2]:
            if k:
                p = lg.players[k]
                out[p.position] = out.get(p.position, 0.0) + p.weekly[w][0]
    return out


def needs(lg: League) -> dict[str, dict[str, float]]:
    """team -> position -> points a week below (+) or above (-) the league's middle team
    at that position, from the starters' rest-of-season points."""
    pts = {t: position_points(lg, team.roster) for t, team in lg.teams.items()}
    positions = sorted({pos for d in pts.values() for pos in d})
    n = max(len(lg.weeks), 1)
    out: dict[str, dict[str, float]] = {t: {} for t in pts}
    for pos in positions:
        vals = sorted(d.get(pos, 0.0) for d in pts.values())
        mid = vals[len(vals) // 2]
        for t, d in pts.items():
            out[t][pos] = round((mid - d.get(pos, 0.0)) / n, 2)
    return out


def score(lg: League, me: str, partner: str, give: tuple[str, ...], get: tuple[str, ...],
          playoff_weighting: bool = False) -> tuple[float, float]:  # fmt: skip
    """(your lineup change, theirs) over the rest of the season, rosters balanced."""
    mine, theirs = lg.teams[me], lg.teams[partner]
    taken: set[str] = set()
    out = []
    for team, gave, got, other in ((mine, give, get, theirs), (theirs, get, give, mine)):
        after, ir = after_trade(team, gave, got, other)
        after, _, _ = balance(lg, team, after, ir, taken, playoff_weighting)
        out.append(value(lg, after, playoff_weighting) - value(lg, team.roster, playoff_weighting))
    return out[0], out[1]


def _gain_if_added(lg: League, roster: list[str], k: str, base: float, pw: bool) -> float:
    return value(lg, [*roster, k], pw) - base


def search(lg: League, me: str, *, give: list[str] | None = None,
           position: str | None = None, partners: list[str] | None = None,
           playoff_weighting: bool = False, top: int = TOP) -> list[Package]:  # fmt: skip
    """The best packages across the league (or `partners`), you-first."""
    pw = playoff_weighting
    n = max(len(lg.weeks), 1)
    mine = lg.teams[me]
    fixed = tuple(give or ())
    bad = [k for k in fixed if k not in mine.roster]
    if bad:
        raise ValueError(f"Not on your roster: {bad}")
    my_base = value(lg, mine.roster, pw)
    found: list[Package] = []
    for tid, team in lg.teams.items():
        if tid == me or (partners and tid not in partners):
            continue
        their_base = value(lg, team.roster, pw)
        # Players you could get: they'd start for you at least once.
        gets = {k: _gain_if_added(lg, mine.roster, k, my_base, pw) for k in team.roster
                if position is None or lg.players[k].position == position}  # fmt: skip
        gets = sorted((k for k, g in gets.items() if g > 0), key=lambda k: -gets[k])
        if not gets:
            continue
        if fixed:
            gives = list(fixed)
        else:
            gv = {k: _gain_if_added(lg, team.roster, k, their_base, pw) for k in mine.roster}
            gives = sorted((k for k, g in gv.items() if g > 0), key=lambda k: -gv[k])
        seeds: list[Package] = []
        for g in gets[:CANDIDATES]:
            offers = [fixed] if fixed else [(x,) for x in gives[:CANDIDATES]]
            for gv_ in offers:
                m, t = score(lg, me, tid, gv_, (g,), pw)
                if m > 0:
                    seeds.append(Package(tid, gv_, (g,), m, t))
        pool = {p.key(): p for p in seeds}
        beam = _best(seeds, n)
        # Grow: one more player on either side while it helps both (sweetener /
        # consolidation). Named offers stay fixed on your side.
        for _ in range(2 * (MAX_SIDE - 1)):
            grown: list[Package] = []
            for p in beam:
                adds_give = [] if fixed else [k for k in gives if k not in p.give]
                adds_get = [k for k in gets[:CANDIDATES] if k not in p.get]
                for k in adds_give[:CANDIDATES] if len(p.give) < MAX_SIDE else []:
                    q = _try(lg, me, tid, p.give + (k,), p.get, pw, pool)
                    if q and q.theirs > p.theirs and q.mine > 0:
                        q.tags = sorted({*p.tags, "sweetener"})
                        grown.append(q)
                for k in adds_get if len(p.get) < MAX_SIDE else []:
                    q = _try(lg, me, tid, p.give, p.get + (k,), pw, pool)
                    if q and q.mine > p.mine and q.theirs >= min(p.theirs, -THEIR_MAX_LOSS * n):
                        q.tags = sorted({*p.tags, "consolidation" if len(q.get) > len(q.give)
                                         else "adds a player"})  # fmt: skip
                        grown.append(q)
            if not grown:
                break
            beam = _best(grown, n)
        keep = [p for p in pool.values()
                if p.mine >= MY_MIN_GAIN * n and p.theirs >= -THEIR_MAX_LOSS * n]  # fmt: skip
        keep.sort(key=lambda p: (-p.mine, -p.theirs))
        found += _distinct(lg, keep)[: top if partners else PER_OPPONENT]
    found.sort(key=lambda p: (-p.mine, -p.theirs))
    return found[:top]


def _try(lg, me, tid, give, get, pw, pool) -> Package | None:
    p = Package(tid, give, get, 0.0, 0.0)
    if p.key() in pool:
        return None
    p.mine, p.theirs = score(lg, me, tid, give, get, pw)
    pool[p.key()] = p
    return p


def _best(ps: list[Package], n: int) -> list[Package]:
    """The packages most worth growing: you gain, and the least painful for them."""
    ok = [p for p in ps if p.mine > 0]
    ok.sort(key=lambda p: -(min(p.mine, p.theirs + THEIR_MAX_LOSS * n) + 0.1 * p.mine))
    return ok[:BEAM]


def _distinct(lg: League, ps: list[Package]) -> list[Package]:
    """One package per core deal: the best player you get for the best player you give.
    Variants with different throw-ins are the same offer."""
    out: list[Package] = []
    seen: set[tuple[str, str]] = set()
    for p in ps:
        best = lambda ks: max(ks, key=lambda k: lg.players[k].ros(lg.weeks))  # noqa: E731
        core = (best(p.give), best(p.get))
        if core in seen:
            continue
        seen.add(core)
        out.append(p)
    return out
