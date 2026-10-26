"""Trade evaluation on rest-of-season projections (pure: no database, no network).

A trade's value to a team is the change in its best weekly lineup, summed over the
remaining weeks (fantasy playoff weeks optionally counted double), after both rosters
change: the side receiving more players drops its least valuable ones, and the side
receiving fewer fills each freed roster spot with the best free agent (named, with what
he adds). Lineups are filled by expected points (chance to play included), most
restrictive slot first, the same rule as my_team.

How the trade looks (what managers judge by): same position, season-to-date positional
rank decides first and the lineup change second; across positions, rest-of-season
projected points decide. A sharp disagreement between a player's rank so far and his
projected rank (lost role, injury) is flagged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ffa.leagues.base import SLOT_ELIGIBILITY

PLAYOFF_WEIGHT = 2.0  # playoff weeks count this much when the toggle is on
CLOSE = 0.5  # points a week: below this, a lineup change is a wash
RANK_GAP = 10  # positional ranks: rank so far vs projected rank, flagged beyond this
FA_PER_POSITION = 8  # free agents considered per position when filling a spot


@dataclass
class Player:
    key: str
    name: str
    position: str
    weekly: dict[int, tuple[float, float]]  # week -> (expected points, sd)
    season_points: float | None = None  # this season so far, league scoring
    season_rank: int | None = None  # positional rank by season_points
    ros_rank: int | None = None  # positional rank by rest-of-season points
    nfl_team: str | None = None

    def ros(self, weeks: list[int]) -> float:
        return sum(self.weekly.get(w, (0.0, 0.0))[0] for w in weeks)


@dataclass
class Team:
    id: str
    name: str
    roster: list[str]  # player keys (IR and taxi excluded)
    size_limit: int


@dataclass
class League:
    slots: list[str]  # starting slots
    weeks: list[int]  # remaining weeks valued (through the championship)
    playoff_weeks: set[int]
    players: dict[str, Player]
    free_agents: list[str]
    teams: dict[str, Team]

    def weight(self, w: int, playoff_weighting: bool) -> float:
        return PLAYOFF_WEIGHT if playoff_weighting and w in self.playoff_weeks else 1.0


# ---------------------------------------------------------------- lineups


def lineup(lg: League, keys: list[str], w: int) -> tuple[float, float, list[str | None]]:
    """Best lineup for week w: (expected points, variance, starter per slot)."""
    order = sorted(range(len(lg.slots)), key=lambda i: len(SLOT_ELIGIBILITY.get(lg.slots[i], ())))
    used: set[str] = set()
    starters: list[str | None] = [None] * len(lg.slots)
    total = var = 0.0
    for i in order:
        elig = SLOT_ELIGIBILITY.get(lg.slots[i], frozenset())
        best, best_e = None, 0.0
        for k in keys:
            p = lg.players.get(k)
            if p is None or k in used or p.position not in elig:
                continue
            e = p.weekly.get(w, (0.0, 0.0))[0]
            if best is None or e > best_e:
                best, best_e = k, e
        if best is not None and best_e > 0:
            used.add(best)
            starters[i] = best
            total += best_e
            var += lg.players[best].weekly[w][1] ** 2
    return total, var, starters


def value(lg: League, keys: list[str], playoff_weighting: bool = False) -> float:
    return sum(lg.weight(w, playoff_weighting) * lineup(lg, keys, w)[0] for w in lg.weeks)


# ---------------------------------------------------------------- roster balance


@dataclass
class Move:
    key: str
    name: str
    position: str
    points: float  # rest-of-season lineup points added (+) or given up (-)


def balance(lg: League, team: Team, after: list[str], received: int, gave: int,
            fa_taken: set[str], playoff_weighting: bool = False,
            ) -> tuple[list[str], list[Move], list[Move]]:  # fmt: skip
    """The roster after the trade made legal: drops (over the limit) or best free agents
    (into spots the trade freed). Returns (roster, drops, adds)."""
    roster, drops, adds = list(after), [], []
    limit = max(team.size_limit, len(team.roster))
    while len(roster) > limit:  # drop whoever costs the lineup least
        base = value(lg, roster, playoff_weighting)
        cost = {k: base - value(lg, [x for x in roster if x != k], playoff_weighting)
                for k in roster}  # fmt: skip
        k = min(cost, key=cost.get)
        roster.remove(k)
        p = lg.players.get(k)
        drops.append(Move(k, p.name if p else k, p.position if p else "?", -cost[k]))
    freed = min(gave - received, limit - len(roster))
    for _ in range(max(freed, 0)):
        base = value(lg, roster, playoff_weighting)
        pool = _fa_candidates(lg, fa_taken)
        gains = {k: value(lg, [*roster, k], playoff_weighting) - base for k in pool}
        if not gains:
            break
        k = max(gains, key=gains.get)
        if gains[k] <= 0:
            break
        roster.append(k)
        fa_taken.add(k)
        p = lg.players[k]
        adds.append(Move(k, p.name, p.position, gains[k]))
    return roster, drops, adds


def _fa_candidates(lg: League, taken: set[str]) -> list[str]:
    by_pos: dict[str, list[str]] = {}
    for k in lg.free_agents:
        if k in taken or k not in lg.players:
            continue
        by_pos.setdefault(lg.players[k].position, []).append(k)
    out = []
    for keys in by_pos.values():
        keys.sort(key=lambda k: -lg.players[k].ros(lg.weeks))
        out += keys[:FA_PER_POSITION]
    return out


# ---------------------------------------------------------------- evaluation


@dataclass
class Side:
    team: str
    before: float
    after: float
    per_week: dict[int, float]  # lineup change by week
    playoff_change: float
    drops: list[Move] = field(default_factory=list)
    adds: list[Move] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    roster: list[str] = field(default_factory=list)  # after the trade and balancing

    @property
    def change(self) -> float:
        return self.after - self.before


def _side(lg: League, team: Team, before: list[str], after: list[str], received: int,
          gave: int, fa_taken: set[str], playoff_weighting: bool) -> Side:  # fmt: skip
    after, drops, adds = balance(lg, team, after, received, gave, fa_taken, playoff_weighting)
    per_week, reasons_pts = {}, {}
    for w in lg.weeks:
        b, _, sb = lineup(lg, before, w)
        a, _, sa = lineup(lg, after, w)
        per_week[w] = a - b
        for k in set(sa) - set(sb) - {None}:
            reasons_pts.setdefault(k, [0.0, 0])
            reasons_pts[k][0] += lg.players[k].weekly[w][0]
            reasons_pts[k][1] += 1
        for k in set(sb) - set(sa) - {None}:
            reasons_pts.setdefault(k, [0.0, 0])
            reasons_pts[k][0] -= lg.players[k].weekly[w][0]
            reasons_pts[k][1] += 1
    side = Side(team.id, value(lg, before, playoff_weighting), value(lg, after, playoff_weighting),
                per_week,
                sum(per_week[w] for w in lg.weeks if w in lg.playoff_weeks),
                drops, adds, roster=after)  # fmt: skip
    n = len(lg.weeks)
    for k, (pts, weeks) in sorted(reasons_pts.items(), key=lambda kv: -abs(kv[1][0]))[:4]:
        p = lg.players[k]
        verb = "starts" if pts > 0 else "no longer starts"
        side.reasons.append(f"{p.name} ({p.position}) {verb} {weeks} of {n} weeks: "
                            f"{pts:+.1f} points")  # fmt: skip
    return side


def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def evaluate(lg: League, me: str, partner: str, give: list[str], get: list[str],
             playoff_weighting: bool = False) -> dict:  # fmt: skip
    """Both sides of a trade: lineup change by week and in total, roster moves, how it
    looks (rank / projection), a verdict and the reasons."""
    mine, theirs = lg.teams[me], lg.teams[partner]
    missing = [k for k in give if k not in mine.roster] + [k for k in get if k not in theirs.roster]
    if missing:
        raise ValueError(f"Not on the right roster: {missing}")
    fa_taken: set[str] = set()
    my_after = [k for k in mine.roster if k not in give] + list(get)
    their_after = [k for k in theirs.roster if k not in get] + list(give)
    you = _side(lg, mine, mine.roster, my_after, len(get), len(give), fa_taken,
                playoff_weighting)  # fmt: skip
    them = _side(lg, theirs, theirs.roster, their_after, len(give), len(get), fa_taken,
                 playoff_weighting)  # fmt: skip
    looks = how_it_looks(lg, give, get)
    weeks = max(len(lg.weeks), 1)
    per_week = you.change / weeks
    # Uncertainty: the moved players' week-to-week spread over the weeks they start.
    var = 0.0
    for w in lg.weeks:
        for k in (*give, *get):
            p = lg.players.get(k)
            if p and w in p.weekly:
                var += p.weekly[w][1] ** 2
    p_ahead = _phi(you.change / math.sqrt(var)) if var > 0 else (1.0 if you.change > 0 else 0.0)
    verdict, why = _verdict(looks, per_week)
    return {
        "verdict": verdict, "verdict_reason": why,
        "you": _side_dict(you, weeks), "them": _side_dict(them, weeks),
        "how_it_looks": looks,
        "p_ahead": round(p_ahead, 3),
        "weeks": lg.weeks, "playoff_weeks": sorted(lg.playoff_weeks),
        "playoff_weighting": playoff_weighting,
    }  # fmt: skip


def _side_dict(s: Side, weeks: int) -> dict:
    return {
        "team": s.team, "ros_change": round(s.change, 1),
        "per_week": round(s.change / weeks, 2),
        "playoff_weeks_change": round(s.playoff_change, 1),
        "by_week": {w: round(v, 2) for w, v in s.per_week.items()},
        "drops": [m.__dict__ | {"points": round(m.points, 1)} for m in s.drops],
        "adds": [m.__dict__ | {"points": round(m.points, 1)} for m in s.adds],
        "reasons": s.reasons, "roster_after": s.roster,
    }  # fmt: skip


def how_it_looks(lg: League, give: list[str], get: list[str]) -> dict:
    """What managers see: same position -> season-to-date rank; else projected points."""
    gp = [lg.players[k] for k in give if k in lg.players]
    rp = [lg.players[k] for k in get if k in lg.players]
    positions = {p.position for p in gp + rp}
    out: dict = {"same_position": len(positions) == 1,
                 "players": [_card(lg, p, "give") for p in gp] + [_card(lg, p, "get") for p in rp],
                 "flags": []}  # fmt: skip
    if out["same_position"]:
        best = lambda ps: min((p.season_rank for p in ps if p.season_rank), default=None)  # noqa: E731
        bg, br = best(gp), best(rp)
        if bg is None or br is None:
            out["basis"], out["favors"] = "rank unavailable", "even"
        else:
            out["basis"] = f"season-to-date rank: you get {next(iter(positions))}{br}, " \
                           f"give {next(iter(positions))}{bg}"  # fmt: skip
            out["favors"] = "you" if br < bg else "them" if br > bg else "even"
    else:
        g, r = sum(p.ros(lg.weeks) for p in gp), sum(p.ros(lg.weeks) for p in rp)
        out["basis"] = f"rest-of-season projected points: you get {r:.1f}, give {g:.1f}"
        out["favors"] = "you" if r > g * 1.03 else "them" if g > r * 1.03 else "even"
    for p in gp + rp:
        if p.season_rank and p.ros_rank and abs(p.season_rank - p.ros_rank) >= RANK_GAP:
            direction = "lower" if p.ros_rank > p.season_rank else "higher"
            out["flags"].append(
                f"{p.name} is {p.position}{p.season_rank} so far but projects {p.position}"
                f"{p.ros_rank} the rest of the way ({direction}): his rank may be stale "
                "(role change or injury)."
            )  # fmt: skip
    return out


def _card(lg: League, p: Player, side: str) -> dict:
    return {"side": side, "key": p.key, "name": p.name, "position": p.position,
            "season_rank": p.season_rank, "season_points": p.season_points,
            "ros_rank": p.ros_rank, "ros_points": round(p.ros(lg.weeks), 1)}  # fmt: skip


def _verdict(looks: dict, per_week: float) -> tuple[str, str]:
    fav = looks.get("favors", "even")
    gain = per_week >= CLOSE
    loss = per_week <= -CLOSE
    if fav == "you" and not loss:
        return "accept", f"{looks['basis']}; your lineup {per_week:+.1f} a week"
    if fav == "them" and not gain:
        return "decline", f"{looks['basis']}; your lineup {per_week:+.1f} a week"
    if fav == "you":  # value says yes, the lineup says no
        return "close call", (f"{looks['basis']}, but your lineup loses {abs(per_week):.1f} a "
                              "week (depth or position fit)")  # fmt: skip
    if fav == "them":  # value says no, the lineup says yes
        return "close call", (f"{looks['basis']}, but your lineup gains {per_week:.1f} a week "
                              "(it fills a need)")  # fmt: skip
    if gain:
        return (
            "accept",
            f"even on {looks.get('basis', 'value')}; your lineup {per_week:+.1f} a week",
        )
    if loss:
        return (
            "decline",
            f"even on {looks.get('basis', 'value')}; your lineup {per_week:+.1f} a week",
        )
    return "close call", f"even on value and your lineup changes {per_week:+.1f} a week"
