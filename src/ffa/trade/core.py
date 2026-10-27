"""Trade evaluation on rest-of-season projections (pure: no database, no network).

A trade's value to a team is the change in its best weekly lineup, summed over the
remaining weeks (fantasy playoff weeks optionally counted double), after both rosters
change, measured from the roster after the waiver moves the team could already make
(with_waivers, applied first by the service, so free pickups aren't credited to a trade):
the side receiving more players drops its least valuable ones, and the side
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
    ros_rank: int | None = None  # positional rank by per_game
    nfl_team: str | None = None
    per_game: float | None = None  # projected points per game if active, rest of season
    play: dict[int, float] = field(default_factory=dict)  # week -> chance to play (default 1)
    # What managers see (how a trade looks to them): this season in league scoring,
    # where he was drafted in this league, this week's designation.
    games: int = 0
    ppg: float | None = None
    last3: float | None = None
    draft_pick: int | None = None
    draft_round: int | None = None
    injury: str | None = None

    def ros(self, weeks: list[int]) -> float:
        return sum(self.weekly.get(w, (0.0, 0.0))[0] for w in weeks)


@dataclass
class Team:
    id: str
    name: str
    roster: list[str]  # player keys, IR included (taxi excluded)
    size_limit: int  # starters + bench; players in IR slots don't count
    ir: set[str] = field(default_factory=set)  # in an IR slot now
    ir_slots: int = 0

    def counted(self, roster: list[str], ir: set[str] | None = None) -> int:
        """Players taking a roster spot (IR-slot players don't)."""
        ir = self.ir if ir is None else ir
        return sum(1 for k in roster if k not in ir)


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
    return total + _insurance(lg, keys, w, starters, used), var, starters


def _insurance(lg: League, keys: list[str], w: int, starters: list[str | None],
               used: set[str]) -> float:  # fmt: skip
    """What the bench adds: a starter who doesn't play is replaced by the best eligible
    bench player, so each backup is worth (chance the starter misses) x his expected
    points. Expected points alone value every backup at 0 (Oct 27: GeumSon 'cut' Lamar
    Jackson, their backup QB, to take a third defense). Riskiest starter gets first pick
    of the bench; each backup covers one slot."""
    misses = []
    for i, k in enumerate(starters):
        if k is None:
            continue
        miss = 1.0 - lg.players[k].play.get(w, 1.0)
        if miss > 0:
            misses.append((miss, i))
    if not misses:
        return 0.0
    backups = {k: lg.players[k].weekly.get(w, (0.0, 0.0))[0] for k in keys
               if k not in used and k in lg.players}  # fmt: skip
    out = 0.0
    for miss, i in sorted(misses, reverse=True):
        elig = SLOT_ELIGIBILITY.get(lg.slots[i], frozenset())
        cand = [k for k, e in backups.items() if e > 0 and lg.players[k].position in elig]
        if not cand:
            continue
        b = max(cand, key=backups.get)
        out += miss * backups.pop(b)
    return out


def value(lg: League, keys: list[str], playoff_weighting: bool = False) -> float:
    return sum(lg.weight(w, playoff_weighting) * lineup(lg, keys, w)[0] for w in lg.weeks)


# ---------------------------------------------------------------- roster balance


@dataclass
class Move:
    key: str
    name: str
    position: str
    points: float  # rest-of-season lineup points added (+) or given up (-)


def after_trade(team: Team, gave: tuple[str, ...] | list[str], got: tuple[str, ...] | list[str],
                giver: Team) -> tuple[list[str], set[str]]:  # fmt: skip
    """A team's roster and IR slots right after a trade: what it gave leaves (IR slots
    included); a player who was in the other team's IR slot goes into an open IR slot
    here, or onto the bench (taking a roster spot) when none is open."""
    roster = [k for k in team.roster if k not in gave] + list(got)
    ir = {k for k in team.ir if k not in gave}
    for k in got:
        if k in giver.ir and len(ir) < team.ir_slots:
            ir.add(k)
    return roster, ir


def _ros(lg: League, k: str) -> float:
    p = lg.players.get(k)
    return p.ros(lg.weeks) if p else 0.0


def balance(lg: League, team: Team, after: list[str], ir_after: set[str],
            fa_taken: set[str], playoff_weighting: bool = False,
            ) -> tuple[list[str], list[Move], list[Move]]:  # fmt: skip
    """The roster after the trade made legal: drops (over the limit) or best free agents
    (into spots the trade freed). IR-slot players don't take a spot and aren't dropped.
    Returns (roster, drops, adds)."""
    roster, drops, adds = list(after), [], []
    limit = max(team.size_limit, team.counted(team.roster))
    open_before = limit - team.counted(team.roster)
    while team.counted(roster, ir_after) > limit:  # drop whoever costs the lineup least
        base = value(lg, roster, playoff_weighting)
        cost = {k: base - value(lg, [x for x in roster if x != k], playoff_weighting)
                for k in roster if k not in ir_after}  # fmt: skip
        k = min(cost, key=lambda x: (round(cost[x], 6), _ros(lg, x)))  # ties: fewest points
        roster.remove(k)
        p = lg.players.get(k)
        drops.append(Move(k, p.name if p else k, p.position if p else "?", -cost[k]))
    freed = (limit - team.counted(roster, ir_after)) - open_before
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


def _fa_candidates(lg: League, taken: set[str], per_position: int = FA_PER_POSITION) -> list[str]:
    by_pos: dict[str, list[str]] = {}
    for k in lg.free_agents:
        if k in taken or k not in lg.players:
            continue
        by_pos.setdefault(lg.players[k].position, []).append(k)
    out = []
    for keys in by_pos.values():
        keys.sort(key=lambda k: -lg.players[k].ros(lg.weeks))
        out += keys[:per_position]
    return out


# ---------------------------------------------------------------- free waiver moves

WAIVER_MIN = 0.3  # points a week a waiver move must add to count as one a manager would make
WAIVER_MOVES = 3  # per team
WAIVER_FA = 3  # free agents per position tried as a pickup
WAIVER_DROPS = 3  # least valuable players tried as the drop


@dataclass
class Waiver:
    team: str
    add: Move
    drop: Move | None  # None: an open roster spot


def with_waivers(lg: League, playoff_weighting: bool = False, protect: frozenset[str] = frozenset(),
                 order: list[str] | None = None) -> tuple[League, list[Waiver]]:  # fmt: skip
    """The league after every team makes the waiver moves that already help it: add the
    best free agent (into an open spot, or dropping its least valuable player), up to
    WAIVER_MOVES each, teams in `order` (worst record first, like waiver priority). A
    player is dropped only for a free agent who outscores him the rest of the season,
    and a dropped player stays out of the pool (no chains of hypothetical moves). A
    trade is then credited only for what it adds beyond moves either manager could make
    without it. `protect`: players named in the trade, never dropped."""
    pw = playoff_weighting
    n = max(len(lg.weeks), 1)
    teams = {tid: Team(t.id, t.name, list(t.roster), t.size_limit, set(t.ir), t.ir_slots)
             for tid, t in lg.teams.items()}  # fmt: skip
    out = League(lg.slots, lg.weeks, lg.playoff_weeks, lg.players, list(lg.free_agents), teams)
    moves: list[Waiver] = []
    taken: set[str] = set()
    for tid in order or list(teams):
        team = teams[tid]
        for _ in range(WAIVER_MOVES):
            roster = team.roster
            base = value(out, roster, pw)
            fas = _fa_candidates(out, taken, WAIVER_FA)
            if team.counted(roster) < max(team.size_limit, 1):
                drops: list[str | None] = [None]
            else:
                cost = {k: base - value(out, [x for x in roster if x != k], pw)
                        for k in roster if k not in protect and k not in team.ir}  # fmt: skip
                drops = sorted(cost, key=lambda x: (round(cost[x], 6), _ros(lg, x)))[:WAIVER_DROPS]
            best, best_gain = None, WAIVER_MIN * n
            for d in drops:
                kept = [x for x in roster if x != d]
                d_ros = lg.players[d].ros(lg.weeks) if d in lg.players else 0.0
                for f in fas:
                    # Managers drop a player for a free agent who outscores him the rest
                    # of the way, not a 10-point-a-game back for a streaming defense.
                    if d is not None and lg.players[f].ros(lg.weeks) <= d_ros:
                        continue
                    g = value(out, [*kept, f], pw) - base
                    if g > best_gain:
                        best, best_gain = (d, f), g
            if best is None:
                break
            d, f = best
            team.roster = [x for x in roster if x != d] + [f]
            taken.add(f)
            pf = lg.players[f]
            add = Move(f, pf.name, pf.position, best_gain)
            drop = None
            if d is not None:
                pd = lg.players.get(d)
                # Not returned to the pool: no other team is assumed to pick him up
                # (Oct 27: one team's model drop of Monangai became another's add).
                drop = Move(d, pd.name if pd else d, pd.position if pd else "?", 0.0)
            moves.append(Waiver(tid, add, drop))
    out.free_agents = [k for k in out.free_agents if k not in taken]
    return out, moves


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


def _side(lg: League, team: Team, before: list[str], after: list[str], ir_after: set[str],
          fa_taken: set[str], playoff_weighting: bool) -> Side:  # fmt: skip
    after, drops, adds = balance(lg, team, after, ir_after, fa_taken, playoff_weighting)
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
    my_after, my_ir = after_trade(mine, give, get, theirs)
    their_after, their_ir = after_trade(theirs, get, give, mine)
    you = _side(lg, mine, mine.roster, my_after, my_ir, fa_taken, playoff_weighting)
    them = _side(lg, theirs, theirs.roster, their_after, their_ir, fa_taken, playoff_weighting)
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
                f"{p.ros_rank} per game the rest of the way ({direction}): his rank may be stale "
                "(role change or injury)."
            )  # fmt: skip
    return out


def _card(lg: League, p: Player, side: str) -> dict:
    return {"side": side, "key": p.key, "name": p.name, "position": p.position,
            "season_rank": p.season_rank, "season_points": p.season_points,
            "ros_rank": p.ros_rank, "ros_per_game": p.per_game,
            "ros_points": round(p.ros(lg.weeks), 1)}  # fmt: skip


def _verdict(looks: dict, per_week: float) -> tuple[str, str]:
    """Your lineup change decides, at every position (Justin, Oct 27): rank so far and
    raw projected totals are how the trade looks to the other manager, given as
    context (acceptance is Phase 4's grades). Same-position trades used to go by
    season-to-date rank first, which hedged to 'close call' when a stale rank and the
    lineup disagreed."""
    seen = f"; {looks.get('basis', 'projected points')} (how it looks to them)"
    if looks.get("favors") in ("you", "them") and looks.get("same_position"):
        seen += f", favoring {looks['favors']}"
    if per_week >= CLOSE:
        return "accept", f"your lineup {per_week:+.1f} a week{seen}"
    if per_week <= -CLOSE:
        return "decline", f"your lineup {per_week:+.1f} a week{seen}"
    return "close call", f"your lineup changes {per_week:+.1f} a week, under {CLOSE}{seen}"
