"""Your chance of winning this week's fantasy matchup, with each start/sit candidate.

The opponent and his lineup come from Sleeper's matchups for the week (ffa.service.
gamestate): his set starters, or his best lineup by expected points if none are set.
Your side: slots whose game has kicked off stay as you set them in Sleeper (they're
locked); the open slots get your best remaining lineup with the candidate forced in and
the other candidates benched, the same rule my_team uses. Both sides are summed
simulation by simulation; the simulations share every NFL game, so stacks and players
facing each other are correlated as they are on the field. Final games (kickoff + 4
hours) count with the points Sleeper awarded instead of their projection.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from ffa.db.models import LeagueRow
from ffa.scoring import score
from ffa.scoring.rules import ScoringRules
from ffa.service import gamestate, startsit

STARTING = {"BN", "IR", "TAXI"}  # non-starting slots
TIEBREAK_MIN = 0.01  # P(win) difference needed to override a close call


def win_odds(con, db, league: LeagueRow, season: int, week: int, candidates: list[str],
             rules: ScoringRules, *, matchups: list[dict] | None = None,
             now: datetime | None = None) -> dict:  # fmt: skip
    """{"available": False, "reason": ...} when it can't be computed, else the opponent,
    how his lineup was chosen, and per candidate: p_win and both sides' expected points."""
    now = now or datetime.now(UTC)
    lw = gamestate.league_week(db, league, week, matchups)
    if not lw.available:
        return {"available": False, "reason": lw.reason}
    if not lw.opp_entry:
        return {"available": False, "reason": "No fantasy matchup for your team this week"}
    if lw.opponent is None:
        return {"available": False, "reason": "Opponent's roster isn't synced yet"}
    slots = [s for s in league.lineup_slots if s not in STARTING]
    pool = lambda roster: {e.player_key for e in roster  # noqa: E731
                           if e.player_key and e.slot_type != "ir"}  # fmt: skip
    my_pool, opp_pool = pool(lw.my_roster), pool(lw.opp_roster)
    my_set, opp_set = lw.starters(lw.my_entry, len(slots)), lw.starters(lw.opp_entry, len(slots))
    keys = sorted(my_pool | set(candidates) | opp_pool | {k for k in my_set + opp_set if k})
    w = startsit.load_week(con, season, week, keys)
    if w.sims.height == 0:
        return {"available": False, "reason": f"No projections for week {week}"}
    views = startsit.player_views(w, rules)
    mat = score(w.sims, rules).select("key", "sim", "fpts").pivot(
        on="key", index="sim", values="fpts").sort("sim").fill_null(0)  # fmt: skip
    n = mat.height
    awarded = lw.points()
    status = {k: gamestate.status(w.kickoffs.get(v.team), now) for k, v in views.items()}
    final = {k for k, st in status.items() if st == gamestate.FINAL and k in awarded}

    def column(k: str) -> np.ndarray:
        if k in final:
            return np.full(n, awarded[k])
        return mat[k].to_numpy() if k in mat.columns else np.zeros(n)  # no projection: 0

    def total(lineup: list[str]) -> np.ndarray:
        return sum((column(k) for k in lineup), np.zeros(n))

    if any(opp_set):
        opp_lineup, how = [k for k in opp_set if k], "his lineup as set in Sleeper"
    else:
        best = startsit.best_lineup({k: v for k, v in views.items() if k in opp_pool}, slots)
        opp_lineup, how = [r["key"] for r in best if r["key"]], "his best lineup (none set)"
    opp_tot = total(opp_lineup)
    mine_views = {k: v for k, v in views.items() if k in my_pool | set(candidates)}
    out = []
    for c in candidates:
        v = views.get(c)
        if v is None or v.play_probability < startsit.OUT_THRESHOLD:
            out.append({"key": c, "name": v.name if v else c, "p_win": None,
                        "note": "not playing"})  # fmt: skip
            continue
        if status.get(c) in gamestate.STARTED:
            out.append({"key": c, "name": v.name, "p_win": None,
                        "note": "game started: his lineup slot is locked"})  # fmt: skip
            continue
        lineup = [r["key"] for r in startsit.locked_lineup(
            mine_views, slots, my_set, status, awarded, force=c,
            exclude=frozenset(set(candidates) - {c})) if r["key"]]  # fmt: skip
        if c not in lineup:
            out.append({"key": c, "name": v.name, "p_win": None,
                        "note": "no open lineup slot he is eligible for"})  # fmt: skip
            continue
        mine_tot = total(lineup)
        p = float(((mine_tot > opp_tot) + 0.5 * (mine_tot == opp_tot)).mean())
        out.append({"key": c, "name": v.name, "p_win": round(p, 3),
                    "my_expected": round(float(mine_tot.mean()), 1)})  # fmt: skip
    teams = {k: v.team for k, v in views.items()}
    return {
        "available": True,
        "week": week,
        "opponent": lw.opponent.name,
        "opponent_lineup": how,
        "opponent_expected": round(float(opp_tot.mean()), 1),
        "final_games_counted": sorted({teams[k] for k in final if k in teams}),
        "candidates": out,
        "unmatched_opponent_starters": sum(
            1
            for s in lw.opp_entry.get("starters") or []
            if s not in ("0", None) and s not in lw.to_key
        ),  # fmt: skip
    }


def apply_tiebreak(c: startsit.Comparison, odds: dict) -> startsit.Comparison:
    """Attach the odds; on a close call only, start the candidate with the higher chance
    of winning the matchup (by at least TIEBREAK_MIN), saying why: floor vs ceiling."""
    c.matchup = odds
    if not odds.get("available"):
        if odds.get("reason"):
            c.notes.append(f"Matchup odds unavailable: {odds['reason']}.")
        return c
    pw = {x["key"]: x["p_win"] for x in odds["candidates"] if x.get("p_win") is not None}
    views = {v.key: v for v in c.players}
    rec = c.recommendation
    if rec in pw:
        line = ", ".join(f"{views[k].name} {p * 100:.0f}%" for k, p in
                         sorted(pw.items(), key=lambda kv: -kv[1]))  # fmt: skip
        c.notes.append(f"Chance you beat {odds['opponent']} this week, by who starts: {line}.")
    if not c.close_call or rec not in pw or len(pw) < 2:
        return c
    best = max(pw, key=pw.get)
    if best == rec or pw[best] - pw[rec] < TIEBREAK_MIN:
        return c
    a, b = views[best], views[rec]
    favored = pw[rec] >= 0.5
    fa, fb = a.p25_if_active or 0, b.p25_if_active or 0
    ca, cb = a.p75_if_active or 0, b.p75_if_active or 0
    if favored and fa > fb:
        why = (
            f"you're favored, so the steadier score helps: {a.name}'s floor (25th "
            f"percentile) is {fa:.1f} vs {fb:.1f}"
        )
    elif not favored and ca > cb:
        why = (
            f"you're the underdog, so upside helps: {a.name}'s ceiling (75th "
            f"percentile) is {ca:.1f} vs {cb:.1f}"
        )
    else:
        why = (
            "the difference comes from how his game lines up with the rest of both "
            "lineups (players in the same NFL games rise and fall together)"
        )
    c.notes.append(
        f"Tiebreaker: start {a.name} over {b.name}. It's a coin flip on points, but your "
        f"chance of winning is {pw[best] * 100:.0f}% with {a.name} vs {pw[rec] * 100:.0f}% "
        f"with {b.name}; {why}."
    )
    c.recommendation = best
    c.players = [a] + [v for v in c.players if v.key != best]
    return c
