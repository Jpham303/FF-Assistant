"""Your chance of winning this week's fantasy matchup, with each start/sit candidate.

The opponent and his lineup come from Sleeper's matchups for the week (fetched when asked,
cached a minute): his set starters, or his best lineup by expected points if none are set.
Your side is your best lineup by expected points with the candidate forced in and the
other candidates benched. Both sides are summed simulation by simulation; the simulations
share every NFL game, so stacks and players facing each other are correlated as they are
on the field. Players whose games are over (kickoff + 4 hours) count with the points
Sleeper awarded instead of their projection.
"""

from __future__ import annotations

import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
from sqlalchemy import select

from ffa.db.models import LeagueRow, RosterEntryRow, TeamRow
from ffa.leagues.sleeper import SleeperClient
from ffa.scoring import score
from ffa.scoring.rules import ScoringRules
from ffa.service import startsit

CACHE_SECONDS = 60
GAME_LENGTH = timedelta(hours=4)  # after kickoff + this, a game's points are final
STARTING = {"BN", "IR", "TAXI"}  # non-starting slots
TIEBREAK_MIN = 0.01  # P(win) difference needed to override a close call

_cache: dict[tuple[str, int], tuple[float, list[dict]]] = {}


def fetch_matchups(platform_league_id: str, week: int,
                   client: SleeperClient | None = None) -> list[dict]:  # fmt: skip
    """Sleeper's matchups for the week (one small request, cached a minute)."""
    k = (platform_league_id, week)
    hit = _cache.get(k)
    if hit and time.monotonic() - hit[0] < CACHE_SECONDS:
        return hit[1]
    data = (client or SleeperClient()).get(f"league/{platform_league_id}/matchups/{week}") or []
    _cache[k] = (time.monotonic(), data)
    return data


def _forced(views: dict[str, startsit.PlayerView], key: str) -> dict[str, startsit.PlayerView]:
    return {k: replace(v, expected_points=1e9) if k == key else v for k, v in views.items()}


def win_odds(con, db, league: LeagueRow, season: int, week: int, candidates: list[str],
             rules: ScoringRules, *, matchups: list[dict] | None = None,
             now: datetime | None = None) -> dict:  # fmt: skip
    """{"available": False, "reason": ...} when it can't be computed, else the opponent,
    how his lineup was chosen, and per candidate: p_win and both sides' expected points."""
    now = now or datetime.now(UTC)
    if league.platform != "sleeper":
        return {"available": False, "reason": f"Matchups for {league.platform} not supported yet"}
    me = db.scalar(select(TeamRow).where(TeamRow.league_id == league.id, TeamRow.is_me))
    if me is None:
        return {"available": False, "reason": "Your team isn't identified in this league"}
    platform_id = league.id.split(":", 1)[1]
    try:
        ms = matchups if matchups is not None else fetch_matchups(platform_id, week)
    except Exception as exc:  # network or Sleeper down: start/sit still answers
        return {"available": False, "reason": f"Couldn't fetch this week's matchups: {exc}"}
    mine = next((m for m in ms if str(m.get("roster_id")) == me.platform_team_id), None)
    if mine is None or mine.get("matchup_id") is None:
        return {"available": False, "reason": "No fantasy matchup for your team this week"}
    opp_m = next((m for m in ms if m.get("matchup_id") == mine["matchup_id"]
                  and str(m.get("roster_id")) != me.platform_team_id), None)  # fmt: skip
    if opp_m is None:
        return {"available": False, "reason": "Opponent not found in this week's matchups"}
    opp = db.scalar(
        select(TeamRow).where(
            TeamRow.league_id == league.id, TeamRow.platform_team_id == str(opp_m["roster_id"])
        )
    )
    if opp is None:
        return {"available": False, "reason": "Opponent's roster isn't synced yet"}

    def entries(team: TeamRow) -> list[RosterEntryRow]:
        return list(db.scalars(select(RosterEntryRow).where(RosterEntryRow.team_id == team.id)))

    my_e, opp_e = entries(me), entries(opp)
    to_key = {e.platform_player_id: e.player_key for e in my_e + opp_e if e.player_key}
    my_pool = {e.player_key for e in my_e if e.player_key and e.slot_type != "ir"}
    starters = [p for p in opp_m.get("starters") or [] if p and p != "0"]
    opp_set = [to_key[p] for p in starters if p in to_key]
    opp_pool = {e.player_key for e in opp_e if e.player_key and e.slot_type != "ir"}
    slots = [s for s in league.lineup_slots if s not in STARTING]

    keys = sorted(my_pool | set(candidates) | opp_pool)
    w = startsit.load_week(con, season, week, keys)
    if w.sims.height == 0:
        return {"available": False, "reason": f"No projections for week {week}"}
    views = startsit.player_views(w, rules)
    pts = score(w.sims, rules).select("key", "sim", "fpts")
    mat = pts.pivot(on="key", index="sim", values="fpts").sort("sim").fill_null(0)
    n = mat.height

    # Games over: the points Sleeper awarded (all teams' players are in the matchups).
    awarded: dict[str, float] = {}
    for m in ms:
        for sid, p in (m.get("players_points") or {}).items():
            if sid in to_key:
                awarded[to_key[sid]] = float(p)
    teams = {v.key: v.team for v in views.values()}
    done = {k for k, t in teams.items()
            if (ko := w.kickoffs.get(t)) is not None and now >= ko + GAME_LENGTH}  # fmt: skip

    def column(k: str) -> np.ndarray:
        if k in done and k in awarded:
            return np.full(n, awarded[k])
        return mat[k].to_numpy() if k in mat.columns else np.zeros(n)  # no projection: 0

    def total(lineup: list[str]) -> np.ndarray:
        return sum((column(k) for k in lineup), np.zeros(n))

    if opp_set:
        opp_lineup, how = opp_set, "his lineup as set in Sleeper"
    else:
        best = startsit.best_lineup({k: v for k, v in views.items() if k in opp_pool}, slots)
        opp_lineup, how = [s["key"] for s in best if s["key"]], "his best lineup (none set)"
    opp_tot = total(opp_lineup)
    out = []
    for c in candidates:
        v = views.get(c)
        if v is None or v.play_probability < startsit.OUT_THRESHOLD:
            out.append({"key": c, "name": v.name if v else c, "p_win": None,
                        "note": "not playing"})  # fmt: skip
            continue
        pool = {k: views[k] for k in (my_pool | {c}) - (set(candidates) - {c}) if k in views}
        lineup = [s["key"] for s in startsit.best_lineup(_forced(pool, c), slots) if s["key"]]
        if c not in lineup:
            out.append({"key": c, "name": v.name, "p_win": None,
                        "note": "no lineup slot he is eligible for"})  # fmt: skip
            continue
        mine_tot = total(lineup)
        p = float(((mine_tot > opp_tot) + 0.5 * (mine_tot == opp_tot)).mean())
        out.append({"key": c, "name": v.name, "p_win": round(p, 3),
                    "my_expected": round(float(mine_tot.mean()), 1)})  # fmt: skip
    return {
        "available": True,
        "week": week,
        "opponent": opp.name,
        "opponent_lineup": how,
        "opponent_expected": round(float(opp_tot.mean()), 1),
        "final_games_counted": sorted({teams[k] for k in done if k in awarded}),
        "candidates": out,
        "unmatched_opponent_starters": len(starters) - len(opp_set),
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
