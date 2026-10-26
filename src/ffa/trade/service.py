"""Evaluate a trade end to end: league context, both sides, playoff odds before/after."""

from __future__ import annotations

from datetime import datetime

from ffa.db.models import LeagueRow
from ffa.scoring.rules import ScoringRules
from ffa.trade import context, core, playoffs, search


def _weekly(ctx: context.Context, rosters: dict[str, list[str]]) -> dict:
    out: dict[str, dict[int, tuple[float, float]]] = {}
    for tid, keys in rosters.items():
        out[tid] = {}
        for w in ctx.schedule:
            if w == ctx.current_week:
                out[tid][w] = ctx.current.get(tid, (0.0, 0.0))
            else:
                mu, var, _ = core.lineup(ctx.league, keys, w)
                out[tid][w] = (mu, var)
    return out


def evaluate_trade(con, db, row: LeagueRow, rules: ScoringRules, season: int, week: int,
                   give: list[str], get: list[str], playoff_weighting: bool = False,
                   now: datetime | None = None, fetch=None) -> dict:  # fmt: skip
    ctx = context.build(con, db, row, rules, season, week, now=now, fetch=fetch)
    lg = ctx.league
    mine = lg.teams[ctx.me]
    not_mine = [k for k in give if k not in mine.roster]
    if not_mine:
        raise ValueError(f"Not on your roster: {', '.join(_names(lg, not_mine))}")
    owner = {k: tid for tid, t in lg.teams.items() if tid != ctx.me for k in t.roster}
    unowned = [k for k in get if k not in owner]
    if unowned:
        raise ValueError(f"Not on another team's roster: {', '.join(_names(lg, unowned))}")
    partners = {owner[k] for k in get}
    if len(partners) != 1:
        raise ValueError("The players you'd get must all be on one team")
    partner = partners.pop()
    # Never value a traded player at 0 because he's missing from the projections.
    unprojected = [k for k in give + get if lg.weeks and not lg.players[k].weekly]
    if unprojected:
        raise LookupError(
            f"No rest-of-season projection for {', '.join(_names(lg, unprojected))}, so this "
            "trade can't be valued. Re-run ros_projections in Dagster; if he's still missing, "
            "it's a bug (not a 0-point player).")  # fmt: skip
    res = _evaluate(ctx, partner, give, get, playoff_weighting)
    res["scoring"], res["as_of"], res["notes"] = ctx.scoring, ctx.as_of, ctx.notes
    if res["playoff_odds"] is None:
        ctx.notes.append("Playoff odds unavailable: the league's playoff settings or schedule "
                         "aren't synced (run leagues_refresh).")  # fmt: skip
    return res


def _evaluate(ctx: context.Context, partner: str, give: list[str], get: list[str],
              playoff_weighting: bool) -> dict:  # fmt: skip
    """One trade, fully scored: both sides, how it looks, verdict, playoff odds."""
    lg = ctx.league
    res = core.evaluate(lg, ctx.me, partner, give, get, playoff_weighting)
    res["you"]["team_name"] = lg.teams[ctx.me].name
    res["them"]["team_name"] = lg.teams[partner].name
    if ctx.schedule and ctx.playoff_teams:
        rosters = {tid: t.roster for tid, t in lg.teams.items()}
        before = playoffs.odds(ctx.records, _weekly(ctx, rosters), ctx.schedule,
                               int(ctx.playoff_teams))  # fmt: skip
        rosters |= {ctx.me: res["you"]["roster_after"], partner: res["them"]["roster_after"]}
        after = playoffs.odds(ctx.records, _weekly(ctx, rosters), ctx.schedule,
                              int(ctx.playoff_teams))  # fmt: skip
        res["playoff_odds"] = {
            "you": {"before": before[ctx.me], "after": after[ctx.me]},
            "them": {"before": before[partner], "after": after[partner]},
            "regular_season_weeks_left": sorted(ctx.schedule),
        }  # fmt: skip
    else:
        res["playoff_odds"] = None
    res["give"], res["get"] = _names(lg, give), _names(lg, get)
    for side in ("you", "them"):
        res[side].pop("roster_after", None)
    return res


def trade_targets(con, db, row: LeagueRow, rules: ScoringRules, season: int, week: int,
                  give: list[str] | None = None, position: str | None = None,
                  partner: str | None = None, playoff_weighting: bool = False,
                  now: datetime | None = None, fetch=None) -> dict:  # fmt: skip
    """Target search: the best packages that raise your lineup and don't hurt theirs much,
    each scored like evaluate_trade."""
    ctx = context.build(con, db, row, rules, season, week, now=now, fetch=fetch)
    lg = ctx.league
    give = list(give or [])
    not_mine = [k for k in give if k not in lg.teams[ctx.me].roster]
    if not_mine:
        raise ValueError(f"Not on your roster: {', '.join(_names(lg, not_mine))}")
    unprojected = [k for k in give if lg.weeks and not lg.players[k].weekly]
    if unprojected:
        who = ", ".join(_names(lg, unprojected))
        raise LookupError(f"No rest-of-season projection for {who}; re-run ros_projections "
                          "in Dagster.")  # fmt: skip
    partners = None
    if partner:
        want = partner.lower()
        match = [t for t in lg.teams.values()
                 if t.id != ctx.me and want in (t.name.lower(), t.id.lower())]  # fmt: skip
        if not match:
            names = sorted(t.name for t in lg.teams.values() if t.id != ctx.me)
            raise ValueError(f"No team named {partner!r}. Teams: {', '.join(names)}")
        partners = [match[0].id]
    pos = position.upper() if position else None
    found = search.search(lg, ctx.me, give=give, position=pos, partners=partners,
                          playoff_weighting=playoff_weighting)  # fmt: skip
    needs = search.needs(lg)
    packages = []
    for p in found:
        res = _evaluate(ctx, p.partner, list(p.give), list(p.get), playoff_weighting)
        res["shape"] = p.tags or ["straight swap"]
        res["their_needs_filled"] = sorted(
            {lg.players[k].position for k in p.give if needs[p.partner].get(
                lg.players[k].position, 0) > 0})  # fmt: skip
        packages.append(res)
    mine = needs[ctx.me]
    return {
        "your_needs": {pos_: v for pos_, v in sorted(mine.items(), key=lambda kv: -kv[1])},
        "needs_note": "Points a week your starters at each position trail (+) or lead (-) the "
                      "league's middle team there, rest of season.",
        "packages": packages,
        "searched": {"give": _names(lg, give), "position": pos,
                     "partner": lg.teams[partners[0]].name if partners else "every team",
                     "rule": f"you gain at least {search.MY_MIN_GAIN} a week, they lose at most "
                             f"{search.THEIR_MAX_LOSS} a week; up to {search.MAX_SIDE} a side"},
        "weeks": lg.weeks, "scoring": ctx.scoring, "as_of": ctx.as_of, "notes": ctx.notes,
    }  # fmt: skip


def _names(lg: core.League, keys: list[str]) -> list[str]:
    return [lg.players[k].name if k in lg.players else k for k in keys]
