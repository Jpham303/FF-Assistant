"""Evaluate a trade end to end: league context, both sides, playoff odds before/after."""

from __future__ import annotations

from datetime import datetime

from ffa.db.models import LeagueRow
from ffa.scoring.rules import ScoringRules
from ffa.trade import context, core, playoffs


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
        ctx.notes.append("Playoff odds unavailable: the league's playoff settings or schedule "
                         "aren't synced (run leagues_refresh).")  # fmt: skip
    res["give"], res["get"] = _names(lg, give), _names(lg, get)
    res["scoring"], res["as_of"], res["notes"] = ctx.scoring, ctx.as_of, ctx.notes
    for side in ("you", "them"):
        res[side].pop("roster_after", None)
    return res


def _names(lg: core.League, keys: list[str]) -> list[str]:
    return [lg.players[k].name if k in lg.players else k for k in keys]
