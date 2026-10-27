"""Evaluate a trade end to end: league context, both sides, playoff odds before/after."""

from __future__ import annotations

from datetime import UTC, datetime

from ffa.db.models import LeagueRow
from ffa.scoring.rules import ScoringRules
from ffa.trade import context, core, packages, playoffs, search


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
    waivers = _apply_waivers(ctx, playoff_weighting, frozenset(give + get))
    res = _evaluate(ctx, partner, give, get, playoff_weighting)
    res["waiver_baseline"] = _waiver_dict(ctx, waivers, {ctx.me, partner})
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
    waivers = _apply_waivers(ctx, playoff_weighting, frozenset(give))
    lg = ctx.league
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
        "waiver_baseline": _waiver_dict(ctx, waivers, {ctx.me}),
        "searched": {"give": _names(lg, give), "position": pos,
                     "partner": lg.teams[partners[0]].name if partners else "every team",
                     "rule": f"you gain at least {search.MY_MIN_GAIN} a week, they lose at most "
                             f"{search.THEIR_MAX_LOSS} a week; up to {search.MAX_SIDE} a side"},
        "weeks": lg.weeks, "scoring": ctx.scoring, "as_of": ctx.as_of, "notes": ctx.notes,
    }  # fmt: skip


WAIVER_NOTE = (
    "Scored after the free waiver moves each team could already make (best free agent added, "
    "least valuable player dropped, when it adds at least {min} points a week): a trade gets "
    "credit only for what it adds beyond those."
)


def _apply_waivers(ctx: context.Context, playoff_weighting: bool,
                   protect: frozenset[str]) -> list[core.Waiver]:  # fmt: skip
    """Swap ctx.league for the league after every team's helpful waiver moves, worst
    record first (waiver priority)."""
    order = sorted(ctx.league.teams, key=lambda t: ctx.records.get(t, (0.0, 0.0)))
    ctx.league, moves = core.with_waivers(ctx.league, playoff_weighting, protect, order)
    return moves


def _waiver_dict(ctx: context.Context, moves: list[core.Waiver], teams: set[str]) -> dict:
    lg = ctx.league
    return {
        "note": WAIVER_NOTE.format(min=core.WAIVER_MIN),
        "moves": [{"team": lg.teams[m.team].name,
                   "add": m.add.name, "add_position": m.add.position,
                   "points_added": round(m.add.points, 1),
                   "drop": m.drop.name if m.drop else None}
                  for m in moves if m.team in teams],
    }  # fmt: skip


def _names(lg: core.League, keys: list[str]) -> list[str]:
    return [lg.players[k].name if k in lg.players else k for k in keys]


def trade_packages(con, db, row: LeagueRow, rules: ScoringRules, season: int, week: int,
                   playoff_weighting: bool = False, partner: str | None = None,
                   now: datetime | None = None, fetch=None) -> dict:  # fmt: skip
    """The package generator: for every opponent (or `partner`), the best offer in each
    acceptance grade where you gain, each scored like evaluate_trade, with reasons for
    both sides and what the other manager sees."""
    t0 = datetime.now(UTC)
    ctx = context.build(con, db, row, rules, season, week, now=now, fetch=fetch)
    lg = ctx.league
    partners = None
    if partner:
        want = partner.lower()
        match = [t for t in lg.teams.values()
                 if t.id != ctx.me and want in (t.name.lower(), t.id.lower())]  # fmt: skip
        if not match:
            names = sorted(t.name for t in lg.teams.values() if t.id != ctx.me)
            raise ValueError(f"No team named {partner!r}. Teams: {', '.join(names)}")
        partners = [match[0].id]
    waivers = _apply_waivers(ctx, playoff_weighting, frozenset())
    lg = ctx.league
    team_needs = search.needs(lg)
    offers = packages.generate(lg, ctx.me, playoff_weighting=playoff_weighting,
                               partners=partners)  # fmt: skip
    by_team, flat = [], []
    for tid, offs in offers.items():
        items = []
        for o in offs:
            p = o.package
            res = _evaluate(ctx, tid, list(p.give), list(p.get), playoff_weighting)
            you, them = packages.reasons(lg, o, res, team_needs[tid])
            item = {
                "grade": o.grade, "partner": lg.teams[tid].name,
                "give": res["give"], "get": res["get"],
                "you_per_week": round(o.mine, 2), "them_per_week": round(o.theirs, 2),
                "you_playoff_weeks": res["you"]["playoff_weeks_change"],
                "reasons_you": you, "reasons_them": them,
                "what_they_see": _seen(lg, p.give, p.get, o.looks),
                "shape": p.tags or ["straight swap"],
                "playoff_odds": res["playoff_odds"],
                "drops_you": res["you"]["drops"], "adds_you": res["you"]["adds"],
                "drops_them": res["them"]["drops"], "adds_them": res["them"]["adds"],
                "verdict": res["verdict"], "p_ahead": res["p_ahead"],
            }  # fmt: skip
            items.append(item)
            flat.append(item)
        by_team.append({"team": lg.teams[tid].name, "packages": items})
    order = {g: i for i, g in enumerate(packages.GRADES)}
    flat.sort(key=lambda x: (order[x["grade"]], -x["you_per_week"]))
    return {
        "grades": {
            "Likely": f"their lineup gains at least {packages.LIKELY_THEIR_GAIN} a week and it "
                      "looks even or better for them",
            "Fair": "both lineups gain",
            "Stretch": f"they lose up to {packages.STRETCH_MAX_LOSS} a week but it looks even "
                       f"or better for them; you gain at least {packages.STRETCH_MY_GAIN}",
            "Long shot": f"they lose up to {packages.LONG_SHOT_MAX_LOSS} a week; only what "
                         "they see (a name, a hot streak) sells it; you gain at least "
                         f"{packages.LONG_SHOT_MY_GAIN}",
        },
        "grade_note": "A scored grade, not a probability: there's no data on offers turned "
                      "down. Completed trades in the league are logged to check it later.",
        "by_team": by_team,
        "league_wide": flat,
        "waiver_baseline": _waiver_dict(ctx, waivers, {ctx.me}),
        "weeks": lg.weeks, "scoring": ctx.scoring, "as_of": ctx.as_of, "notes": ctx.notes,
        "generated_at": t0.isoformat(),
        "seconds": round((datetime.now(UTC) - t0).total_seconds(), 1),
    }  # fmt: skip


def _seen(lg: core.League, give, get, looks: float) -> dict:
    """What the other manager sees, per player, and the verdict on it."""

    def card(k: str) -> dict:
        p = lg.players[k]
        return {"name": p.name, "position": p.position, "ppg_so_far": p.ppg,
                "last3": p.last3, "season_rank": p.season_rank, "draft_pick": p.draft_pick,
                "injury": p.injury,
                "perceived_per_game": round(packages.perceived(p), 1)}  # fmt: skip

    return {"they_get": [card(k) for k in give], "they_give": [card(k) for k in get],
            "looks_for_them": round(looks, 2),
            "reads_as": "better for them" if looks > packages.LOOKS_EVEN
            else "even" if looks >= -packages.LOOKS_EVEN else "worse for them"}  # fmt: skip


DATASET = "trade_packages"  # precomputed daily after rest of season


def save_all(warehouse_dir, season: int, week: int, payloads: dict[str, dict],
             now: datetime | None = None) -> None:  # fmt: skip
    """Every league's run in one write (a week's latest write is what's read back)."""
    import json

    import polars as pl

    from ffa.warehouse import store

    texts = [json.dumps(p, default=str) for p in payloads.values()]
    df = pl.DataFrame({"league": list(payloads), "season": [season] * len(payloads),
                       "week": [week] * len(payloads), "payload": texts})  # fmt: skip
    store.write(df, warehouse_dir, DATASET, volatile=True, ingested_at=now or datetime.now(UTC))


def saved_packages(con, league_id: str, season: int, week: int) -> dict | None:
    """The latest precomputed run for this league and week, or None."""
    import json

    from ffa.warehouse import duck

    try:
        t = duck.week(con, DATASET, season, week)
    except duck.MissingPartition:
        return None
    rows = con.sql(f"select payload, _ingested_at from {t} where league = ? "
                   "order by _ingested_at desc limit 1", params=[league_id]).fetchall()  # fmt: skip
    if not rows:
        return None
    return json.loads(rows[0][0]) | {"precomputed_at": str(rows[0][1])}
